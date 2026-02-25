"""Training entry point for MetaFormer."""

import os
import sys
import yaml
import random
import argparse
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' not in os.environ:
        return 0, 1, False
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    dist.init_process_group('nccl')
    torch.cuda.set_device(local_rank)
    return rank, world_size, True


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def build_distributed_loaders(config, rank, world_size, is_distributed):
    """Build data loaders with optional distributed sampling."""
    from data_preparation.ecg_dataset import build_dataloader

    if is_distributed:
        from torch.utils.data import DataLoader, DistributedSampler
        from data_preparation.ecg_dataset import (
            build_dataloader as _build,
            PTBXLDataset, CSNDataset, CombinedECGDataset, collate_fn
        )
        data_cfg = config.get('data', {})
        exp_cfg = config.get('experiment', {})
        dataset_name = data_cfg.get('dataset', 'ptbxl')
        signal_length = data_cfg.get('signal_length', 512)

        datasets_train, datasets_val = [], []
        for name in (['ptbxl'] if dataset_name == 'ptbxl' else
                     ['csn'] if dataset_name == 'csn' else ['ptbxl', 'csn']):
            cls = PTBXLDataset if name == 'ptbxl' else CSNDataset
            path = data_cfg.get(f'{name}_path', f'data/{name}')
            if os.path.exists(path):
                datasets_train.append(cls(path, 'train', signal_length))
                datasets_val.append(cls(path, 'val', signal_length))

        train_ds = CombinedECGDataset(datasets_train) if len(datasets_train) > 1 else datasets_train[0]
        val_ds = CombinedECGDataset(datasets_val) if len(datasets_val) > 1 else datasets_val[0]

        train_sampler = DistributedSampler(train_ds, world_size, rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, world_size, rank, shuffle=False)

        train_loader = DataLoader(
            train_ds, batch_size=data_cfg.get('batch_size', 32),
            sampler=train_sampler, num_workers=exp_cfg.get('num_workers', 4),
            pin_memory=True, drop_last=True, collate_fn=collate_fn)
        val_loader = DataLoader(
            val_ds, batch_size=data_cfg.get('batch_size', 32),
            sampler=val_sampler, num_workers=exp_cfg.get('num_workers', 4),
            pin_memory=True, drop_last=False, collate_fn=collate_fn)
        return train_loader, val_loader
    else:
        train_loader = build_dataloader(config, 'train')
        val_loader = build_dataloader(config, 'val')
        return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='MetaFormer Training')
    parser.add_argument('--config', type=str, default='config/metaformer.yaml')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("MetaFormer Training")
    print("=" * 60)

    config = load_config(args.config)
    seed = config.get('experiment', {}).get('seed', 42)
    set_seed(seed)

    rank, world_size, is_distributed = setup_distributed()
    train_loader, val_loader = build_distributed_loaders(config, rank, world_size, is_distributed)

    from model.condition_unet_recon import build_model
    model = build_model(config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    output_dir = config.get('experiment', {}).get('output_dir', 'outputs/metaformer')
    os.makedirs(output_dir, exist_ok=True)

    if is_distributed:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        model = model.to(local_rank)
        model = DDP(model, device_ids=[local_rank])

    from experiment.trainer import build_trainer
    trainer = build_trainer(model, train_loader, val_loader, config, output_dir, args.resume)
    trainer.train()

    if is_distributed:
        cleanup_distributed()

    print("\nTraining completed!")


if __name__ == '__main__':
    main()

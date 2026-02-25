"""Evaluation entry point for MetaFormer."""

import os
import sys
import yaml
import random
import argparse
import numpy as np
import torch
from pathlib import Path


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_model(config: dict, checkpoint_path: str, device: torch.device):
    """Load a trained MetaFormer model from checkpoint."""
    from model.condition_unet_recon import build_model

    model = build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='MetaFormer Evaluation')
    parser.add_argument('--config', type=str, default='config/metaformer.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['ptbxl', 'csn', 'all'])
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("MetaFormer Evaluation")
    print("=" * 60)

    config = load_config(args.config)
    seed = config.get('experiment', {}).get('seed', 42)
    set_seed(seed)

    if args.dataset:
        config['data']['dataset'] = args.dataset

    device = torch.device(config.get('experiment', {}).get('device', 'cuda'))

    # Determine checkpoint path
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        output_dir = config.get('experiment', {}).get('output_dir', 'outputs/metaformer')
        checkpoint_path = os.path.join(output_dir, 'checkpoints', 'best.pth')

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Checkpoint: {checkpoint_path}")

    model = load_model(config, checkpoint_path, device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e6:.2f}M")

    # Build test data loader
    from data_preparation.ecg_dataset import build_dataloader
    eval_batch_size = config.get('evaluation', {}).get('batch_size', 512)
    test_loader = build_dataloader(config, 'test', shuffle=False, batch_size=eval_batch_size)
    print(f"Test samples: {len(test_loader.dataset)}")

    # Run evaluation
    eval_output_dir = args.output_dir or config.get('experiment', {}).get(
        'output_dir', 'outputs/metaformer')
    from experiment.evaluator import Evaluator
    evaluator = Evaluator(model, test_loader, config, eval_output_dir)
    evaluator.evaluate()

    print("\nEvaluation completed!")


if __name__ == '__main__':
    main()

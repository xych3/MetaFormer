"""Unified CLI entry point for MetaFormer (prepare / train / eval)."""

import os
import sys
import yaml
import random
import argparse
import numpy as np
import torch
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


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


def prepare_data(args):
    """Run data preprocessing pipelines."""
    if args.dataset in ['ptbxl', 'all']:
        from data_preparation.prepare_ptbxl import main as prep_ptbxl
        print("Preparing PTB-XL dataset...")
        prep_ptbxl(
            path=args.data_root or os.path.expanduser("~/ECG_Dataset/ptb-xl"),
            output_dir=args.output or 'data/ptbxl',
            target_length=args.target_length,
            n_processes=args.n_processes,
        )

    if args.dataset in ['csn', 'all']:
        from data_preparation.prepare_csn import main as prep_csn
        print("Preparing CSN dataset...")
        prep_csn(
            path=args.data_root or os.path.expanduser("~/ECG_Dataset/csn"),
            output_dir=args.output or 'data/csn',
            target_length=args.target_length,
            n_processes=args.n_processes,
        )


def train_model(args):
    """Run MetaFormer training."""
    from train import main as train_main
    sys.argv = ['train.py', '--config', args.config]
    if args.resume:
        sys.argv += ['--resume', args.resume]
    train_main()


def evaluate_model(args):
    """Run MetaFormer evaluation."""
    from eval import main as eval_main
    sys.argv = ['eval.py', '--config', args.config]
    if args.checkpoint:
        sys.argv += ['--checkpoint', args.checkpoint]
    if args.dataset:
        sys.argv += ['--dataset', args.dataset]
    eval_main()


def show_info():
    """Display framework information."""
    print("=" * 60)
    print("MetaFormer: Metadata-Guided Transformer for")
    print("Patient-Aware 12-Lead ECG Reconstruction")
    print("=" * 60)
    print()
    print("Commands:")
    print("  prepare  - Preprocess ECG datasets (PTB-XL, CSN)")
    print("  train    - Train the MetaFormer model")
    print("  eval     - Evaluate a trained model")
    print("  info     - Show this help message")
    print()
    print("Examples:")
    print("  python run.py prepare --dataset all")
    print("  python run.py train --config config/metaformer.yaml")
    print("  python run.py eval --checkpoint outputs/best.pth")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='MetaFormer CLI')
    subparsers = parser.add_subparsers(dest='command')

    # Prepare
    prep_parser = subparsers.add_parser('prepare', help='Preprocess datasets')
    prep_parser.add_argument('--dataset', type=str, default='all',
                             choices=['ptbxl', 'csn', 'all'])
    prep_parser.add_argument('--data-root', type=str, default=None)
    prep_parser.add_argument('--output', type=str, default=None)
    prep_parser.add_argument('--target-length', type=int, default=512)
    prep_parser.add_argument('--n-processes', type=int, default=None)

    # Train
    train_parser = subparsers.add_parser('train', help='Train MetaFormer')
    train_parser.add_argument('--config', type=str, default='config/metaformer.yaml')
    train_parser.add_argument('--resume', type=str, default=None)

    # Eval
    eval_parser = subparsers.add_parser('eval', help='Evaluate MetaFormer')
    eval_parser.add_argument('--config', type=str, default='config/metaformer.yaml')
    eval_parser.add_argument('--checkpoint', type=str, default=None)
    eval_parser.add_argument('--dataset', type=str, default=None,
                             choices=['ptbxl', 'csn', 'all'])

    # Info
    subparsers.add_parser('info', help='Show framework information')

    args = parser.parse_args()

    if args.command is None or args.command == 'info':
        show_info()
    elif args.command == 'prepare':
        prepare_data(args)
    elif args.command == 'train':
        train_model(args)
    elif args.command == 'eval':
        evaluate_model(args)


if __name__ == '__main__':
    main()

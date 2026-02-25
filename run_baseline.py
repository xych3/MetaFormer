import os
import sys
import yaml
import random
import logging
import argparse
import numpy as np
import torch
from pathlib import Path


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# ==================== Training entry points ====================

def train_mecgnet(config_path: str):
    """Train mEcgNet model."""
    from experiment.trainer_mecgnet import train_mecgnet as _train
    config = load_config(config_path)
    set_seed(config.get('seed', 42))
    _train(config)


def train_ekgan(config_path: str):
    """Train EKGAN model."""
    from experiment.trainer_ekgan import train_ekgan as _train
    config = load_config(config_path)
    set_seed(config.get('seed', 42))
    _train(config)


def train_ecgrecover(config_path: str):
    """Train ECGRecover model."""
    from experiment.trainer_ecgrecover import train_ecgrecover as _train
    config = load_config(config_path)
    set_seed(config.get('seed', 42))
    _train(config)


# ==================== Evaluation entry points ====================

def eval_mecgnet(config_path: str, checkpoint_path: str = None,
                 dataset: str = None, num_samples: int = None):
    """Evaluate mEcgNet model."""
    from experiment.evaluator_mecgnet import evaluate_mecgnet as _eval
    config = load_config(config_path)
    set_seed(config.get('seed', 42))
    _eval(config, checkpoint_path=checkpoint_path,
          dataset=dataset, num_samples=num_samples)


def eval_ekgan(config_path: str, checkpoint_path: str = None,
               dataset: str = None, num_samples: int = None):
    """Evaluate EKGAN model."""
    from experiment.evaluator_ekgan import evaluate_ekgan as _eval
    config = load_config(config_path)
    set_seed(config.get('seed', 42))
    _eval(config, checkpoint_path=checkpoint_path,
          dataset=dataset, num_samples=num_samples)


def eval_ecgrecover(config_path: str, checkpoint_path: str = None,
                    dataset: str = None, num_samples: int = None):
    """Evaluate ECGRecover model."""
    from experiment.evaluator_ecgrecover import evaluate_ecgrecover as _eval
    config = load_config(config_path)
    set_seed(config.get('seed', 42))
    _eval(config, checkpoint_path=checkpoint_path,
          dataset=dataset, num_samples=num_samples)


def show_info():
    """Display system and environment information."""
    print("=" * 60)
    print("ECG Reconstruction - Baseline Models")
    print("=" * 60)
    print(f"Python:  {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA:    {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    print("Supported models: mecgnet, ekgan, ecgrecover")
    print("=" * 60)


def main():
    """Main entry point for baseline model training and evaluation."""
    parser = argparse.ArgumentParser(
        description='ECG Reconstruction - Baseline Models')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'eval', 'info'],
                        help='Run mode: train, eval, or info.')
    parser.add_argument('--model', type=str,
                        choices=['mecgnet', 'ekgan', 'ecgrecover'],
                        help='Baseline model name.')
    parser.add_argument('--config', type=str,
                        help='Path to configuration file.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file (for eval mode).')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['ptbxl', 'csn', 'all'],
                        help='Dataset to evaluate on.')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to evaluate.')
    args = parser.parse_args()

    if args.mode == 'info':
        show_info()
        return

    if not args.model or not args.config:
        parser.error("--model and --config are required for train/eval mode.")

    # Training
    if args.mode == 'train':
        train_funcs = {
            'mecgnet': train_mecgnet,
            'ekgan': train_ekgan,
            'ecgrecover': train_ecgrecover,
        }
        train_funcs[args.model](args.config)

    # Evaluation
    elif args.mode == 'eval':
        eval_funcs = {
            'mecgnet': eval_mecgnet,
            'ekgan': eval_ekgan,
            'ecgrecover': eval_ecgrecover,
        }
        eval_funcs[args.model](
            args.config,
            checkpoint_path=args.checkpoint,
            dataset=args.dataset,
            num_samples=args.num_samples,
        )


if __name__ == '__main__':
    main()

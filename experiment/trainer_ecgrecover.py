"""Trainer class for ECGRecover baseline model."""

import os
import time
import json
import logging
from datetime import datetime
from typing import Optional, Dict
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from baseline.ecgrecover import ECGRecover, build_ecgrecover, MSEPearsonLoss
from data_preparation.ecg_dataset import build_dataloader


class ECGRecoverTrainer:
    """Trainer for ECGRecover model."""

    def __init__(
        self,
        model: ECGRecover,
        config: dict,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        resume_from: Optional[str] = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.exp_config = config.get('experiment', {})
        self.train_config = config.get('training', {})
        self.data_config = config.get('data', {})

        self.device = torch.device(self.exp_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = self.model.to(self.device)

        self.num_epochs = self.train_config.get('num_epochs', 10)

        # Lead configuration
        self.condition_leads = self.data_config.get('condition_leads', [0])
        self.target_leads = [i for i in range(12) if i not in self.condition_leads]

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # Loss function
        loss_config = self.train_config.get('loss', {})
        pearson_weight = loss_config.get('pearson_weight', 0.1)
        self.criterion = MSEPearsonLoss(pearson_weight=pearson_weight)

        # Directories
        self.output_dir = Path(self.exp_config.get('output_dir', 'outputs_baseline/ecgrecover'))
        self.log_dir = self.output_dir / self.exp_config.get('log_dir', 'logs')
        self.checkpoint_dir = self.output_dir / self.exp_config.get('checkpoint_dir', 'checkpoints')

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.train_correlations = []
        self.val_correlations = []

        if resume_from:
            self._load_checkpoint(resume_from)

        self._save_config()

        self.logger.info(f"ECGRecover Trainer initialized on device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        self.logger.info(f"Training samples: {len(train_loader.dataset)}")
        if val_loader:
            self.logger.info(f"Validation samples: {len(val_loader.dataset)}")

    def _setup_logging(self):
        """Set up logging."""
        self.logger = logging.getLogger('ECGRecover')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir / f'train_{timestamp}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer."""
        opt_config = self.train_config.get('optimizer', {})
        return Adam(
            self.model.parameters(),
            lr=opt_config.get('lr', 1e-2),
            betas=tuple(opt_config.get('betas', [0.9, 0.999])),
            weight_decay=opt_config.get('weight_decay', 0.0),
        )

    def _build_scheduler(self):
        """Build learning rate scheduler."""
        sched_config = self.train_config.get('scheduler', {})
        return ReduceLROnPlateau(
            self.optimizer,
            mode=sched_config.get('mode', 'min'),
            factor=sched_config.get('factor', 0.1),
            patience=sched_config.get('patience', 5),
            verbose=True,
        )

    def _save_config(self):
        """Save configuration."""
        config_path = self.output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)

    def _save_checkpoint(self, filename: str, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_correlations': self.train_correlations,
            'val_correlations': self.val_correlations,
            'config': self.config,
        }

        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")

        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model: {best_path}")

    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        self.logger.info(f"Loading checkpoint from: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_correlations = checkpoint.get('train_correlations', [])
        self.val_correlations = checkpoint.get('val_correlations', [])

        self.logger.info(f"Resumed from epoch {self.current_epoch}")

    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()

        epoch_losses = []
        epoch_correlations = []
        epoch_mses = []

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs}")
        log_every_n_steps = self.train_config.get('logging', {}).get('log_every_n_steps', 50)

        for batch_idx, batch in enumerate(progress_bar):
            signal = batch['signal'].to(self.device)

            lead_i = signal[:, self.condition_leads, :]
            target = signal

            output = self.model(lead_i)
            loss, pearson, mse = self.criterion(output, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.global_step += 1

            loss_value = loss.item()
            pearson_value = pearson.item()
            mse_value = mse.item()

            epoch_losses.append(loss_value)
            epoch_correlations.append(pearson_value)
            epoch_mses.append(mse_value)

            progress_bar.set_postfix({
                'loss': f'{loss_value:.6f}',
                'corr': f'{pearson_value:.4f}',
                'mse': f'{mse_value:.6f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })

            if self.global_step % log_every_n_steps == 0:
                self.logger.info(
                    f"Step {self.global_step} | Loss: {loss_value:.6f} | "
                    f"Corr: {pearson_value:.4f} | MSE: {mse_value:.6f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )

        avg_loss = np.mean(epoch_losses)
        avg_corr = np.mean(epoch_correlations)
        avg_mse = np.mean(epoch_mses)

        self.train_losses.append(avg_loss)
        self.train_correlations.append(avg_corr)

        return {'loss': avg_loss, 'correlation': avg_corr, 'mse': avg_mse}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        if self.val_loader is None:
            return {'loss': float('inf'), 'correlation': 0.0, 'mse': float('inf')}

        self.model.eval()

        val_losses = []
        val_correlations = []
        val_mses = []

        for batch in tqdm(self.val_loader, desc="Validation"):
            signal = batch['signal'].to(self.device)
            lead_i = signal[:, self.condition_leads, :]
            target = signal

            output = self.model(lead_i)
            loss, pearson, mse = self.criterion(output, target)

            val_losses.append(loss.item())
            val_correlations.append(pearson.item())
            val_mses.append(mse.item())

        avg_loss = np.mean(val_losses)
        avg_corr = np.mean(val_correlations)
        avg_mse = np.mean(val_mses)

        self.val_losses.append(avg_loss)
        self.val_correlations.append(avg_corr)

        return {'loss': avg_loss, 'correlation': avg_corr, 'mse': avg_mse}

    def train(self):
        """Full training loop."""
        self.logger.info("=" * 60)
        self.logger.info("Starting ECGRecover training...")
        self.logger.info(f"Total epochs: {self.num_epochs}")
        self.logger.info(f"Batch size: {self.train_loader.batch_size}")
        self.logger.info("=" * 60)

        ckpt_config = self.train_config.get('checkpoint', {})
        save_every_n_epochs = ckpt_config.get('save_every_n_epochs', 5)
        save_best = ckpt_config.get('save_best', True)
        save_latest = ckpt_config.get('save_latest', True)

        val_config = self.train_config.get('validation', {})
        val_enabled = val_config.get('enabled', True) and self.val_loader is not None
        val_every_n_epochs = val_config.get('every_n_epochs', 1)

        start_time = time.time()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            train_metrics = self.train_epoch()
            epoch_time = time.time() - epoch_start_time

            self.logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs} completed | "
                f"Train Loss: {train_metrics['loss']:.6f} | "
                f"Train Corr: {train_metrics['correlation']:.4f} | "
                f"Train MSE: {train_metrics['mse']:.6f} | "
                f"Time: {epoch_time:.1f}s"
            )

            val_loss = float('inf')
            if val_enabled and (epoch + 1) % val_every_n_epochs == 0:
                val_metrics = self.validate()
                val_loss = val_metrics['loss']

                self.logger.info(
                    f"Validation Loss: {val_metrics['loss']:.6f} | "
                    f"Validation Corr: {val_metrics['correlation']:.4f} | "
                    f"Validation MSE: {val_metrics['mse']:.6f}"
                )

                self.scheduler.step(val_loss)

                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    if save_best:
                        self._save_checkpoint('best.pth', is_best=True)
            else:
                self.scheduler.step(train_metrics['loss'])

            if (epoch + 1) % save_every_n_epochs == 0:
                self._save_checkpoint(f'epoch_{epoch + 1}.pth')

            if save_latest:
                self._save_checkpoint('latest.pth')

        total_time = time.time() - start_time
        self.logger.info("=" * 60)
        self.logger.info(f"Training completed in {total_time / 3600:.2f} hours")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        self.logger.info("=" * 60)

        self._save_checkpoint('final.pth')

        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_correlations': self.train_correlations,
            'val_correlations': self.val_correlations,
            'best_val_loss': self.best_val_loss,
            'total_epochs': self.current_epoch + 1,
            'total_time': total_time,
        }
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        return history


def train_ecgrecover(config: dict, resume_from: Optional[str] = None):
    """ECGRecover training entry point."""
    model = build_ecgrecover(config)
    train_loader = build_dataloader(config, split='train')
    val_loader = build_dataloader(config, split='val')

    trainer = ECGRecoverTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        resume_from=resume_from,
    )

    history = trainer.train()
    return history

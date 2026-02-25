"""Trainer class for EKGAN baseline model."""

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
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from baseline.ekgan import EKGAN, build_ekgan
from data_preparation.ecg_dataset import build_dataloader


class EKGANTrainer:
    """Trainer for EKGAN model."""

    def __init__(
        self,
        model: EKGAN,
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

        # Loss weights
        loss_config = self.train_config.get('loss', {})
        self.lambda_l1 = loss_config.get('lambda_l1', 50.0)
        self.alpha_lv = loss_config.get('alpha_lv', 1.0)

        # Lead configuration
        self.condition_leads = self.data_config.get('condition_leads', [0])
        self.target_leads = [i for i in range(12) if i not in self.condition_leads]

        # Optimizers and schedulers
        self.optimizer_ig, self.optimizer_lg, self.optimizer_d = self._build_optimizers()
        self.scheduler_ig, self.scheduler_lg, self.scheduler_d = self._build_schedulers()

        # Loss functions
        self.criterion_bce = nn.BCEWithLogitsLoss()
        self.criterion_l1 = nn.L1Loss()

        # Directories
        self.output_dir = Path(self.exp_config.get('output_dir', 'outputs_baseline/ekgan'))
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
        self.train_losses = {'ig': [], 'lg': [], 'd': []}
        self.val_losses = []

        if resume_from:
            self._load_checkpoint(resume_from)

        self._save_config()

        self.logger.info(f"EKGAN Trainer initialized on device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        self.logger.info(f"Training samples: {len(train_loader.dataset)}")
        if val_loader:
            self.logger.info(f"Validation samples: {len(val_loader.dataset)}")

    def _setup_logging(self):
        """Set up logging."""
        self.logger = logging.getLogger('EKGAN')
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

    def _build_optimizers(self):
        """Build optimizers for all sub-networks."""
        opt_config = self.train_config.get('optimizer', {})
        lr = opt_config.get('lr', 1e-4)
        betas = tuple(opt_config.get('betas', [0.5, 0.999]))

        optimizer_ig = Adam(self.model.inference_generator.parameters(), lr=lr, betas=betas)
        optimizer_lg = Adam(self.model.label_generator.parameters(), lr=lr, betas=betas)
        optimizer_d = Adam(self.model.discriminator.parameters(), lr=lr, betas=betas)

        return optimizer_ig, optimizer_lg, optimizer_d

    def _build_schedulers(self):
        """Build learning rate schedulers."""
        sched_config = self.train_config.get('scheduler', {})
        decay_epoch_start = sched_config.get('decay_epoch_start', 6)
        decay_rate = sched_config.get('decay_rate', 0.95)
        steps_per_epoch = len(self.train_loader)

        def lr_lambda(step):
            epoch = step // steps_per_epoch
            if epoch < decay_epoch_start - 1:
                return 1.0
            else:
                decays = epoch - (decay_epoch_start - 1)
                return decay_rate ** decays

        scheduler_ig = LambdaLR(self.optimizer_ig, lr_lambda)
        scheduler_lg = LambdaLR(self.optimizer_lg, lr_lambda)
        scheduler_d = LambdaLR(self.optimizer_d, lr_lambda)

        return scheduler_ig, scheduler_lg, scheduler_d

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
            'optimizer_ig_state_dict': self.optimizer_ig.state_dict(),
            'optimizer_lg_state_dict': self.optimizer_lg.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'scheduler_ig_state_dict': self.scheduler_ig.state_dict(),
            'scheduler_lg_state_dict': self.scheduler_lg.state_dict(),
            'scheduler_d_state_dict': self.scheduler_d.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
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
        self.optimizer_ig.load_state_dict(checkpoint['optimizer_ig_state_dict'])
        self.optimizer_lg.load_state_dict(checkpoint['optimizer_lg_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.scheduler_ig.load_state_dict(checkpoint['scheduler_ig_state_dict'])
        self.scheduler_lg.load_state_dict(checkpoint['scheduler_lg_state_dict'])
        self.scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])

        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.train_losses = checkpoint.get('train_losses', {'ig': [], 'lg': [], 'd': []})
        self.val_losses = checkpoint.get('val_losses', [])

        self.logger.info(f"Resumed from epoch {self.current_epoch}")

    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()

        epoch_losses = {'ig_total': [], 'ig_adv': [], 'ig_l1': [], 'lg': [], 'd': [], 'lv': []}
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs}")

        log_every_n_steps = self.train_config.get('logging', {}).get('log_every_n_steps', 50)

        for batch_idx, batch in enumerate(progress_bar):
            signal = batch['signal'].to(self.device)

            lead_i = signal[:, self.condition_leads, :]
            target = signal

            outputs = self.model.train_step(lead_i, target)

            ig_output = outputs['ig_output']
            ig_latent = outputs['ig_latent']
            lg_output = outputs['lg_output']
            lg_latent = outputs['lg_latent']
            target_padded = outputs['target_padded']
            input_padded = outputs['input_padded']

            # Train label generator
            self.optimizer_lg.zero_grad()
            lg_loss = self.criterion_l1(lg_output, input_padded)
            lg_loss.backward(retain_graph=True)
            self.optimizer_lg.step()

            # Train discriminator
            self.optimizer_d.zero_grad()
            disc_real = self.model.discriminator(input_padded, target_padded)
            disc_fake = self.model.discriminator(input_padded, ig_output.detach())

            real_loss = self.criterion_bce(disc_real, torch.ones_like(disc_real))
            fake_loss = self.criterion_bce(disc_fake, torch.zeros_like(disc_fake))
            d_loss = real_loss + fake_loss
            d_loss.backward()
            self.optimizer_d.step()

            # Train inference generator
            self.optimizer_ig.zero_grad()
            ig_output, ig_latent = self.model.inference_generator(input_padded)
            _, lg_latent = self.model.label_generator(input_padded)
            disc_fake = self.model.discriminator(input_padded, ig_output)

            ig_adv_loss = self.criterion_bce(disc_fake, torch.ones_like(disc_fake))
            ig_l1_loss = self.criterion_l1(ig_output, target_padded)
            lv_loss = self.criterion_l1(ig_latent, lg_latent.detach())

            ig_total_loss = ig_adv_loss + self.lambda_l1 * ig_l1_loss + self.alpha_lv * lv_loss
            ig_total_loss.backward()
            self.optimizer_ig.step()

            # Update learning rates
            self.scheduler_ig.step()
            self.scheduler_lg.step()
            self.scheduler_d.step()

            self.global_step += 1

            epoch_losses['ig_total'].append(ig_total_loss.item())
            epoch_losses['ig_adv'].append(ig_adv_loss.item())
            epoch_losses['ig_l1'].append(ig_l1_loss.item())
            epoch_losses['lg'].append(lg_loss.item())
            epoch_losses['d'].append(d_loss.item())
            epoch_losses['lv'].append(lv_loss.item())

            progress_bar.set_postfix({
                'IG': f'{ig_total_loss.item():.4f}',
                'D': f'{d_loss.item():.4f}',
                'LG': f'{lg_loss.item():.4f}',
            })

            if self.global_step % log_every_n_steps == 0:
                self.logger.info(
                    f"Step {self.global_step} | "
                    f"IG_total: {ig_total_loss.item():.4f} | "
                    f"IG_adv: {ig_adv_loss.item():.4f} | "
                    f"IG_L1: {ig_l1_loss.item():.4f} | "
                    f"LV: {lv_loss.item():.4f} | "
                    f"LG: {lg_loss.item():.4f} | "
                    f"D: {d_loss.item():.4f}"
                )

        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        self.train_losses['ig'].append(avg_losses['ig_total'])
        self.train_losses['lg'].append(avg_losses['lg'])
        self.train_losses['d'].append(avg_losses['d'])

        return avg_losses

    @torch.no_grad()
    def validate(self) -> float:
        """Validate model."""
        if self.val_loader is None:
            return float('inf')

        self.model.eval()
        val_losses = []

        for batch in tqdm(self.val_loader, desc="Validation"):
            signal = batch['signal'].to(self.device)
            lead_i = signal[:, self.condition_leads, :]
            target = signal

            output = self.model(lead_i)
            pred = output['output']
            loss = self.criterion_l1(pred, target)
            val_losses.append(loss.item())

        avg_loss = np.mean(val_losses)
        self.val_losses.append(avg_loss)
        return avg_loss

    def train(self):
        """Full training loop."""
        self.logger.info("=" * 60)
        self.logger.info("Starting EKGAN training...")
        self.logger.info(f"Total epochs: {self.num_epochs}")
        self.logger.info(f"Batch size: {self.train_loader.batch_size}")
        self.logger.info(f"Lambda L1: {self.lambda_l1}")
        self.logger.info(f"Alpha LV: {self.alpha_lv}")
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

            train_losses = self.train_epoch()
            epoch_time = time.time() - epoch_start_time

            self.logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs} completed | "
                f"IG Loss: {train_losses['ig_total']:.4f} | "
                f"D Loss: {train_losses['d']:.4f} | "
                f"LG Loss: {train_losses['lg']:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

            val_loss = float('inf')
            if val_enabled and (epoch + 1) % val_every_n_epochs == 0:
                val_loss = self.validate()
                self.logger.info(f"Validation Loss: {val_loss:.6f}")

                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    if save_best:
                        self._save_checkpoint('best.pth', is_best=True)

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
            'best_val_loss': self.best_val_loss,
            'total_epochs': self.current_epoch + 1,
            'total_time': total_time,
        }
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        return history


def train_ekgan(config: dict, resume_from: Optional[str] = None):
    """EKGAN training entry point."""
    model = build_ekgan(config)
    train_loader = build_dataloader(config, split='train')
    val_loader = build_dataloader(config, split='val')

    trainer = EKGANTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        resume_from=resume_from,
    )

    history = trainer.train()
    return history

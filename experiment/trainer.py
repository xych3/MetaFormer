"""MetaFormer trainer with logging, checkpointing, and validation."""

import os
import json
import time
import logging
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from typing import Optional, Dict
from datetime import datetime


class Trainer:
    """Training loop for MetaFormer with AMP, gradient accumulation, and early stopping."""

    def __init__(self, model, train_loader, val_loader, config: dict,
                 output_dir: str, resume_from: Optional[str] = None):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        exp_config = config.get('experiment', {})
        train_config = config.get('training', {})

        self.device = torch.device(exp_config.get('device', 'cuda'))
        self.num_epochs = train_config.get('num_epochs', 100)
        self.grad_accum_steps = train_config.get('gradient_accumulation_steps', 1)
        self.max_grad_norm = train_config.get('max_grad_norm', 1.0)

        self.condition_leads = config.get('data', {}).get('condition_leads', [0])
        self.target_leads = [i for i in range(12) if i not in self.condition_leads]

        modality_dropout = train_config.get('modality_dropout', {})
        self.hw_drop_prob = modality_dropout.get('hw_drop_prob', 0.0)
        self.uncond_prob = modality_dropout.get('uncond_prob', 0.0)

        self.output_dir = output_dir
        self.checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        self.log_dir = os.path.join(output_dir, 'logs')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('MetaFormer.Trainer')

        self.model.to(self.device)
        self.optimizer = self._build_optimizer(train_config)
        self.scheduler = self._build_scheduler(train_config)

        amp_config = train_config.get('mixed_precision', {})
        self.use_amp = amp_config.get('enabled', False)
        self.scaler = GradScaler() if self.use_amp else None

        ckpt_config = train_config.get('checkpoint', {})
        self.save_every = ckpt_config.get('save_every_n_epochs', 9999)
        self.save_best = ckpt_config.get('save_best', True)
        self.save_latest = ckpt_config.get('save_latest', True)

        log_config = train_config.get('logging', {})
        self.log_every = log_config.get('log_every_n_steps', 100)

        val_config = train_config.get('validation', {})
        self.val_enabled = val_config.get('enabled', True)
        self.val_every = val_config.get('every_n_epochs', 1)

        es_config = train_config.get('early_stopping', {})
        self.early_stop_enabled = es_config.get('enabled', False)
        self.patience = es_config.get('patience', 20)
        self.min_delta = es_config.get('min_delta', 0.0001)

        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.current_epoch = 0
        self.global_step = 0
        self.history = {'train_loss': [], 'val_loss': [], 'lr': []}

        if resume_from:
            self._load_checkpoint(resume_from)

        self._save_config()

    def _build_optimizer(self, train_config: dict):
        opt_config = train_config.get('optimizer', {})
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=opt_config.get('lr', 1e-4),
            betas=tuple(opt_config.get('betas', [0.9, 0.999])),
            weight_decay=opt_config.get('weight_decay', 0.1),
            eps=opt_config.get('eps', 1e-8),
        )

    def _build_scheduler(self, train_config: dict):
        sch_config = train_config.get('scheduler', {})
        warmup_epochs = sch_config.get('warmup_epochs', 5)
        min_lr = sch_config.get('min_lr', 1e-6)
        base_lr = train_config.get('optimizer', {}).get('lr', 1e-4)

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                warmup_start = sch_config.get('warmup_start_lr', 1e-7)
                return warmup_start / base_lr + (1 - warmup_start / base_lr) * epoch / warmup_epochs
            progress = (epoch - warmup_epochs) / max(self.num_epochs - warmup_epochs, 1)
            import math
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return max(min_lr / base_lr, cosine)

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _save_config(self):
        config_path = os.path.join(self.output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)

    def _save_checkpoint(self, path: str, is_best: bool = False):
        state = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
        }
        if self.scaler is not None:
            state['scaler_state_dict'] = self.scaler.state_dict()
        torch.save(state, path)

    def _load_checkpoint(self, path: str):
        self.logger.info(f"Resuming from checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if self.scaler and 'scaler_state_dict' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        self.current_epoch = ckpt.get('epoch', 0) + 1
        self.global_step = ckpt.get('global_step', 0)
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        self.history = ckpt.get('history', self.history)

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs}")
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            signal = batch['signal'].to(self.device)
            metadata = {k: v.to(self.device) for k, v in batch['metadata'].items()}

            x_ref = signal[:, self.condition_leads, :]
            x_target = signal[:, self.target_leads, :]

            with autocast(enabled=self.use_amp):
                loss = self.model(
                    x_ref=x_ref,
                    x_target=x_target,
                    metadata=metadata,
                    hw_drop_prob=self.hw_drop_prob,
                    uncond_prob=self.uncond_prob,
                )
                loss = loss / self.grad_accum_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss.item() * self.grad_accum_steps
            num_batches += 1
            pbar.set_postfix({'loss': f'{total_loss / num_batches:.6f}',
                              'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'})

            if self.global_step % self.log_every == 0 and self.global_step > 0:
                self.logger.info(
                    f"Step {self.global_step} | Loss: {total_loss / num_batches:.6f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            signal = batch['signal'].to(self.device)
            metadata = {k: v.to(self.device) for k, v in batch['metadata'].items()}

            x_ref = signal[:, self.condition_leads, :]
            x_target = signal[:, self.target_leads, :]

            with autocast(enabled=self.use_amp):
                loss = self.model(
                    x_ref=x_ref,
                    x_target=x_target,
                    metadata=metadata,
                )

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def train(self):
        self.logger.info(f"Starting training for {self.num_epochs} epochs")
        self.logger.info(f"Output directory: {self.output_dir}")

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

            val_loss = None
            if self.val_enabled and self.val_loader and (epoch + 1) % self.val_every == 0:
                val_loss = self.validate()
                self.history['val_loss'].append(val_loss)
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.num_epochs} | "
                    f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

                if self.save_best and val_loss < self.best_val_loss - self.min_delta:
                    self.best_val_loss = val_loss
                    self.epochs_no_improve = 0
                    self._save_checkpoint(os.path.join(self.checkpoint_dir, 'best.pth'), is_best=True)
                    self.logger.info(f"Saved best model (val_loss={val_loss:.6f})")
                else:
                    self.epochs_no_improve += 1

                if self.early_stop_enabled and self.epochs_no_improve >= self.patience:
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.num_epochs} | Train Loss: {train_loss:.6f}")

            self.scheduler.step()

            if self.save_latest:
                self._save_checkpoint(os.path.join(self.checkpoint_dir, 'latest.pth'))
            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(
                    os.path.join(self.checkpoint_dir, f'epoch_{epoch + 1}.pth'))

        self._save_checkpoint(os.path.join(self.checkpoint_dir, 'final.pth'))
        history_path = os.path.join(self.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        self.logger.info("Training completed")
        return self.history


def build_trainer(model, train_loader, val_loader, config: dict,
                  output_dir: str, resume_from: Optional[str] = None) -> Trainer:
    """Factory function to create a Trainer instance."""
    return Trainer(model, train_loader, val_loader, config, output_dir, resume_from)

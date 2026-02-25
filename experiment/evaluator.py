"""MetaFormer evaluator for test-time metrics computation."""

import os
import json
import logging
import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
from typing import Optional, Dict, List
from datetime import datetime

from utils.metrics import (
    compute_all_metrics, compute_per_lead_metrics,
    metrics_to_dict, print_metrics
)


class Evaluator:
    """Evaluates a trained MetaFormer model on the test set."""

    def __init__(self, model, test_loader, config: dict, output_dir: str):
        self.config = config
        self.model = model
        self.test_loader = test_loader

        exp_config = config.get('experiment', {})
        eval_config = config.get('evaluation', {})

        self.device = torch.device(exp_config.get('device', 'cuda'))
        self.condition_leads = config.get('data', {}).get('condition_leads', [0])
        self.target_leads = [i for i in range(12) if i not in self.condition_leads]

        self.metrics_list = eval_config.get('metrics', [
            'mse', 'mae', 'rmse', 'prd', 'snr', 'psnr',
            'cosine_similarity', 'pearson', 'ssim'
        ])
        self.per_lead = eval_config.get('per_lead_metrics', True)
        self.num_samples = eval_config.get('num_samples', None)
        self.eval_batch_size = eval_config.get('batch_size', 512)

        self.lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                           'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.target_lead_names = [self.lead_names[i] for i in self.target_leads]

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.eval_dir = os.path.join(output_dir, 'evaluation', timestamp)
        os.makedirs(self.eval_dir, exist_ok=True)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('MetaFormer.Evaluator')

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def evaluate(self) -> Dict:
        """Run evaluation and compute metrics."""
        all_preds = []
        all_targets = []
        sample_count = 0

        for batch in tqdm(self.test_loader, desc="Evaluating"):
            signal = batch['signal'].to(self.device)
            metadata = {k: v.to(self.device) for k, v in batch['metadata'].items()}

            x_ref = signal[:, self.condition_leads, :]
            x_target = signal[:, self.target_leads, :]

            with autocast(enabled=False):
                pred = self.model.reconstruct(x_ref=x_ref, metadata=metadata)

            all_preds.append(pred.cpu().numpy())
            all_targets.append(x_target.cpu().numpy())
            sample_count += pred.shape[0]

            if self.num_samples and sample_count >= self.num_samples:
                break

        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        if self.num_samples:
            preds = preds[:self.num_samples]
            targets = targets[:self.num_samples]

        self.logger.info(f"Evaluated {preds.shape[0]} samples")

        results = self._compute_metrics(preds, targets)
        self._print_results(results)
        self._save_results(results)
        return results

    def _compute_metrics(self, preds: np.ndarray, targets: np.ndarray) -> Dict:
        overall = compute_all_metrics(preds, targets, self.metrics_list)
        results = {'overall': metrics_to_dict(overall)}

        if self.per_lead:
            per_lead = compute_per_lead_metrics(
                preds, targets, self.target_lead_names, self.metrics_list)
            results['per_lead'] = {
                lead: metrics_to_dict(m) for lead, m in per_lead.items()
            }

        results['num_samples'] = int(preds.shape[0])
        results['timestamp'] = datetime.now().isoformat()
        return results

    def _print_results(self, results: Dict):
        print_metrics(results['overall'], "Overall Metrics")
        if 'per_lead' in results:
            for lead, metrics in results['per_lead'].items():
                print(f"  {lead}: MSE={metrics.get('mse', 'N/A'):.6f}, "
                      f"Pearson={metrics.get('pearson', 'N/A'):.4f}")

    def _save_results(self, results: Dict):
        path = os.path.join(self.eval_dir, 'evaluation_results.json')
        with open(path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        self.logger.info(f"Results saved to {path}")

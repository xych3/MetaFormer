"""Evaluator class for ECGRecover baseline model."""

import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from baseline.ecgrecover import ECGRecover, build_ecgrecover
from data_preparation.ecg_dataset import build_dataloader, get_dataset_info
from utils.metrics import compute_all_metrics, compute_per_lead_metrics, MetricResult


class ECGRecoverEvaluator:
    """Evaluator for ECGRecover model."""

    def __init__(
        self,
        model: ECGRecover,
        config: dict,
        test_loader: DataLoader,
        output_dir: Optional[str] = None,
    ):
        self.model = model
        self.config = config
        self.test_loader = test_loader

        self.exp_config = config.get('experiment', {})
        self.eval_config = config.get('evaluation', {})
        self.data_config = config.get('data', {})

        self.device = torch.device(self.exp_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = self.model.to(self.device)
        self.model.eval()

        # Lead configuration
        self.condition_leads = self.data_config.get('condition_leads', [0])
        self.target_leads = [i for i in range(12) if i not in self.condition_leads]
        self.lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.target_lead_names = [self.lead_names[i] for i in self.target_leads]

        self.metrics_to_compute = self.eval_config.get('metrics', [
            'mse', 'mae', 'rmse', 'prd', 'cosine_similarity', 'snr', 'psnr'
        ])
        self.per_lead_metrics = self.eval_config.get('per_lead_metrics', True)

        if output_dir is None:
            output_dir = self.exp_config.get('output_dir', 'outputs_baseline/ecgrecover')
        self.output_dir = Path(output_dir) / 'evaluation'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()
        self.logger.info(f"ECGRecover Evaluator initialized on device: {self.device}")
        self.logger.info(f"Test samples: {len(test_loader.dataset)}")

    def _setup_logging(self):
        """Set up logging."""
        self.logger = logging.getLogger('ECGRecover.Evaluator')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.output_dir / f'eval_{timestamp}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    @torch.no_grad()
    def evaluate(
        self,
        num_samples: Optional[int] = None,
        return_samples: bool = False,
    ) -> Dict[str, Any]:
        """Run evaluation."""
        self.model.eval()
        self.logger.info("Starting evaluation...")

        all_predictions = []
        all_targets = []
        all_file_ids = []

        sample_count = 0
        max_samples = num_samples or len(self.test_loader.dataset)

        for batch in tqdm(self.test_loader, desc="Evaluating"):
            if sample_count >= max_samples:
                break

            signal = batch['signal'].to(self.device)
            file_ids = batch['file_id']
            batch_size = signal.shape[0]

            lead_i = signal[:, self.condition_leads, :]
            target = signal[:, self.target_leads, :]

            pred_full = self.model(lead_i)
            pred = pred_full[:, self.target_leads, :]

            all_predictions.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_file_ids.extend(file_ids)
            sample_count += batch_size

        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        self.logger.info(f"Evaluated {len(predictions)} samples")
        self.logger.info("Computing metrics...")

        metrics = compute_all_metrics(
            predictions, targets,
            metrics=self.metrics_to_compute,
            channel_wise=False,
        )

        if self.per_lead_metrics:
            per_lead = compute_per_lead_metrics(
                predictions, targets,
                lead_names=self.target_lead_names,
                metrics=self.metrics_to_compute
            )
            metrics['per_lead'] = per_lead

        results = {
            'config': {'num_samples': len(predictions)},
            'metrics': metrics,
        }

        self._print_results(metrics)
        self._save_results(results)

        if return_samples:
            results['predictions'] = predictions
            results['targets'] = targets
            results['file_ids'] = all_file_ids

        return results

    def _print_results(self, metrics: Dict[str, Any]):
        """Print evaluation results."""
        self.logger.info("=" * 60)
        self.logger.info("Evaluation Results:")
        self.logger.info("-" * 60)

        for key, value in metrics.items():
            if key != 'per_lead':
                if isinstance(value, MetricResult):
                    self.logger.info(f"  {key:<25}: {value.mean:.6f} +/- {value.std:.6f}")
                elif isinstance(value, float):
                    self.logger.info(f"  {key:<25}: {value:.6f}")

        if 'per_lead' in metrics:
            self.logger.info("-" * 60)
            self.logger.info("Per-Lead Metrics (MSE):")
            for lead_name, lead_metrics in metrics['per_lead'].items():
                mse = lead_metrics.get('mse', MetricResult(0, 0))
                if isinstance(mse, MetricResult):
                    self.logger.info(f"  {lead_name}: {mse.mean:.6f} +/- {mse.std:.6f}")

        self.logger.info("=" * 60)

    def _save_results(self, results: Dict):
        """Save evaluation results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'results_{timestamp}.json'

        def convert_to_serializable(obj):
            if isinstance(obj, MetricResult):
                return {'mean': float(obj.mean), 'std': float(obj.std)}
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            return obj

        results_clean = convert_to_serializable(results)
        results_clean.pop('predictions', None)
        results_clean.pop('targets', None)

        results_path = self.output_dir / filename
        with open(results_path, 'w') as f:
            json.dump(results_clean, f, indent=2)

        self.logger.info(f"Results saved to: {results_path}")


def evaluate_ecgrecover(config: dict, checkpoint_path: str):
    """ECGRecover evaluation entry point."""
    device = torch.device(config.get('experiment', {}).get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

    model = build_ecgrecover(config)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    eval_batch_size = config.get('evaluation', {}).get('batch_size', 256)
    test_loader = build_dataloader(config, split='test', shuffle=False, batch_size=eval_batch_size)

    evaluator = ECGRecoverEvaluator(model=model, config=config, test_loader=test_loader)

    results = evaluator.evaluate(
        num_samples=config.get('evaluation', {}).get('num_samples', None),
    )
    return results

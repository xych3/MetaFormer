"""Reconstruction methods for ECG lead synthesis."""

import yaml
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _load_yaml_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _prepare_metadata_batch(
    metadata_list: List[Dict],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Convert a list of metadata dictionaries to a batched tensor dictionary.

    Args:
        metadata_list: List of sample-level metadata dicts.
        device: Target device for tensors.

    Returns:
        Batched metadata dictionary with tensor values.
    """
    if not metadata_list:
        return {}

    keys = metadata_list[0].keys()
    batch = {}
    for key in keys:
        values = [m[key] for m in metadata_list]
        if isinstance(values[0], (int, float)):
            batch[key] = torch.tensor(values, device=device, dtype=torch.float32)
        elif isinstance(values[0], torch.Tensor):
            batch[key] = torch.stack(values).to(device)
        elif isinstance(values[0], np.ndarray):
            batch[key] = torch.from_numpy(np.stack(values)).to(device)
        else:
            batch[key] = torch.tensor(values, device=device)
    return batch


# ==================== Base class ====================

class ReconMethod(ABC):
    """Abstract base class for ECG reconstruction methods."""

    def __init__(self, config: dict, device: str = 'cuda'):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    @abstractmethod
    def load_model(self, checkpoint_path: str):
        """Load a pretrained model from checkpoint."""
        pass

    @abstractmethod
    def reconstruct(self, lead_i: torch.Tensor,
                    metadata: Optional[Dict] = None) -> torch.Tensor:
        """Reconstruct 12-lead ECG from Lead-I input.

        Args:
            lead_i: Lead-I signal, shape (B, 1, L).
            metadata: Optional patient metadata dictionary.

        Returns:
            Reconstructed 12-lead ECG, shape (B, 12, L).
        """
        pass


# ==================== MetaFormer ====================

class MetaFormerRecon(ReconMethod):
    """Reconstruction using MetaFormer."""

    def __init__(self, config: dict, device: str = 'cuda'):
        super().__init__(config, device)
        self.model = None

    def load_model(self, checkpoint_path: str):
        """Load MetaFormer from checkpoint."""
        from model.condition_unet_recon import build_model
        self.model = build_model(self.config)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        logger.info("MetaFormer model loaded from %s", checkpoint_path)

    @torch.no_grad()
    def reconstruct(self, lead_i: torch.Tensor,
                    metadata: Optional[Dict] = None) -> torch.Tensor:
        """Reconstruct 12-lead ECG using MetaFormer.

        Args:
            lead_i: Lead-I signal, shape (B, 1, L).
            metadata: Patient metadata dictionary.

        Returns:
            Reconstructed 12-lead ECG, shape (B, 12, L).
        """
        lead_i = lead_i.to(self.device)
        if metadata is not None:
            metadata = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in metadata.items()}
        return self.model.reconstruct_12lead(x_ref=lead_i, metadata=metadata)


# ==================== EKGAN ====================

class EKGANRecon(ReconMethod):
    """Reconstruction using EKGAN."""

    def __init__(self, config: dict, device: str = 'cuda'):
        super().__init__(config, device)
        self.model = None

    def load_model(self, checkpoint_path: str):
        """Load EKGAN from checkpoint."""
        from baseline.ekgan import build_ekgan
        self.model = build_ekgan(self.config)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        logger.info("EKGAN model loaded from %s", checkpoint_path)

    @torch.no_grad()
    def reconstruct(self, lead_i: torch.Tensor,
                    metadata: Optional[Dict] = None) -> torch.Tensor:
        """Reconstruct 12-lead ECG using EKGAN.

        Args:
            lead_i: Lead-I signal, shape (B, 1, L).
            metadata: Not used by EKGAN.

        Returns:
            Reconstructed 12-lead ECG, shape (B, 12, L).
        """
        lead_i = lead_i.to(self.device)
        result = self.model(lead_i)
        return result['output']


# ==================== mEcgNet ====================

class MEcgNetRecon(ReconMethod):
    """Reconstruction using mEcgNet."""

    def __init__(self, config: dict, device: str = 'cuda'):
        super().__init__(config, device)
        self.model = None

    def load_model(self, checkpoint_path: str):
        """Load mEcgNet from checkpoint."""
        from baseline.mecgnet import build_mecgnet
        self.model = build_mecgnet(self.config)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        logger.info("mEcgNet model loaded from %s", checkpoint_path)

    @torch.no_grad()
    def reconstruct(self, lead_i: torch.Tensor,
                    metadata: Optional[Dict] = None) -> torch.Tensor:
        """Reconstruct 12-lead ECG using mEcgNet.

        Args:
            lead_i: Lead-I signal, shape (B, 1, L).
            metadata: Not used by mEcgNet.

        Returns:
            Reconstructed 12-lead ECG, shape (B, 12, L).
        """
        lead_i = lead_i.to(self.device)
        return self.model(lead_i)


# ==================== ECGRecover ====================

class ECGrecoverRecon(ReconMethod):
    """Reconstruction using ECGRecover."""

    def __init__(self, config: dict, device: str = 'cuda'):
        super().__init__(config, device)
        self.model = None

    def load_model(self, checkpoint_path: str):
        """Load ECGRecover from checkpoint."""
        from baseline.ecgrecover import build_ecgrecover
        self.model = build_ecgrecover(self.config)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        logger.info("ECGRecover model loaded from %s", checkpoint_path)

    @torch.no_grad()
    def reconstruct(self, lead_i: torch.Tensor,
                    metadata: Optional[Dict] = None) -> torch.Tensor:
        """Reconstruct 12-lead ECG using ECGRecover.

        Args:
            lead_i: Lead-I signal, shape (B, 1, L).
            metadata: Not used by ECGRecover.

        Returns:
            Reconstructed 12-lead ECG, shape (B, 12, L).
        """
        lead_i = lead_i.to(self.device)
        return self.model(lead_i)


# ==================== Factory ====================

def build_recon_method(method_name: str, config: dict,
                       device: str = 'cuda') -> ReconMethod:
    """Build a reconstruction method by name.

    Args:
        method_name: One of 'metaformer', 'ekgan', 'mecgnet', 'ecgrecover'.
        config: Model configuration dictionary.
        device: Target device.

    Returns:
        Instantiated ReconMethod.
    """
    methods = {
        'metaformer': MetaFormerRecon,
        'ekgan': EKGANRecon,
        'mecgnet': MEcgNetRecon,
        'ecgrecover': ECGrecoverRecon,
    }
    if method_name not in methods:
        raise ValueError(f"Unknown method '{method_name}'. "
                         f"Supported: {list(methods.keys())}")
    return methods[method_name](config, device=device)

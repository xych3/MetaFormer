"""ECG Dataset classes for training and evaluation.

Supports dynamic loading of PTB-XL and CSN datasets.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import Optional, Dict, List, Tuple, Union
import random


class ECGDataset(Dataset):
    """Base ECG dataset class with dynamic signal and metadata loading."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        signal_length: int = 512,
        num_leads: int = 12,
        transform: Optional[callable] = None,
        return_metadata: bool = True,
        cache_signals: bool = False,
    ):
        """
        Args:
            data_dir: Preprocessed data directory.
            split: Dataset split ("train", "val", "test").
            signal_length: Signal length.
            num_leads: Number of leads.
            transform: Optional data transform.
            return_metadata: Whether to return metadata.
            cache_signals: Whether to cache signals in memory.
        """
        super().__init__()
        
        self.data_dir = data_dir
        self.split = split
        self.signal_length = signal_length
        self.num_leads = num_leads
        self.transform = transform
        self.return_metadata = return_metadata
        self.cache_signals = cache_signals
        
        # Load data split
        self.file_list = self._load_split()
        
        # Signal cache
        self.signal_cache = {}
        
        # Metadata cache
        self.metadata_cache = {}
    
    def _load_split(self) -> List[str]:
        """Load dataset split file."""
        split_path = os.path.join(self.data_dir, "splits.json")
        
        if os.path.exists(split_path):
            with open(split_path, 'r') as f:
                splits = json.load(f)
            return splits.get(self.split, [])
        else:
            # If no split file exists, scan signals directory
            signal_dir = os.path.join(self.data_dir, "signals")
            if os.path.exists(signal_dir):
                files = [f.replace('.npy', '') for f in os.listdir(signal_dir) if f.endswith('.npy')]
                return files
            return []
    
    def _load_signal(self, file_id: str) -> np.ndarray:
        """Load signal data."""
        if self.cache_signals and file_id in self.signal_cache:
            return self.signal_cache[file_id]
        
        signal_path = os.path.join(self.data_dir, "signals", f"{file_id}.npy")
        signal = np.load(signal_path)
        
        # Ensure correct shape
        if signal.ndim == 1:
            signal = signal.reshape(1, -1)
        
        # Resample if signal length does not match
        if signal.shape[1] != self.signal_length:
            from scipy import signal as scipy_signal
            signal = scipy_signal.resample(signal, self.signal_length, axis=1)
        
        if self.cache_signals:
            self.signal_cache[file_id] = signal
        
        return signal.astype(np.float32)
    
    def _load_metadata(self, file_id: str) -> Dict:
        """Load metadata."""
        if file_id in self.metadata_cache:
            return self.metadata_cache[file_id]
        
        meta_path = os.path.join(self.data_dir, "metadata", f"{file_id}_meta.json")
        
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        self.metadata_cache[file_id] = metadata
        return metadata
    
    def _process_metadata(self, metadata: Dict) -> Dict[str, torch.Tensor]:
        """Process metadata into model input format."""
        # Extract age
        age = metadata.get('age', None)
        if age is None or not isinstance(age, (int, float)) or np.isnan(age):
            age = float('nan')
        else:
            age = float(age)
        
        # Extract sex (0=Male, 1=Female)
        sex = metadata.get('sex', None)
        if sex is None or sex not in [0, 1]:
            sex = -1
        else:
            sex = int(sex)
        
        # Extract height
        height = metadata.get('height', None)
        if height is None or not isinstance(height, (int, float)) or np.isnan(height):
            height = float('nan')
        else:
            height = float(height)
        
        # Extract weight
        weight = metadata.get('weight', None)
        if weight is None or not isinstance(weight, (int, float)) or np.isnan(weight):
            weight = float('nan')
        else:
            weight = float(weight)
        
        return {
            'age': torch.tensor(age, dtype=torch.float32),
            'sex': torch.tensor(sex, dtype=torch.long),
            'height': torch.tensor(height, dtype=torch.float32),
            'weight': torch.tensor(weight, dtype=torch.float32),
        }
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a data sample.
        
        Returns:
            Dictionary with keys:
            - 'signal': shape (num_leads, signal_length)
            - 'metadata': metadata dict (if return_metadata=True)
            - 'file_id': file identifier
        """
        file_id = self.file_list[idx]
        
        # Load signal
        signal = self._load_signal(file_id)
        signal = torch.from_numpy(signal)
        
        # Apply transform
        if self.transform is not None:
            signal = self.transform(signal)
        
        result = {
            'signal': signal,
            'file_id': file_id,
        }
        
        # Load metadata
        if self.return_metadata:
            raw_metadata = self._load_metadata(file_id)
            processed_metadata = self._process_metadata(raw_metadata)
            result['metadata'] = processed_metadata
            result['raw_metadata'] = raw_metadata
        
        return result


class PTBXLDataset(ECGDataset):
    """PTB-XL ECG dataset."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        signal_length: int = 512,
        transform: Optional[callable] = None,
        return_metadata: bool = True,
        cache_signals: bool = False,
    ):
        super().__init__(
            data_dir=data_dir,
            split=split,
            signal_length=signal_length,
            num_leads=12,
            transform=transform,
            return_metadata=return_metadata,
            cache_signals=cache_signals,
        )
        
        self.dataset_name = "PTB-XL"
    
    def get_diagnostic_labels(self, file_id: str) -> Dict:
        """Get diagnostic labels for a record."""
        metadata = self._load_metadata(file_id)
        return {
            'scp_codes': metadata.get('scp_codes', {}),
            'scp_descriptions': metadata.get('scp_descriptions', {}),
        }


class CSNDataset(ECGDataset):
    """Chapman-Shaoxing-Ningbo (CSN) ECG dataset."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        signal_length: int = 512,
        transform: Optional[callable] = None,
        return_metadata: bool = True,
        cache_signals: bool = False,
    ):
        super().__init__(
            data_dir=data_dir,
            split=split,
            signal_length=signal_length,
            num_leads=12,
            transform=transform,
            return_metadata=return_metadata,
            cache_signals=cache_signals,
        )
        
        self.dataset_name = "CSN"
    
    def get_diagnostic_labels(self, file_id: str) -> Dict:
        """Get diagnostic labels for a record."""
        metadata = self._load_metadata(file_id)
        return {
            'dx_codes': metadata.get('dx_codes', []),
            'dx_descriptions': metadata.get('dx_descriptions', {}),
        }


class CombinedECGDataset(Dataset):
    """Combined dataset from multiple ECG data sources."""
    
    def __init__(
        self,
        datasets: List[ECGDataset],
        weights: Optional[List[float]] = None,
    ):
        """
        Args:
            datasets: List of datasets.
            weights: Sampling weights for each dataset.
        """
        super().__init__()
        
        self.datasets = datasets
        self.weights = weights
        
        # Compute cumulative sizes for each dataset
        self.cumulative_sizes = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cumulative_sizes.append(total)
    
    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Find the corresponding dataset and local index
        dataset_idx = 0
        for i, cum_size in enumerate(self.cumulative_sizes):
            if idx < cum_size:
                dataset_idx = i
                break
        
        if dataset_idx > 0:
            local_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        else:
            local_idx = idx
        
        sample = self.datasets[dataset_idx][local_idx]
        sample['dataset_name'] = self.datasets[dataset_idx].dataset_name
        
        return sample



class ECGTransform:
    """ECG data augmentation transform."""
    
    def __init__(
        self,
        noise_std: float = 0.0,
        scale_range: Tuple[float, float] = (1.0, 1.0),
        shift_range: Tuple[float, float] = (0.0, 0.0),
        flip_prob: float = 0.0,
    ):
        """
        Args:
            noise_std: Gaussian noise standard deviation.
            scale_range: Amplitude scaling range.
            shift_range: Baseline drift range.
            flip_prob: Flip probability.
        """
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.flip_prob = flip_prob
    
    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signal: shape (num_leads, signal_length)
        
        Returns:
            Transformed signal.
        """
        # Add noise
        if self.noise_std > 0:
            noise = torch.randn_like(signal) * self.noise_std
            signal = signal + noise
        
        # Amplitude scaling
        if self.scale_range != (1.0, 1.0):
            scale = random.uniform(*self.scale_range)
            signal = signal * scale
        
        # Baseline drift
        if self.shift_range != (0.0, 0.0):
            shift = random.uniform(*self.shift_range)
            signal = signal + shift
        
        # Flip
        if self.flip_prob > 0 and random.random() < self.flip_prob:
            signal = -signal
        
        return signal


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collation function for batching."""
    signals = torch.stack([item['signal'] for item in batch], dim=0)
    file_ids = [item['file_id'] for item in batch]
    
    result = {
        'signal': signals,
        'file_id': file_ids,
    }
    
    # Process metadata
    if 'metadata' in batch[0]:
        metadata = {
            'age': torch.stack([item['metadata']['age'] for item in batch]),
            'sex': torch.stack([item['metadata']['sex'] for item in batch]),
            'height': torch.stack([item['metadata']['height'] for item in batch]),
            'weight': torch.stack([item['metadata']['weight'] for item in batch]),
        }
        result['metadata'] = metadata
    
    # Dataset name
    if 'dataset_name' in batch[0]:
        result['dataset_name'] = [item['dataset_name'] for item in batch]
    
    return result


def build_dataloader(
    config: dict,
    split: str = "train",
    shuffle: Optional[bool] = None,
    batch_size: Optional[int] = None,
) -> DataLoader:
    """
    Build DataLoader from configuration.
    
    Args:
        config: Configuration dictionary.
        split: Data split.
        shuffle: Whether to shuffle data.
        batch_size: Override batch size.
    
    Returns:
        DataLoader instance.
    """
    data_config = config.get('data', {})
    exp_config = config.get('experiment', {})
    
    dataset_name = data_config.get('dataset', 'ptbxl')
    signal_length = data_config.get('signal_length', 512)
    batch_size = batch_size if batch_size is not None else data_config.get('batch_size', 32)
    num_workers = exp_config.get('num_workers', 4)
    pin_memory = data_config.get('pin_memory', True)
    prefetch_factor = data_config.get('prefetch_factor', 2)
    cache_signals = data_config.get('cache_signals', False)
    
    if shuffle is None:
        shuffle = data_config.get('shuffle', True) if split == "train" else False
    
    drop_last = data_config.get('drop_last', True) if split == "train" else False
    
    # Data augmentation
    transform = None
    aug_config = data_config.get('augmentation', {})
    if aug_config.get('enabled', False) and split == "train":
        transform = ECGTransform(
            noise_std=aug_config.get('noise_std', 0.01),
            scale_range=tuple(aug_config.get('scale_range', [0.9, 1.1])),
        )
    
    # Build dataset
    datasets = []
    
    if dataset_name in ['ptbxl', 'all']:
        ptbxl_path = data_config.get('ptbxl_path', 'data/ptbxl')
        if os.path.exists(ptbxl_path):
            ptbxl_dataset = PTBXLDataset(
                data_dir=ptbxl_path,
                split=split,
                signal_length=signal_length,
                transform=transform,
                return_metadata=True,
                cache_signals=cache_signals,
            )
            datasets.append(ptbxl_dataset)
    
    if dataset_name in ['csn', 'all']:
        csn_path = data_config.get('csn_path', 'data/csn')
        if os.path.exists(csn_path):
            csn_dataset = CSNDataset(
                data_dir=csn_path,
                split=split,
                signal_length=signal_length,
                transform=transform,
                return_metadata=True,
                cache_signals=cache_signals,
            )
            datasets.append(csn_dataset)
    
    if len(datasets) == 0:
        raise ValueError(f"No valid dataset found for {dataset_name}")
    elif len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = CombinedECGDataset(datasets)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )


def get_dataset_info(config: dict) -> Dict:
    """Get dataset information from configuration."""
    data_config = config.get('data', {})
    
    info = {
        'signal_length': data_config.get('signal_length', 512),
        'num_leads': data_config.get('num_leads', 12),
        'condition_leads': data_config.get('condition_leads', [0]),
        'lead_names': ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
    }
    
    target_leads = [i for i in range(12) if i not in info['condition_leads']]
    info['target_leads'] = target_leads
    info['target_lead_names'] = [info['lead_names'][i] for i in target_leads]
    info['condition_lead_names'] = [info['lead_names'][i] for i in info['condition_leads']]
    
    return info

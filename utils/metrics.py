"""Evaluation metrics for ECG signal reconstruction.

Includes MSE, MAE, RMSE, PRD, MMD, DTW, Cosine Similarity,
SSIM, SNR, PSNR, Pearson Correlation, and spectral metrics.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple, NamedTuple
from scipy import signal as scipy_signal
from scipy.spatial.distance import cdist
import torch
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Data Types
# ============================================================================

class MetricResult(NamedTuple):
    """Metric result containing mean and standard deviation."""
    mean: float
    std: float


# ============================================================================
# Utility Functions
# ============================================================================

def _ensure_3d(x: np.ndarray) -> np.ndarray:
    """Ensure input is a 3D array of shape (N, C, L)."""
    if x.ndim == 1:
        return x.reshape(1, 1, -1)
    elif x.ndim == 2:
        return x[:, np.newaxis, :]
    elif x.ndim == 3:
        return x
    else:
        raise ValueError(f"Unsupported input dimension: {x.ndim}")


def _aggregate_per_sample(values: np.ndarray) -> MetricResult:
    """Aggregate per-sample metric values, filtering invalid entries."""
    valid_mask = np.isfinite(values)
    valid_values = values[valid_mask]

    if len(valid_values) == 0:
        return MetricResult(mean=float('nan'), std=float('nan'))

    return MetricResult(
        mean=float(np.mean(valid_values)),
        std=float(np.std(valid_values))
    )


# ============================================================================
# Basic Metrics
# ============================================================================

def compute_mse(pred: np.ndarray, target: np.ndarray,
                channel_wise: bool = False) -> MetricResult:
    """Compute Mean Squared Error per sample."""
    pred = _ensure_3d(pred)
    target = _ensure_3d(target)

    if channel_wise:
        per_sample = np.mean((pred - target) ** 2, axis=2).flatten()
    else:
        per_sample = np.mean((pred - target) ** 2, axis=(1, 2))

    return _aggregate_per_sample(per_sample)


def compute_mae(pred: np.ndarray, target: np.ndarray,
                channel_wise: bool = False) -> MetricResult:
    """Compute Mean Absolute Error per sample."""
    pred = _ensure_3d(pred)
    target = _ensure_3d(target)

    if channel_wise:
        per_sample = np.mean(np.abs(pred - target), axis=2).flatten()
    else:
        per_sample = np.mean(np.abs(pred - target), axis=(1, 2))

    return _aggregate_per_sample(per_sample)


def compute_rmse(pred: np.ndarray, target: np.ndarray,
                 channel_wise: bool = False) -> MetricResult:
    """Compute Root Mean Squared Error per sample."""
    pred = _ensure_3d(pred)
    target = _ensure_3d(target)

    if channel_wise:
        per_sample = np.sqrt(np.mean((pred - target) ** 2, axis=2)).flatten()
    else:
        per_sample = np.sqrt(np.mean((pred - target) ** 2, axis=(1, 2)))

    return _aggregate_per_sample(per_sample)


# ============================================================================
# Signal Quality Metrics
# ============================================================================

def compute_prd(pred: np.ndarray, target: np.ndarray,
                channel_wise: bool = False) -> MetricResult:
    """Compute Percentage Root-mean-square Difference (PRD).

    PRD = sqrt(sum((pred - target)^2) / sum(target^2)) * 100
    """
    pred = _ensure_3d(pred)
    target = _ensure_3d(target)

    if channel_wise:
        numerator = np.sum((pred - target) ** 2, axis=2)
        denominator = np.maximum(np.sum(target ** 2, axis=2), 1e-10)
        per_sample = np.sqrt(numerator / denominator) * 100
        per_sample = per_sample.flatten()
    else:
        numerator = np.sum((pred - target) ** 2, axis=(1, 2))
        denominator = np.maximum(np.sum(target ** 2, axis=(1, 2)), 1e-10)
        per_sample = np.sqrt(numerator / denominator) * 100

    return _aggregate_per_sample(per_sample)


def compute_snr(pred: np.ndarray, target: np.ndarray,
                channel_wise: bool = False) -> MetricResult:
    """Compute Signal-to-Noise Ratio (dB).

    SNR = 10 * log10(sum(target^2) / sum((pred - target)^2))
    """
    pred = _ensure_3d(pred)
    target = _ensure_3d(target)

    if channel_wise:
        signal_power = np.sum(target ** 2, axis=2)
        noise_power = np.maximum(np.sum((pred - target) ** 2, axis=2), 1e-10)
        per_sample = 10 * np.log10(signal_power / noise_power)
        per_sample = per_sample.flatten()
    else:
        signal_power = np.sum(target ** 2, axis=(1, 2))
        noise_power = np.maximum(np.sum((pred - target) ** 2, axis=(1, 2)), 1e-10)
        per_sample = 10 * np.log10(signal_power / noise_power)

    return _aggregate_per_sample(per_sample)


def compute_psnr(pred: np.ndarray, target: np.ndarray,
                 max_val: Optional[float] = None,
                 channel_wise: bool = False) -> MetricResult:
    """Compute Peak Signal-to-Noise Ratio (dB).

    PSNR = 10 * log10(max_val^2 / MSE)
    """
    pred = _ensure_3d(pred)
    target = _ensure_3d(target)

    if max_val is None:
        max_val = np.max(np.abs(target)) - np.min(target)
        max_val = max(max_val, 1.0)

    if channel_wise:
        mse = np.maximum(np.mean((pred - target) ** 2, axis=2), 1e-10)
        per_sample = 10 * np.log10(max_val ** 2 / mse)
        per_sample = per_sample.flatten()
    else:
        mse = np.maximum(np.mean((pred - target) ** 2, axis=(1, 2)), 1e-10)
        per_sample = 10 * np.log10(max_val ** 2 / mse)

    return _aggregate_per_sample(per_sample)


# ============================================================================
# Similarity Metrics
# ============================================================================

def compute_cosine_similarity(pred: np.ndarray, target: np.ndarray,
                              channel_wise: bool = False) -> MetricResult:
    """Compute cosine similarity between prediction and target."""
    pred = _ensure_3d(pred)
    target = _ensure_3d(target)

    if channel_wise:
        pred_norm = np.maximum(np.linalg.norm(pred, axis=2, keepdims=True), 1e-10)
        target_norm = np.maximum(np.linalg.norm(target, axis=2, keepdims=True), 1e-10)
        cos_sim = np.sum(pred * target, axis=2) / (pred_norm.squeeze(-1) * target_norm.squeeze(-1))
        per_sample = cos_sim.flatten()
    else:
        pred_flat = pred.reshape(pred.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)

        pred_norm = np.maximum(np.linalg.norm(pred_flat, axis=1), 1e-10)
        target_norm = np.maximum(np.linalg.norm(target_flat, axis=1), 1e-10)

        per_sample = np.sum(pred_flat * target_flat, axis=1) / (pred_norm * target_norm)

    return _aggregate_per_sample(per_sample)


def compute_pearson_correlation(pred: np.ndarray, target: np.ndarray,
                                channel_wise: bool = False) -> MetricResult:
    """Compute Pearson correlation coefficient."""
    pred = _ensure_3d(pred)
    target = _ensure_3d(target)

    correlations = []

    if channel_wise:
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                corr = np.corrcoef(pred[i, j], target[i, j])[0, 1]
                if np.isfinite(corr):
                    correlations.append(corr)
    else:
        for i in range(pred.shape[0]):
            pred_flat = pred[i].flatten()
            target_flat = target[i].flatten()
            corr = np.corrcoef(pred_flat, target_flat)[0, 1]
            if np.isfinite(corr):
                correlations.append(corr)

    if len(correlations) == 0:
        return MetricResult(mean=float('nan'), std=float('nan'))

    correlations = np.array(correlations)
    return _aggregate_per_sample(correlations)


# ============================================================================
# Structural Similarity (SSIM)
# ============================================================================

def compute_ssim_1d(pred: np.ndarray, target: np.ndarray,
                    window_size: int = 11,
                    channel_wise: bool = False) -> MetricResult:
    """Compute 1D Structural Similarity Index (SSIM).

    Uses Gaussian sliding window with mode='same' to preserve length.
    """
    pred = _ensure_3d(pred)
    target = _ensure_3d(target)

    C1 = (0.01 * 2) ** 2  # dynamic range [-1, 1]
    C2 = (0.03 * 2) ** 2

    window = scipy_signal.windows.gaussian(window_size, std=1.5)
    window = window / window.sum()

    ssim_values = []

    for i in range(pred.shape[0]):
        sample_ssim = []
        for j in range(pred.shape[1]):
            x = pred[i, j]
            y = target[i, j]

            mu_x = np.convolve(x, window, mode='same')
            mu_y = np.convolve(y, window, mode='same')

            sigma_x_sq = np.maximum(np.convolve(x ** 2, window, mode='same') - mu_x ** 2, 0)
            sigma_y_sq = np.maximum(np.convolve(y ** 2, window, mode='same') - mu_y ** 2, 0)
            sigma_xy = np.convolve(x * y, window, mode='same') - mu_x * mu_y

            numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
            denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x_sq + sigma_y_sq + C2)

            ssim = numerator / (denominator + 1e-10)
            sample_ssim.append(np.mean(ssim))

        if channel_wise:
            ssim_values.extend(sample_ssim)
        else:
            ssim_values.append(np.mean(sample_ssim))

    return _aggregate_per_sample(np.array(ssim_values))


# ============================================================================
# Distribution Metric (MMD)
# ============================================================================

def compute_mmd(pred: np.ndarray, target: np.ndarray,
                kernel: str = 'rbf',
                gamma: Optional[float] = None,
                max_samples: int = 500) -> MetricResult:
    """Compute Maximum Mean Discrepancy (MMD).

    MMD is a distribution-level metric. The reported std is from
    bootstrap estimation.
    """
    pred = _ensure_3d(pred)
    target = _ensure_3d(target)

    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)

    if pred_flat.shape[0] > max_samples:
        indices = np.random.choice(pred_flat.shape[0], max_samples, replace=False)
        pred_flat = pred_flat[indices]
        target_flat = target_flat[indices]

    n = pred_flat.shape[0]
    m = target_flat.shape[0]

    if n < 2 or m < 2:
        return MetricResult(mean=float('nan'), std=float('nan'))

    if gamma is None:
        combined = np.vstack([pred_flat, target_flat])
        pairwise_dists = cdist(combined, combined, 'sqeuclidean')
        gamma = 1.0 / np.median(pairwise_dists[pairwise_dists > 0])

    def rbf_kernel(X, Y, g):
        dist = cdist(X, Y, 'sqeuclidean')
        return np.exp(-g * dist)

    def compute_mmd_value(pred_sample, target_sample):
        n_s = pred_sample.shape[0]
        m_s = target_sample.shape[0]

        if kernel == 'rbf':
            K_xx = rbf_kernel(pred_sample, pred_sample, gamma)
            K_yy = rbf_kernel(target_sample, target_sample, gamma)
            K_xy = rbf_kernel(pred_sample, target_sample, gamma)
        else:
            K_xx = np.dot(pred_sample, pred_sample.T)
            K_yy = np.dot(target_sample, target_sample.T)
            K_xy = np.dot(pred_sample, target_sample.T)

        mmd = (np.sum(K_xx) - np.trace(K_xx)) / (n_s * (n_s - 1) + 1e-10) + \
              (np.sum(K_yy) - np.trace(K_yy)) / (m_s * (m_s - 1) + 1e-10) - \
              2 * np.mean(K_xy)

        return max(mmd, 0)

    # Bootstrap for std estimation
    mmd_values = []
    n_bootstrap = 10

    for _ in range(n_bootstrap):
        idx_pred = np.random.choice(n, n, replace=True)
        idx_target = np.random.choice(m, m, replace=True)
        mmd_val = compute_mmd_value(pred_flat[idx_pred], target_flat[idx_target])
        mmd_values.append(mmd_val)

    return MetricResult(
        mean=float(np.mean(mmd_values)),
        std=float(np.std(mmd_values))
    )


# ============================================================================
# Dynamic Time Warping (DTW)
# ============================================================================

def compute_dtw_distance(pred: np.ndarray, target: np.ndarray,
                         max_samples: int = 100,
                         channel_wise: bool = False,
                         use_softdtw: bool = True,
                         gamma: float = 0.1) -> MetricResult:
    """Compute Dynamic Time Warping distance.

    Uses SoftDTW by default for consistency with ECGRecover reference.
    """
    pred = _ensure_3d(pred)
    target = _ensure_3d(target)

    n_samples = min(pred.shape[0], max_samples)

    if use_softdtw:
        try:
            from utils.soft_dtw import SoftDTW
            sdtw = SoftDTW(use_cuda=False, gamma=gamma)
            dtw_func = lambda x, y: _compute_softdtw_distance(x, y, sdtw)
        except ImportError:
            print("Warning: Failed to import SoftDTW, falling back to alternative methods")
            use_softdtw = False

    if not use_softdtw:
        try:
            from fastdtw import fastdtw
            from scipy.spatial.distance import euclidean
            dtw_func = lambda x, y: fastdtw(x, y, dist=euclidean)[0]
        except ImportError:
            try:
                from tslearn.metrics import dtw as tslearn_dtw
                dtw_func = tslearn_dtw
            except ImportError:
                dtw_func = _optimized_dtw

    dtw_distances = []
    indices = np.random.choice(pred.shape[0], n_samples, replace=False)

    for i in indices:
        sample_distances = []
        for j in range(pred.shape[1]):
            try:
                dist = dtw_func(pred[i, j], target[i, j])
                # Normalize by sequence length
                length = len(pred[i, j])
                normalized_dist = dist / length
                sample_distances.append(normalized_dist)
            except Exception:
                continue

        if len(sample_distances) > 0:
            if channel_wise:
                dtw_distances.extend(sample_distances)
            else:
                dtw_distances.append(np.mean(sample_distances))

    if len(dtw_distances) == 0:
        return MetricResult(mean=float('nan'), std=float('nan'))

    return _aggregate_per_sample(np.array(dtw_distances))


def _compute_softdtw_distance(x: np.ndarray, y: np.ndarray, sdtw) -> float:
    """Compute SoftDTW distance between two 1D signals.

    Input format follows ECGRecover reference: (batch, seq_len, dims).
    """
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

    dist = sdtw(x_tensor, y_tensor)
    return dist[0].numpy()


def _optimized_dtw(x: np.ndarray, y: np.ndarray) -> float:
    """Optimized DTW with Sakoe-Chiba band constraint."""
    n, m = len(x), len(y)
    window = max(int(min(n, m) * 0.1), 10)

    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0

    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m, i + window) + 1

        for j in range(j_start, j_end):
            cost = abs(x[i - 1] - y[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    return dtw[n, m]


# ============================================================================
# Spectral Metrics
# ============================================================================

def compute_spectral_correlation(pred: np.ndarray, target: np.ndarray,
                                 fs: float = 500.0,
                                 channel_wise: bool = False) -> MetricResult:
    """Compute spectral correlation via Welch PSD estimation."""
    pred = _ensure_3d(pred)
    target = _ensure_3d(target)

    correlations = []

    for i in range(pred.shape[0]):
        sample_corrs = []
        for j in range(pred.shape[1]):
            nperseg = min(256, len(pred[i, j]))
            _, psd_pred = scipy_signal.welch(pred[i, j], fs=fs, nperseg=nperseg)
            _, psd_target = scipy_signal.welch(target[i, j], fs=fs, nperseg=nperseg)

            if len(psd_pred) == len(psd_target):
                corr = np.corrcoef(psd_pred, psd_target)[0, 1]
                if np.isfinite(corr):
                    sample_corrs.append(corr)

        if channel_wise:
            correlations.extend(sample_corrs)
        elif len(sample_corrs) > 0:
            correlations.append(np.mean(sample_corrs))

    if len(correlations) == 0:
        return MetricResult(mean=float('nan'), std=float('nan'))

    return _aggregate_per_sample(np.array(correlations))


def compute_spectral_mse(pred: np.ndarray, target: np.ndarray,
                         fs: float = 500.0,
                         channel_wise: bool = False) -> MetricResult:
    """Compute MSE in the frequency domain (normalized FFT magnitudes)."""
    pred = _ensure_3d(pred)
    target = _ensure_3d(target)

    mse_values = []

    for i in range(pred.shape[0]):
        sample_mse = []
        for j in range(pred.shape[1]):
            fft_pred = np.abs(np.fft.rfft(pred[i, j]))
            fft_target = np.abs(np.fft.rfft(target[i, j]))

            fft_pred = fft_pred / (np.max(fft_pred) + 1e-10)
            fft_target = fft_target / (np.max(fft_target) + 1e-10)

            mse = np.mean((fft_pred - fft_target) ** 2)
            sample_mse.append(mse)

        if channel_wise:
            mse_values.extend(sample_mse)
        else:
            mse_values.append(np.mean(sample_mse))

    return _aggregate_per_sample(np.array(mse_values))


# ============================================================================
# Aggregation Functions
# ============================================================================

def compute_all_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    metrics: Optional[List[str]] = None,
    channel_wise: bool = False,
) -> Dict[str, MetricResult]:
    """Compute all specified metrics.

    Args:
        pred: Predictions, shape (N, C, L) or (N, L).
        target: Targets, same shape as pred.
        metrics: List of metric names to compute.
        channel_wise: Whether to compute per-channel metrics.

    Returns:
        Dictionary mapping metric names to MetricResult.
    """
    if metrics is None:
        metrics = ['mse', 'mae', 'rmse', 'prd', 'snr', 'psnr',
                   'cosine_similarity', 'ssim', 'pearson']

    results = {}

    metric_functions = {
        'mse': lambda p, t: compute_mse(p, t, channel_wise),
        'mae': lambda p, t: compute_mae(p, t, channel_wise),
        'rmse': lambda p, t: compute_rmse(p, t, channel_wise),
        'prd': lambda p, t: compute_prd(p, t, channel_wise),
        'snr': lambda p, t: compute_snr(p, t, channel_wise),
        'psnr': lambda p, t: compute_psnr(p, t, channel_wise=channel_wise),
        'cosine_similarity': lambda p, t: compute_cosine_similarity(p, t, channel_wise),
        'pearson': lambda p, t: compute_pearson_correlation(p, t, channel_wise),
        'ssim': lambda p, t: compute_ssim_1d(p, t, channel_wise=channel_wise),
        'mmd': compute_mmd,
        'dtw': lambda p, t: compute_dtw_distance(p, t, channel_wise=channel_wise),
        'spectral_correlation': lambda p, t: compute_spectral_correlation(p, t, channel_wise=channel_wise),
        'spectral_mse': lambda p, t: compute_spectral_mse(p, t, channel_wise=channel_wise),
    }

    for metric in metrics:
        if metric in metric_functions:
            try:
                results[metric] = metric_functions[metric](pred, target)
            except Exception as e:
                print(f"Warning: Failed to compute {metric}: {e}")
                results[metric] = MetricResult(mean=float('nan'), std=float('nan'))

    return results


def metrics_to_dict(metrics: Dict[str, MetricResult],
                    include_std: bool = True) -> Dict[str, float]:
    """Convert MetricResult dictionary to plain dictionary."""
    result = {}
    for key, value in metrics.items():
        if isinstance(value, MetricResult):
            result[key] = value.mean
            if include_std:
                result[f'{key}_std'] = value.std
        elif isinstance(value, dict):
            result[key] = metrics_to_dict(value, include_std)
        else:
            result[key] = value
    return result


def aggregate_metrics(
    metrics_list: List[Dict[str, MetricResult]],
) -> Dict[str, MetricResult]:
    """Aggregate multiple metric result dictionaries.

    Uses combined variance: Var = Var(means) + Mean(variances).
    """
    if not metrics_list:
        return {}

    all_keys = set()
    for m in metrics_list:
        all_keys.update(m.keys())

    aggregated = {}

    for key in all_keys:
        means = []
        stds = []

        for m in metrics_list:
            if key in m:
                val = m[key]
                if isinstance(val, MetricResult):
                    if np.isfinite(val.mean):
                        means.append(val.mean)
                    if np.isfinite(val.std):
                        stds.append(val.std)

        if means:
            overall_mean = np.mean(means)
            variance_of_means = np.var(means)
            mean_of_variances = np.mean([s**2 for s in stds]) if stds else 0
            overall_std = np.sqrt(variance_of_means + mean_of_variances)

            aggregated[key] = MetricResult(
                mean=float(overall_mean),
                std=float(overall_std)
            )
        else:
            aggregated[key] = MetricResult(mean=float('nan'), std=float('nan'))

    return aggregated


def print_metrics(metrics: Dict[str, Union[MetricResult, float]], title: str = "Metrics"):
    """Pretty-print metrics to console."""
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")

    for key, value in sorted(metrics.items()):
        if isinstance(value, MetricResult):
            print(f"  {key:<25}: {value.mean:.6f} +/- {value.std:.6f}")
        elif isinstance(value, float):
            print(f"  {key:<25}: {value:.6f}")
        else:
            print(f"  {key:<25}: {value}")

    print(f"{'=' * 60}\n")


def compute_per_lead_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    lead_names: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
) -> Dict[str, Dict[str, MetricResult]]:
    """Compute metrics for each lead independently.

    Args:
        pred: Predictions, shape (N, num_leads, L).
        target: Targets, same shape.
        lead_names: Optional list of lead name strings.
        metrics: Metric names to compute.

    Returns:
        Nested dictionary: {lead_name: {metric: MetricResult}}.
    """
    pred = _ensure_3d(pred)
    target = _ensure_3d(target)

    num_leads = pred.shape[1]

    if lead_names is None:
        lead_names = [f"lead_{i}" for i in range(num_leads)]

    per_lead_results = {}

    for i in range(num_leads):
        lead_pred = pred[:, i:i+1, :]
        lead_target = target[:, i:i+1, :]

        lead_metrics = compute_all_metrics(lead_pred, lead_target, metrics)
        per_lead_results[lead_names[i]] = lead_metrics

    return per_lead_results

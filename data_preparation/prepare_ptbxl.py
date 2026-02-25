"""Data preprocessing pipeline for PTB-XL Database.

Extracts ECG signals and metadata for model training.
"""

import argparse
import os
import ast
import numpy as np
import pandas as pd
import wfdb
import sys
import json
from tqdm import tqdm
from scipy import signal as scipy_signal
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

home_dir = os.path.expanduser("~") + '/'
file_dir = os.path.dirname(os.path.realpath(__file__)) + '/'

MAX_PROCESS = 16
SCP_DICT = None


def init_worker(scp_dict):
    """Initialize worker process with shared SCP dictionary."""
    import warnings
    warnings.filterwarnings('ignore')
    global SCP_DICT
    SCP_DICT = scp_dict


def get_parser():
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(description="Preprocess PTB-XL dataset for ECG reconstruction")
    parser.add_argument(
        "--root", type=str, metavar="DIR",
        default=os.path.join(home_dir, "ECG_Dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"),
        help="Root directory of PTB-XL dataset"
    )
    parser.add_argument(
        "--output", type=str, metavar="DIR",
        default=os.path.join(file_dir, "../data/ptbxl"),
        help="Output directory for preprocessed data"
    )
    parser.add_argument(
        "--target-length", type=int, default=512,
        help="Target signal length after downsampling"
    )
    parser.add_argument(
        "--lowcut", type=float, default=0.05,
        help="Low cutoff frequency for bandpass filter (Hz)"
    )
    parser.add_argument(
        "--highcut", type=float, default=150.0,
        help="High cutoff frequency for bandpass filter (Hz)"
    )
    parser.add_argument(
        "--n-processes", type=int, default=None,
        help="Number of processes for parallel processing"
    )
    return parser


def minmax_normalize(signal, min_val=-1, max_val=1):
    """Apply global MinMax normalization across all 12 leads.

    Args:
        signal: Array of shape (n_leads, n_samples) or (n_samples, n_leads).
        min_val: Target minimum value.
        max_val: Target maximum value.

    Returns:
        normalized_signal: Normalized signal array.
        norm_params: Dictionary of normalization parameters.
    """
    global_min = np.min(signal)
    global_max = np.max(signal)

    if global_max - global_min < 1e-8:
        normalized = np.zeros_like(signal)
    else:
        normalized = (signal - global_min) / (global_max - global_min)
        normalized = normalized * (max_val - min_val) + min_val

    norm_params = {
        'global_min': float(global_min),
        'global_max': float(global_max),
        'target_min': min_val,
        'target_max': max_val
    }

    return normalized, norm_params


def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """Apply Butterworth bandpass filter.

    Args:
        signal: Array of shape (n_leads, n_samples).
        lowcut: Low cutoff frequency (Hz).
        highcut: High cutoff frequency (Hz).
        fs: Sampling rate (Hz).
        order: Filter order.

    Returns:
        Filtered signal array.
    """
    nyquist = fs / 2.0
    low = max(lowcut / nyquist, 0.001)
    high = min(highcut / nyquist, 0.999)

    if low >= high:
        return signal

    try:
        b, a = scipy_signal.butter(order, [low, high], btype='band')

        if signal.ndim == 1:
            filtered = scipy_signal.filtfilt(b, a, signal)
        else:
            filtered = np.zeros_like(signal)
            for i in range(signal.shape[0]):
                filtered[i] = scipy_signal.filtfilt(b, a, signal[i])

        return filtered
    except Exception as e:
        print(f"Warning: Bandpass filter failed: {e}")
        return signal


def resample_signal(signal, target_length):
    """Resample signal to the target length.

    Args:
        signal: Array of shape (n_leads, n_samples).
        target_length: Desired output length.

    Returns:
        Resampled signal array.
    """
    if signal.ndim == 1:
        return scipy_signal.resample(signal, target_length)
    else:
        resampled = np.zeros((signal.shape[0], target_length), dtype=signal.dtype)
        for i in range(signal.shape[0]):
            resampled[i] = scipy_signal.resample(signal[i], target_length)
        return resampled


def clean_nan(obj):
    """Recursively replace NaN/Inf floats with None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan(v) for v in obj]
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def prepare_scp_dict(data_dir):
    """Load the SCP statements dictionary from CSV."""
    scp_statements_path = os.path.join(data_dir, "scp_statements.csv")
    if not os.path.exists(scp_statements_path):
        print(f"Warning: scp_statements.csv not found at {scp_statements_path}")
        return {}

    df = pd.read_csv(scp_statements_path)
    scp_dict = {}
    for i in range(len(df)):
        scp_dict[str(df.iloc[i, 0])] = df.iloc[i, 1:].to_dict()

    return scp_dict


def extract_metadata(row, scp_dict):
    """Extract metadata from a PTB-XL database row.

    Args:
        row: pandas Series representing one record.
        scp_dict: SCP code dictionary.

    Returns:
        Metadata dictionary.
    """
    metadata = {
        'ecg_id': int(float(row['ecg_id'])) if pd.notna(row.get('ecg_id')) and row.get('ecg_id') != '' else None,
        'patient_id': int(float(row['patient_id'])) if pd.notna(row.get('patient_id')) and row.get('patient_id') != '' else None,
        'age': float(row['age']) if pd.notna(row.get('age')) and row.get('age') != '' else None,
        'sex': int(float(row['sex'])) if pd.notna(row.get('sex')) and row.get('sex') != '' else None,  # 0: Male, 1: Female
        'height': float(row['height']) if pd.notna(row.get('height')) and row.get('height') != '' else None,
        'weight': float(row['weight']) if pd.notna(row.get('weight')) and row.get('weight') != '' else None,
        'recording_date': str(row['recording_date']) if pd.notna(row.get('recording_date')) else None,
        'report': str(row['report']) if pd.notna(row.get('report')) else None,
        'strat_fold': int(float(row['strat_fold'])) if pd.notna(row.get('strat_fold')) and row.get('strat_fold') != '' else None,
    }

    # Parse SCP codes
    scp_codes = row.get('scp_codes', {})
    if isinstance(scp_codes, str):
        try:
            scp_codes = ast.literal_eval(scp_codes)
        except:
            scp_codes = {}

    metadata['scp_codes'] = scp_codes

    # Add SCP descriptions
    scp_descriptions = {}
    for code in scp_codes.keys():
        if code in scp_dict:
            scp_descriptions[code] = scp_dict[code]
    metadata['scp_descriptions'] = scp_descriptions

    # Diagnostic fields
    metadata['heart_axis'] = str(row['heart_axis']) if pd.notna(row.get('heart_axis')) else None
    metadata['infarction_stadium1'] = str(row['infarction_stadium1']) if pd.notna(row.get('infarction_stadium1')) else None
    metadata['infarction_stadium2'] = str(row['infarction_stadium2']) if pd.notna(row.get('infarction_stadium2')) else None

    # Signal quality fields
    metadata['baseline_drift'] = str(row['baseline_drift']) if pd.notna(row.get('baseline_drift')) else None
    metadata['static_noise'] = str(row['static_noise']) if pd.notna(row.get('static_noise')) else None
    metadata['burst_noise'] = str(row['burst_noise']) if pd.notna(row.get('burst_noise')) else None
    metadata['electrodes_problems'] = str(row['electrodes_problems']) if pd.notna(row.get('electrodes_problems')) else None

    return metadata


def process_single_record(args):
    """Process a single ECG record: normalize, filter, resample, and save.

    Args:
        args: Tuple of (filename, data_path, output_dir, row_dict,
              target_length, lowcut, highcut).

    Returns:
        Tuple of (filename, success, error_message).
    """
    filename, data_path, output_dir, row_dict, target_length, lowcut, highcut = args

    global SCP_DICT

    try:
        record_path = os.path.join(data_path, filename)
        record = wfdb.rdsamp(record_path)
        signal = record[0]  # (n_samples, n_leads)
        header = record[1]

        original_fs = header['fs']

        # Fill NaN values with zero
        nan_count = np.isnan(signal).sum()
        if nan_count > 0:
            signal = np.nan_to_num(signal, nan=0.0)

        # Transpose to (n_leads, n_samples)
        signal = signal.T

        # Step 1: MinMax normalization to [-1, 1]
        signal_normalized, norm_params = minmax_normalize(signal, min_val=-1, max_val=1)

        # Step 2: Bandpass filter
        signal_filtered = bandpass_filter(signal_normalized, lowcut, highcut, original_fs)

        # Step 3: Downsample to target length
        signal_resampled = resample_signal(signal_filtered, target_length)

        signal_final = signal_resampled.astype(np.float32)

        row_series = pd.Series(row_dict)
        metadata = extract_metadata(row_series, SCP_DICT)
        metadata['original_fs'] = original_fs
        metadata['original_length'] = header['sig_len']
        metadata['target_length'] = target_length
        metadata['lead_names'] = header['sig_name']
        metadata['norm_params'] = norm_params
        metadata['filter_params'] = {'lowcut': lowcut, 'highcut': highcut}
        metadata = clean_nan(metadata)

        save_name = filename

        # Save signal
        signal_path = os.path.join(output_dir, 'signals', f'{save_name}.npy')
        os.makedirs(os.path.dirname(signal_path), exist_ok=True)
        np.save(signal_path, signal_final)

        # Save metadata
        meta_path = os.path.join(output_dir, 'metadata', f'{save_name}_meta.json')
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

        return filename, True, None

    except Exception as e:
        return filename, False, str(e)


def create_split_files(Y, output_dir):
    """Create train/val/test splits based on PTB-XL stratified folds.

    Official recommendation: folds 1-8 for training, fold 9 for validation,
    fold 10 for testing.
    """
    train_ids = []
    val_ids = []
    test_ids = []

    for idx, row in Y.iterrows():
        fold = row['strat_fold']
        filename = row['filename_hr']

        if fold <= 8:
            train_ids.append(filename)
        elif fold == 9:
            val_ids.append(filename)
        else:
            test_ids.append(filename)

    splits = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }

    split_path = os.path.join(output_dir, 'splits.json')
    with open(split_path, 'w') as f:
        json.dump(splits, f, indent=2)

    print(f"Split files created:")
    print(f"  Train: {len(train_ids)} samples")
    print(f"  Val: {len(val_ids)} samples")
    print(f"  Test: {len(test_ids)} samples")

    return splits


def main(
    path: str,
    output_dir: str,
    target_length: int = 512,
    lowcut: float = 0.05,
    highcut: float = 150.0,
    n_processes: int = None,
):
    """Main preprocessing pipeline.

    Args:
        path: PTB-XL dataset root directory.
        output_dir: Output directory.
        target_length: Target signal length.
        lowcut: Low cutoff frequency (Hz).
        highcut: High cutoff frequency (Hz).
        n_processes: Number of parallel processes.
    """
    assert os.path.isdir(path), f"Path does not exist: {path}"

    print(f"PTB-XL Preprocessing Pipeline")
    print(f"=" * 50)
    print(f"Input path: {path}")
    print(f"Output path: {output_dir}")
    print(f"Target length: {target_length}")
    print(f"Bandpass filter: {lowcut}-{highcut} Hz")
    print(f"=" * 50)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'signals'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'metadata'), exist_ok=True)

    csv_path = os.path.join(path, 'ptbxl_database.csv')
    Y = pd.read_csv(csv_path, index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    print(f"Total records: {len(Y)}")

    scp_dict = prepare_scp_dict(path)

    process_args = []
    for idx, row in Y.iterrows():
        row_dict = row.to_dict()
        row_dict['ecg_id'] = idx
        process_args.append((
            row['filename_hr'],
            path,
            output_dir,
            row_dict,
            target_length,
            lowcut,
            highcut
        ))

    results = []
    failed_files = []

    if n_processes is None or n_processes == 0:
        print("Processing ECG records sequentially...")

        global SCP_DICT
        SCP_DICT = scp_dict

        for args in tqdm(process_args, desc="Processing"):
            filename, success, msg = process_single_record(args)
            if success:
                results.append(filename)
            else:
                failed_files.append((filename, msg))
    else:
        n_processes = min(n_processes, cpu_count(), MAX_PROCESS)
        print(f"Processing ECG records with {n_processes} processes...")

        with Pool(processes=n_processes, initializer=init_worker, initargs=(scp_dict,)) as pool:
            for result in tqdm(
                pool.imap_unordered(process_single_record, process_args),
                total=len(process_args),
                desc="Processing"
            ):
                filename, success, msg = result
                if success:
                    results.append(filename)
                else:
                    failed_files.append((filename, msg))

    create_split_files(Y, output_dir)

    config = {
        'dataset': 'PTB-XL',
        'target_length': target_length,
        'lowcut': lowcut,
        'highcut': highcut,
        'total_records': len(Y),
        'successful': len(results),
        'failed': len(failed_files),
        'lead_order': ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
        'sex_mapping': {'0': 'Male', '1': 'Female'}
    }

    config_path = os.path.join(output_dir, 'preprocessing_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n" + "=" * 50)
    print(f"Processing completed:")
    print(f"  Successfully processed: {len(results)}")
    print(f"  Failed: {len(failed_files)}")

    if failed_files:
        print(f"\nFailed files:")
        for filename, error in failed_files[:10]:
            print(f"  {filename}: {error}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")

        failed_path = os.path.join(output_dir, 'failed_records.json')
        with open(failed_path, 'w') as f:
            json.dump(failed_files, f, indent=2)

    return results, failed_files


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print(f"Arguments: {args}")

    main(
        args.root,
        args.output,
        args.target_length,
        args.lowcut,
        args.highcut,
        args.n_processes
    )

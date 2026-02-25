"""Data preprocessing pipeline for Chapman-Shaoxing-Ningbo (CSN) Database.

Extracts ECG signals and metadata for model training.
"""

import argparse
import os
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

MAX_PROCESS = 24
SNOMED_DICT = None


def init_worker(snomed_dict):
    """Initialize worker process with shared SNOMED dictionary."""
    import warnings
    warnings.filterwarnings('ignore')
    global SNOMED_DICT
    SNOMED_DICT = snomed_dict


def get_parser():
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(description="Preprocess CSN dataset for ECG reconstruction")
    parser.add_argument(
        "--root", type=str, metavar="DIR",
        default=os.path.join(home_dir, "ECG_Dataset/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0"),
        help="Root directory of CSN dataset"
    )
    parser.add_argument(
        "--output", type=str, metavar="DIR",
        default=os.path.join(file_dir, "../data/csn"),
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
    parser.add_argument(
        "--train-ratio", type=float, default=0.7,
        help="Training set ratio"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15,
        help="Validation set ratio"
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


def prepare_snomed_dict(data_dir):
    """Load SNOMED-CT diagnostic code dictionary."""
    snomed_path = os.path.join(data_dir, "ConditionNames_SNOMED-CT.csv")

    snomed_dict = {}

    if os.path.exists(snomed_path):
        df = pd.read_csv(snomed_path)
        for i in range(len(df)):
            snomed_dict[str(df["Snomed_CT"].values[i])] = df['Full Name'].values[i]

    # Additional SNOMED mappings
    extra_mappings = {
        "74390002": "Wolff-Parkinson-White pattern",
        "55827005": "left ventricle hypertrophy",
        "67751000119106": "Right atrial enlargement",
        "251166008": "Atrioventricular nodal re-entry tachycardia",
        "17366009": "Atrial arrhythmia",
        "713427006": "complete right bundle branch block",
        "713426002": "incomplete right bundle branch block",
        "55930002": "ST segment changes",
        "29320008": "Ectopic rhythm",
        "10370003": "Rhythm from artificial pacing",
        "365413008": "R wave",
        "251223006": "Tall P wave",
        "6374002": "Bundle branch block",
        "81898007": "Ventricular escape rhythm",
        "427172004": "premature ventricular contractions",
        "445118002": "Left anterior fascicular block",
        "425856008": "paroxysmal ventricular tachycardia",
        "733534002": "complete left bundle branch block",
        "106068003": "Atrial rhythm",
        "50799005": "Atrioventricular dissociation",
        "61721007": "Counterclockwise vectorcardiographic loop",
        "57054005": "Acute myocardial infarction",
        "426627000": "bradycardia",
        "54329005": "Acute myocardial infarction of anterior wall",
        "251205003": "Prolonged P wave",
        "61277005": "Accelerated idioventricular rhythm",
        "164896001": "ventricular fibrillation",
        "111288001": "Ventricular flutter",
        "233892002": "Ectopic atrial tachycardia",
        "251120003": "Incomplete left bundle branch block",
        "251170000": "Blocked premature atrial contraction",
        "251187003": "Atrial escape complex",
        "418818005": "Brugada syndrome",
        "426183003": "Mobitz type II atrioventricular block",
        "426648003": "junctional tachycardia",
        "426664006": "accelerated junctional rhythm",
        "445211001": "Left posterior fascicular block",
        "446813000": "Left atrial hypertrophy",
        "49578007": "Shortened PR interval",
        "5609005": "Sinus arrest",
        "63593006": "Supraventricular premature beats",
        "65778007": "Sinoatrial block",
        "67741000119109": "Left atrial enlargement",
        "77867006": "Shortened QT interval",
    }
    snomed_dict.update(extra_mappings)

    return snomed_dict


def clean_nan(obj):
    """Recursively replace NaN/Inf floats with None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan(v) for v in obj]
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def parse_comments(comments):
    """Parse the comments field from CSN records to extract metadata.

    Args:
        comments: List of comment strings from wfdb record header.

    Returns:
        Parsed metadata dictionary.
    """
    metadata = {
        'age': None,
        'sex': None,  # 0: Male, 1: Female
        'height': None,
        'weight': None,
        'dx': None,
        'rx': None,
        'hx': None,
        'sx': None,
    }

    for comment in comments:
        if isinstance(comment, str) and ': ' in comment:
            key, value = comment.split(': ', 1)
            key = key.strip('#').strip().lower()
            value = value.strip()

            if key == 'age':
                try:
                    metadata['age'] = float(value) if value.lower() != 'unknown' else None
                except:
                    metadata['age'] = None
            elif key == 'sex':
                if value.lower() in ['female', 'f']:
                    metadata['sex'] = 1
                elif value.lower() in ['male', 'm']:
                    metadata['sex'] = 0
                else:
                    metadata['sex'] = None
            elif key == 'dx':
                metadata['dx'] = value if value.lower() != 'unknown' else None
            elif key == 'rx':
                metadata['rx'] = value if value.lower() != 'unknown' else None
            elif key == 'hx':
                metadata['hx'] = value if value.lower() != 'unknown' else None
            elif key == 'sx':
                metadata['sx'] = value if value.lower() != 'unknown' else None

    return metadata


def process_single_record(args):
    """Process a single ECG record: normalize, filter, resample, and save.

    Args:
        args: Tuple of (filename, data_path, output_dir, target_length,
              lowcut, highcut).

    Returns:
        Tuple of (filename, success, error_message).
    """
    filename, data_path, output_dir, target_length, lowcut, highcut = args

    global SNOMED_DICT

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

        # Parse metadata from comments
        parsed_meta = parse_comments(header.get('comments', []))

        metadata = {
            'record_id': os.path.basename(filename),
            'age': parsed_meta['age'],
            'sex': parsed_meta['sex'],
            'height': None,
            'weight': None,
            'original_fs': original_fs,
            'original_length': header['sig_len'],
            'target_length': target_length,
            'lead_names': header['sig_name'],
            'norm_params': norm_params,
            'filter_params': {'lowcut': lowcut, 'highcut': highcut},
            'nan_filled': nan_count > 0,
            'nan_count': int(nan_count),
        }

        # Extract diagnostic codes
        if parsed_meta['dx']:
            dx_codes = parsed_meta['dx'].split(',')
            metadata['dx_codes'] = dx_codes
            metadata['dx_descriptions'] = {
                code.strip(): SNOMED_DICT.get(code.strip(), 'Unknown')
                for code in dx_codes if code.strip()
            }
        else:
            metadata['dx_codes'] = []
            metadata['dx_descriptions'] = {}

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


def collect_file_list(data_path):
    """Collect all record file paths from the CSN dataset.

    Args:
        data_path: Dataset root directory.

    Returns:
        List of relative file paths (without extension).
    """
    file_list = []
    wfdb_records_path = os.path.join(data_path, "WFDBRecords")

    if not os.path.exists(wfdb_records_path):
        print(f"Warning: WFDBRecords directory not found at {wfdb_records_path}")
        return file_list

    for root, dirs, files in os.walk(wfdb_records_path):
        for file in files:
            if file.endswith('.mat'):
                full_path = os.path.join(root, file)
                full_path_no_ext = os.path.splitext(full_path)[0]
                relative_path = os.path.relpath(full_path_no_ext, data_path)
                file_list.append(relative_path)

    return file_list


def create_split_files(file_list, output_dir, train_ratio=0.7, val_ratio=0.15):
    """Create train/val/test split files with random partitioning.

    Args:
        file_list: List of successfully processed filenames.
        output_dir: Output directory.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
    """
    np.random.seed(42)

    save_names = [f for f in file_list]

    indices = np.random.permutation(len(save_names))

    n_train = int(len(indices) * train_ratio)
    n_val = int(len(indices) * val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    splits = {
        'train': [save_names[i] for i in train_indices],
        'val': [save_names[i] for i in val_indices],
        'test': [save_names[i] for i in test_indices]
    }

    split_path = os.path.join(output_dir, 'splits.json')
    with open(split_path, 'w') as f:
        json.dump(splits, f, indent=2)

    print(f"Split files created:")
    print(f"  Train: {len(splits['train'])} samples ({train_ratio*100:.1f}%)")
    print(f"  Val: {len(splits['val'])} samples ({val_ratio*100:.1f}%)")
    print(f"  Test: {len(splits['test'])} samples ({(1-train_ratio-val_ratio)*100:.1f}%)")

    return splits


def main(
    path: str,
    output_dir: str,
    target_length: int = 512,
    lowcut: float = 0.05,
    highcut: float = 150.0,
    n_processes: int = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    """Main preprocessing pipeline.

    Args:
        path: CSN dataset root directory.
        output_dir: Output directory.
        target_length: Target signal length.
        lowcut: Low cutoff frequency (Hz).
        highcut: High cutoff frequency (Hz).
        n_processes: Number of parallel processes.
        train_ratio: Training set ratio.
        val_ratio: Validation set ratio.
    """
    assert os.path.isdir(path), f"Path does not exist: {path}"

    print(f"CSN Preprocessing Pipeline")
    print(f"=" * 50)
    print(f"Input path: {path}")
    print(f"Output path: {output_dir}")
    print(f"Target length: {target_length}")
    print(f"Bandpass filter: {lowcut}-{highcut} Hz")
    print(f"=" * 50)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'signals'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'metadata'), exist_ok=True)

    print("Collecting file list...")
    file_list = collect_file_list(path)
    print(f"Total records found: {len(file_list)}")

    if len(file_list) == 0:
        print("No records found. Please check the data path.")
        return [], []

    snomed_dict = prepare_snomed_dict(path)

    process_args = [(f, path, output_dir, target_length, lowcut, highcut) for f in file_list]

    results = []
    failed_files = []

    if n_processes is None or n_processes == 0:
        print("Processing ECG records sequentially...")

        global SNOMED_DICT
        SNOMED_DICT = snomed_dict

        for args in tqdm(process_args, desc="Processing"):
            filename, success, msg = process_single_record(args)
            if success:
                results.append(filename)
            else:
                failed_files.append((filename, msg))
    else:
        n_processes = min(n_processes, cpu_count(), MAX_PROCESS)
        print(f"Processing ECG records with {n_processes} processes...")

        with Pool(processes=n_processes, initializer=init_worker, initargs=(snomed_dict,)) as pool:
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

    create_split_files(results, output_dir, train_ratio, val_ratio)

    config = {
        'dataset': 'CSN',
        'target_length': target_length,
        'lowcut': lowcut,
        'highcut': highcut,
        'total_records': len(file_list),
        'successful': len(results),
        'failed': len(failed_files),
        'lead_order': ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
        'sex_mapping': {'0': 'Male', '1': 'Female'},
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
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
        args.n_processes,
        args.train_ratio,
        args.val_ratio,
    )

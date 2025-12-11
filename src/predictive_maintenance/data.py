import gc
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Optional, Tuple, List, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tensorflow as tf
from keras.utils import timeseries_dataset_from_array
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from .config import Config

# --- CONFIGURATION & LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

FAILURE_MAP = {1: [3, 4], 2: [1], 3: [3]}
TEST_CONFIGS = [
    ("1st_test", 1, 4),
    ("2nd_test", 2, 4),
    ("3rd_test", 3, 4),
]


# --- CUSTOM EXCEPTIONS ---
class ETLException(Exception):
    """Custom exception for errors during the ETL process."""
    pass


# --- MAIN ETL CLASS ---
class IMSDataETL:
    def __init__(self):
        self.scaler = MinMaxScaler()

    @staticmethod
    def get_healthy_bearings() -> List[Tuple[int, int]]:
        """Identifies bearings that never failed throughout the test."""
        logger.info(">>> Scanning for healthy bearings...")

        df_meta = pd.read_parquet(
            Config.DATA_PATH,
            columns=["test_id", "bearing_id", "is_failure"]
        )

        # Vectorized check for failures
        failures = df_meta[df_meta["is_failure"]].groupby(["test_id", "bearing_id"], observed=True).any()
        all_bearings = df_meta[["test_id", "bearing_id"]].drop_duplicates().set_index(["test_id", "bearing_id"])

        # Filter healthy ones
        healthy_mask = ~all_bearings.index.isin(failures.index)
        healthy_indices = all_bearings.index[healthy_mask].tolist()

        logger.info(f"✔ Healthy bearings list: {healthy_indices}")
        return healthy_indices

    @staticmethod
    def calculate_features(signal: np.ndarray) -> List[float]:
        """Calculates statistical features for a given signal."""
        # Ensure input is suitable for calculation
        signal = np.asarray(signal, dtype=np.float32)
        abs_sig = np.abs(signal)

        # Standard statistical features
        mav = np.mean(abs_sig)
        rms = np.sqrt(np.mean(signal ** 2))
        p2p = np.ptp(signal)
        kur = kurtosis(signal)
        skw = skew(signal)
        cf = np.max(abs_sig) / (rms + 1e-6)

        return [mav, rms, p2p, kur, skw, cf]

    def load_features(self, bearing_filter: Optional[List[Tuple[int, int]]] = None) -> Tuple[pd.DataFrame, np.ndarray]:
        """Loads data from Parquet and computes features."""
        filters = None
        if bearing_filter:
            tids = list({b[0] for b in bearing_filter})
            filters = [("test_id", "in", tids)]

        df = pd.read_parquet(
            Config.DATA_PATH,
            columns=["test_id", "bearing_id", "timestamp", "vibration", "is_failure"],
            filters=filters,
        )

        df.sort_values(["test_id", "bearing_id", "timestamp"], inplace=True)

        if bearing_filter:
            # Create a MultiIndex mask for faster filtering than list comprehension
            index_tuples = list(zip(df["test_id"], df["bearing_id"]))
            filter_set = set(bearing_filter)
            mask = [x in filter_set for x in index_tuples]
            df = df[mask]

        # Extract signals and compute features
        # Note: Using list comprehension to maintain strict API return type compatibility per requirements
        raw_signals = df["vibration"].values
        features_list = [self.calculate_features(sig) for sig in raw_signals]
        features = np.array(features_list, dtype=np.float32)

        return df, features

    def generate_training_dataset(self) -> tf.data.Dataset:
        """Generates a TensorFlow dataset for training."""
        healthy = self.get_healthy_bearings()
        df, feats = self.load_features(healthy)

        logger.info("Fitting Scaler...")
        self.scaler.fit(feats)

        full_ds = None
        # Group indices to avoid splitting dataframes repeatedly
        groups = df.groupby(["test_id", "bearing_id"], observed=True).indices

        for _, idx in groups.items():
            vals = feats[idx]
            limit = int(len(vals) * Config.TRAIN_RATIO)
            train_vals = vals[:limit]

            if len(train_vals) < Config.WINDOW:
                continue

            scaled = self.scaler.transform(train_vals)

            ds = timeseries_dataset_from_array(
                data=scaled,
                targets=None,
                sequence_length=Config.WINDOW,
                sequence_stride=1,
                batch_size=Config.BATCH,
                shuffle=True,
            )

            # Autoencoder format: (input, target) -> (x, x)
            ds = ds.map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE)
            full_ds = ds if full_ds is None else full_ds.concatenate(ds)

        return full_ds.prefetch(tf.data.AUTOTUNE)

    def load_test_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        return self.load_features(bearing_filter=None)


# --- HELPER FUNCTIONS ---
def parse_filename_to_timestamp(filename: str) -> datetime:
    try:
        # Optimized parsing assuming consistent format
        base = os.path.basename(filename)
        parts = base.split(".")
        if len(parts) >= 6:
            return datetime(*map(int, parts[:6]))
        return datetime.now()
    except Exception as e:
        logger.error(f"Timestamp parsing error: {filename} - {e}")
        raise ETLException(f"Invalid file format: {filename}")


# --- MODULE 1: FEATURE EXTRACTION (PCA) ---
def _process_single_file_pca(args: Tuple[Path, Path]) -> Optional[str]:
    input_path, output_dir = args
    try:
        # Optimization: Use pandas read_csv (C engine) instead of np.genfromtxt
        df_raw = pd.read_csv(input_path, sep=r'\s+', header=None, dtype=np.float32)
        data = df_raw.values

        if data.shape[1] < 8:
            return f"Missing columns: {input_path.name}"

        processed_channels = []
        pca = PCA(n_components=1)

        # Process 4 bearings (2 columns each)
        for i in range(4):
            segment = data[:, i * 2: (i + 1) * 2]
            transformed = pca.fit_transform(segment).flatten()
            processed_channels.append(transformed)

        result_data = np.column_stack(processed_channels)
        output_filepath = output_dir / input_path.name

        # Save efficiently
        pd.DataFrame(result_data).to_csv(output_filepath, sep="\t", header=False, index=False, float_format="%.6f")

        return None
    except Exception as e:
        return f"Error ({input_path.name}): {str(e)}"


def run_pca_pipeline(input_dir_str: str, output_dir_str: str, n_workers: int = -1) -> None:
    input_dir = Path(input_dir_str)
    output_dir = Path(output_dir_str)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(list(input_dir.glob("*")))
    if not files:
        logger.warning("No files found to process.")
        return

    logger.info(f"PCA Process Starting. File count: {len(files)}")

    tasks = [(f, output_dir) for f in files]
    max_workers = os.cpu_count() if n_workers == -1 else n_workers

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(_process_single_file_pca, tasks),
            total=len(files),
            desc="PCA Transformation"
        ))

    errors = [res for res in results if res is not None]
    if errors:
        logger.error(f"Errors occurred in {len(errors)} files.")
        for err in errors[:5]:
            logger.error(err)
    else:
        logger.info("PCA process completed successfully.")


# --- MODULE 2: DATA CLEANING ---
def clean_text_data(source_dir: str, target_dir: str) -> None:
    os.makedirs(target_dir, exist_ok=True)
    files = sorted(glob(os.path.join(source_dir, "[0-9.]*")))

    if not files:
        logger.warning(f"No data to clean in: {source_dir}")
        return

    logger.info(f"Data cleaning starting: {len(files)} files.")

    for source_path in tqdm(files, desc="Cleaning Text"):
        filename = os.path.basename(source_path)
        target_path = os.path.join(target_dir, filename)

        try:
            # Optimization: Read/Write usually doesn't require line-by-line python loop for simple space cleanup
            # using pandas to normalize delimiters is faster and safer
            df = pd.read_csv(source_path, sep=r'\s+', header=None, engine='c')
            df.to_csv(target_path, header=False, index=False)
        except Exception as e:
            logger.error(f"Cleaning error in {filename}: {e}")


# --- MODULE 3: RAW TO PARQUET (ETL & Storage) ---
def create_parquet_dataset(base_input_path: str, output_parquet_path: str, batch_size: int = 500) -> None:
    logger.info("-" * 50)
    logger.info(f"Parquet Conversion Starting (Batch Size: {batch_size})")

    if os.path.exists(output_parquet_path):
        shutil.rmtree(output_parquet_path)

    total_files = 0

    for dir_name, test_id, num_channels in TEST_CONFIGS:
        test_path = os.path.join(base_input_path, dir_name)
        if not os.path.isdir(test_path):
            continue

        files = sorted(glob(os.path.join(test_path, "[0-9.]*")))
        if not files:
            continue

        last_ts = parse_filename_to_timestamp(files[-1])
        failed_bearings = set(FAILURE_MAP.get(test_id, []))

        logger.info(f"Processing Test {test_id} ({len(files)} files)")

        for i in tqdm(range(0, len(files), batch_size), desc=f"Test {test_id}"):
            batch_files = files[i: i + batch_size]

            timestamps, test_ids, bearing_ids, is_failures, vibration_data = [], [], [], [], []

            for f_path in batch_files:
                try:
                    ts = parse_filename_to_timestamp(f_path)

                    # --- FIX START: ROBUST READ STRATEGY ---
                    try:
                        # 1. Try reading as standard CSV (Comma separated) - FASTEST
                        df_raw = pd.read_csv(f_path, sep=',', header=None, dtype=np.float32)

                        # Check if it failed to split (e.g. read as 1 column instead of 4)
                        if df_raw.shape[1] < num_channels:
                            raise ValueError("Wrong separator")

                    except ValueError:
                        # 2. Fallback to Whitespace separated (Original IMS format)
                        df_raw = pd.read_csv(f_path, sep=r'\s+', header=None, dtype=np.float32)

                    raw_vals = df_raw.values
                    # --- FIX END ---

                    for ch_idx in range(num_channels):
                        # Ensure we don't access out of bounds if file is corrupted
                        if ch_idx >= raw_vals.shape[1]:
                            continue

                        b_id = ch_idx + 1
                        fail_flag = (ts == last_ts) and (b_id in failed_bearings)

                        timestamps.append(ts)
                        test_ids.append(test_id)
                        bearing_ids.append(b_id)
                        is_failures.append(fail_flag)
                        vibration_data.append(raw_vals[:, ch_idx])

                except Exception as e:
                    logger.error(f"Read error {f_path}: {e}")

            if timestamps:
                try:
                    table = pa.Table.from_arrays(
                        arrays=[
                            pa.array(timestamps),
                            pa.array(test_ids, type=pa.int8()),
                            pa.array(bearing_ids, type=pa.int8()),
                            pa.array(is_failures, type=pa.bool_()),
                            pa.array(vibration_data, type=pa.list_(pa.float32())),
                        ],
                        names=["timestamp", "test_id", "bearing_id", "is_failure", "vibration"],
                    )

                    pq.write_to_dataset(
                        table,
                        root_path=output_parquet_path,
                        partition_cols=["test_id", "bearing_id"],
                        compression="zstd",
                        existing_data_behavior="overwrite_or_ignore",
                    )
                    total_files += len(batch_files)
                except Exception as e:
                    logger.error(f"Batch save error: {e}")

            del timestamps, test_ids, bearing_ids, is_failures, vibration_data
            if i % (batch_size * 5) == 0:
                gc.collect()

    logger.info(f"Process Completed. Files Processed: {total_files}")
    logger.info(f"Output Path: {output_parquet_path}")
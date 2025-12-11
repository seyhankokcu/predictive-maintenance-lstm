import random
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import scipy.stats as stats

from predictive_maintenance import Config

plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "seaborn-whitegrid"
)
print(f"Figures will be saved to: '{Config.FIGURES_DIR.absolute()}'")


def _load_signal_csv(
    file_path: Path, use_cols: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Loads raw signal CSV files efficiently.
    Args:
        file_path (Path): Path to the CSV file.
        use_cols (list, optional): List of column indices to load.
    Returns:
        pd.DataFrame: Loaded data.
    """
    try:
        return pd.read_csv(
            file_path, sep=r"\s+", header=None, engine="c", usecols=use_cols
        )
    except Exception as e:
        raise IOError(f"Error reading {file_path.name}: {e}")


def _insert_time_gaps(
    df: pd.DataFrame,
    time_col: str = "timestamp",
    threshold: pd.Timedelta = Config.GAP_THRESHOLD,
) -> pd.DataFrame:
    """
    Identifies discontinuities in the time series and inserts NaN rows.
    This prevents Matplotlib from drawing straight lines across large time gaps,
    ensuring visually accurate plots.
    Args:
        df (pd.DataFrame): Input dataframe containing a timestamp column.
        time_col (str): Name of the timestamp column.
        threshold (pd.Timedelta): The duration above which a gap is considered a break.
    Returns:
        pd.DataFrame: DataFrame with NaNs inserted at gap locations.
    """
    df = df.sort_values(time_col).copy()
    time_diffs = df[time_col].diff()
    gap_mask = time_diffs > threshold
    if not gap_mask.any():
        return df
    gap_indices = df.index[gap_mask]
    gaps_df = df.loc[gap_indices].copy()
    cols_to_nan = [c for c in df.columns if c != time_col]
    gaps_df[cols_to_nan] = np.nan
    gaps_df[time_col] -= pd.Timedelta(seconds=1)
    try:
        gaps_df = gaps_df.astype(df.dtypes)
    except Exception:
        pass
    return pd.concat([df, gaps_df]).sort_values(time_col).reset_index(drop=True)


def calculate_all_features(
    df_input: pd.DataFrame, sampling_rate: int = Config.SAMPLING_RATE
) -> pd.DataFrame:
    """
    Extracts Time and Frequency domain features using vectorized NumPy operations.
    Features extracted:
    - Time: RMS, Peak-to-Peak, Kurtosis, Skewness, Crest Factor.
    - Frequency: Spectral Kurtosis, Band Power (0-2k, 2-4k, 4-6k, 6-10k Hz).
    Args:
        df_input (pd.DataFrame): Input dataframe with 'vibration' column containing lists/arrays.
        sampling_rate (int): Sampling frequency in Hz.
    Returns:
        pd.DataFrame: DataFrame enriched with feature columns.
    """
    df = df_input.copy()
    try:
        if "vibration" not in df.columns or df["vibration"].empty:
            return df

        signal_matrix = np.vstack(df["vibration"].tolist())
        _, n_points = signal_matrix.shape

        sq_signal = np.square(signal_matrix)
        rms_vals = np.sqrt(np.mean(sq_signal, axis=1))
        df["rms"] = rms_vals
        df["peak_to_peak"] = np.ptp(signal_matrix, axis=1)
        df["kurtosis"] = stats.kurtosis(signal_matrix, axis=1)
        df["skewness"] = stats.skew(signal_matrix, axis=1)

        peak_abs_vals = np.max(np.abs(signal_matrix), axis=1)
        df["crest_factor"] = np.divide(peak_abs_vals, rms_vals + 1e-9)

        fft_matrix = np.fft.rfft(signal_matrix, axis=1)
        fft_abs = np.abs(fft_matrix)
        freqs = np.fft.rfftfreq(n_points, d=1 / sampling_rate)
        df["spectral_kurtosis"] = stats.kurtosis(fft_abs, axis=1)

        bands = {
            "power_band_0_2k": (0, 2000),
            "power_band_2_4k": (2000, 4000),
            "power_band_4_6k": (4000, 6000),
            "power_band_6_10k": (6000, 10000),
        }
        fft_sq = np.square(fft_abs)
        for name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs < high)
            if np.any(mask):
                df[name] = np.sqrt(np.mean(fft_sq[:, mask], axis=1))
            else:
                df[name] = 0.0
    except Exception as e:
        print(f"Feature calculation failed: {e}")
    return df


def calculate_wavelet_features(
    df_input: pd.DataFrame, wavelet: str = "db8", level: int = 5
) -> pd.DataFrame:
    """
    Performs Discrete Wavelet Transform (DWT) to extract energy features.
    Args:
        df_input (pd.DataFrame): DataFrame containing vibration signals.
        wavelet (str): Wavelet family (e.g., 'db8').
        level (int): Decomposition level.
    Returns:
        pd.DataFrame: DataFrame with added wavelet energy columns.
    """
    df = df_input.copy()
    try:
        if "vibration" not in df.columns:
            return df
        signal_matrix = np.vstack(df["vibration"].tolist())
        coeffs = pywt.wavedec(signal_matrix, wavelet, level=level, axis=1)

        df[f"wavelet_energy_A{level}"] = np.sqrt(np.mean(np.square(coeffs[0]), axis=1))

        for i, coeff_array in enumerate(coeffs[1:]):
            current_level = level - i
            df[f"wavelet_energy_D{current_level}"] = np.sqrt(
                np.mean(np.square(coeff_array), axis=1)
            )
    except Exception as e:
        print(f"Wavelet calculation failed: {e}")
    return df


def _generic_signal_plotter(
    df: pd.DataFrame,
    features: List[str],
    title: str,
    filename_suffix: str,
    y_limits: Optional[Dict[str, Tuple[float, float]]] = None,
    plot_color: str = "dodgerblue",
) -> None:
    """
    Generates a grid of subplots for specified features across all bearings.
    Args:
        df (pd.DataFrame): Data to plot.
        features (list): List of feature column names.
        title (str): Main title of the figure.
        filename_suffix (str): Output filename.
        y_limits (dict, optional): specific Y-axis limits for features.
        plot_color (str): Hex code or name for plot line color.
    """
    n_rows = 4
    n_cols = len(features)
    if n_cols == 0:
        return
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        sharex=True,
        layout="constrained",
    )

    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    if y_limits is None:
        y_limits = {}
        for feat in features:
            if feat in df.columns:
                f_min, f_max = df[feat].min(), df[feat].max()
                y_limits[feat] = (f_min, f_max * 1.1)
    for b_idx, bearing_id in enumerate(range(1, 5)):
        df_b = df[df["bearing_id"] == bearing_id].sort_values("timestamp")

        is_fail = df_b["is_failure"].any() if "is_failure" in df_b.columns else False
        row_label_color = "red" if is_fail else "green"
        axes[b_idx, 0].set_ylabel(
            f"B{bearing_id}\n({'FAIL' if is_fail else 'OK'})",
            fontsize=14,
            color=row_label_color,
            weight="bold",
        )
        if df_b.empty:
            continue

        df_plot = _insert_time_gaps(df_b, "timestamp")
        fail_pts = df_b[df_b["is_failure"]] if is_fail else pd.DataFrame()
        for f_idx, feat in enumerate(features):
            if feat not in df_plot.columns:
                continue
            ax = axes[b_idx, f_idx]

            if b_idx == 0:
                ax.set_title(
                    feat.replace("_", " ").replace("power band", "Band").title(),
                    fontsize=14,
                    weight="bold",
                )

            ax.plot(
                df_plot["timestamp"],
                df_plot[feat],
                color=plot_color,
                alpha=0.8,
                linewidth=1,
            )

            if not fail_pts.empty:
                ax.scatter(
                    fail_pts["timestamp"],
                    fail_pts[feat],
                    color="red",
                    s=60,
                    edgecolors="black",
                    zorder=5,
                )
            if feat in y_limits:
                ax.set_ylim(y_limits[feat])
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="x", rotation=30)
    fig.suptitle(title, fontsize=20, weight="bold")
    save_path = Config.FIGURES_DIR / filename_suffix
    fig.savefig(save_path, dpi=300)
    print(f"Saved figure: {save_path.name}")
    plt.show()
    plt.close(fig)


def plot_pca_vs_raw_comparison(
    base_dir: str, pca_dir: str, bearing_id: int = 1
) -> None:
    """
    Compares the raw vibration signal against PCA-reduced signal components.
    """
    base_path, pca_path = Path(base_dir), Path(pca_dir)
    print(f"\n--- PCA vs Raw Comparison (Bearing {bearing_id}) ---")
    test_map = {"t1": "1st_test", "t2": "2nd_test", "t3": "3rd_test"}
    try:
        files = {}
        for k, folder in test_map.items():
            p = base_path / folder
            if p.exists():
                file_list = sorted(list(p.glob("*")))
                if file_list:
                    files[k] = random.choice(file_list)
                else:
                    print(f"No files found in {p}")
                    return
            else:
                print(f"Missing path: {p}")
                return

        df_t1 = _load_signal_csv(files["t1"])
        df_t2 = _load_signal_csv(files["t2"])
        df_t3 = _load_signal_csv(files["t3"])
        b_idx_start = (bearing_id - 1) * 2
        if df_t1.shape[1] <= b_idx_start + 1:
            return
        signals = {
            "t1_x": df_t1.iloc[:, b_idx_start],
            "t1_y": df_t1.iloc[:, b_idx_start + 1],
            "t2": df_t2.iloc[:, bearing_id - 1],
            "t3": df_t3.iloc[:, bearing_id - 1],
        }

        fig1, axs1 = plt.subplots(
            3, 1, figsize=(12, 8), sharex=True, layout="constrained"
        )
        fig1.suptitle(f"Raw Signals (Bearing {bearing_id})", fontsize=14, weight="bold")
        axs1[0].plot(signals["t1_x"], label="X", alpha=0.7)
        axs1[0].plot(signals["t1_y"], label="Y", alpha=0.7)
        axs1[0].set_title(f"Test 1 - {files['t1'].name}")
        axs1[0].legend()
        axs1[1].plot(signals["t2"], color="#2ca02c")
        axs1[1].set_title("Test 2 (Reference)")
        axs1[2].plot(signals["t3"], color="#9467bd")
        axs1[2].set_title("Test 3 (Reference)")
        fig1.savefig(
            Config.FIGURES_DIR / f"raw_signals_bearing_{bearing_id}.png", dpi=150
        )
        plt.show()

        file_pca = pca_path / files["t1"].name
        if file_pca.exists():
            df_pca = _load_signal_csv(file_pca)
            pca_col = 0 if df_pca.shape[1] == 1 else (bearing_id - 1)
            fig2, axs2 = plt.subplots(
                3, 1, figsize=(12, 8), sharex=True, layout="constrained"
            )
            fig2.suptitle(
                f"PCA Processed (Bearing {bearing_id})", fontsize=14, weight="bold"
            )
            axs2[0].plot(df_pca.iloc[:, pca_col], color="#d62728", label="PCA")
            axs2[0].legend()
            axs2[1].plot(signals["t2"], color="#2ca02c")
            axs2[2].plot(signals["t3"], color="#9467bd")
            fig2.savefig(
                Config.FIGURES_DIR / f"pca_signals_bearing_{bearing_id}.png", dpi=150
            )
            plt.show()
    except Exception:
        traceback.print_exc()


def visualize_timestamp_continuity(dataset_path: str) -> None:
    """
    Visualizes the density and continuity of timestamps across tests and bearings.
    Useful for identifying missing data periods.
    """
    print("Starting Timestamp Continuity Analysis...")
    fig, axes = plt.subplots(4, 3, figsize=(18, 12), layout="constrained")
    for test_id in range(1, 4):
        try:
            df_full = pd.read_parquet(
                dataset_path,
                filters=[("test_id", "==", test_id)],
                columns=["bearing_id", "timestamp"],
            )
        except Exception as e:
            print(f"Could not load Test {test_id}: {e}")
            continue
        for bearing_id in range(1, 5):
            ax = axes[bearing_id - 1, test_id - 1]
            df = df_full[df_full["bearing_id"] == bearing_id].sort_values("timestamp")
            if df.empty:
                continue
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            # Plot continuous line for data presence
            ax.plot(df["timestamp"], np.ones(len(df)), color="dodgerblue", lw=3)
            # Mark gaps specifically
            time_diffs = df["timestamp"].diff()
            gap_indices = df.index[time_diffs > Config.GAP_THRESHOLD]
            for idx in gap_indices:
                t_end = df.loc[idx, "timestamp"]
                t_start = df.loc[idx - 1, "timestamp"] if idx > 0 else t_end
                ax.plot([t_start, t_end], [1, 1], color="red", ls="--", lw=2)
            if bearing_id == 1:
                ax.set_title(f"Test {test_id}", weight="bold")
            if test_id == 1:
                ax.set_ylabel(f"Bearing {bearing_id}", weight="bold")
            ax.set_yticks([])
            ax.tick_params(axis="x", rotation=30, labelsize=8)
    fig.suptitle("Timestamp Continuity Analysis", fontsize=16, weight="bold")
    fig.savefig(Config.FIGURES_DIR / "timestamp_continuity_analysis.png", dpi=300)
    plt.show()


def plot_experiment_eda_with_fixed_scales(test_id: int, dataset_path: str) -> None:
    """Wrapper for plotting Time-Domain features."""
    print(f"\n===== Time-Domain Analysis: Test {test_id} =====")
    try:
        df = pd.read_parquet(dataset_path, filters=[("test_id", "==", test_id)])
        if df.empty:
            return
        df = calculate_all_features(df)
        features = ["rms", "kurtosis", "crest_factor", "skewness", "peak_to_peak"]
        _generic_signal_plotter(
            df,
            features,
            title=f"Time Domain Features | Test {test_id}",
            filename_suffix=f"experiment_{test_id}_time_features.png",
            plot_color="dodgerblue",
        )
    except Exception:
        traceback.print_exc()


def plot_frequency_features_comparison(test_id: int, dataset_path: str) -> None:
    """Wrapper for plotting Frequency-Domain features."""
    print(f"\n===== Frequency Analysis: Test {test_id} =====")
    try:
        df = pd.read_parquet(dataset_path, filters=[("test_id", "==", test_id)])
        if df.empty:
            return
        df = calculate_all_features(df)
        features = [
            "spectral_kurtosis",
            "power_band_0_2k",
            "power_band_2_4k",
            "power_band_4_6k",
            "power_band_6_10k",
        ]
        _generic_signal_plotter(
            df,
            features,
            title=f"Frequency Domain Features | Test {test_id}",
            filename_suffix=f"experiment_{test_id}_freq_features.png",
            plot_color="purple",
        )
    except Exception:
        traceback.print_exc()


def plot_wavelet_features_comparison(test_id: int, dataset_path: str) -> None:
    """Wrapper for plotting Wavelet Energy features."""
    print(f"\n===== Wavelet Analysis: Test {test_id} =====")
    try:
        df = pd.read_parquet(dataset_path, filters=[("test_id", "==", test_id)])
        if df.empty:
            return
        LEVEL = 5
        df = calculate_wavelet_features(df, wavelet="db8", level=LEVEL)
        features = [f"wavelet_energy_A{LEVEL}"] + [
            f"wavelet_energy_D{i}" for i in range(LEVEL, 0, -1)
        ]
        _generic_signal_plotter(
            df,
            features,
            title=f"Wavelet Energy (db8 L{LEVEL}) | Test {test_id}",
            filename_suffix=f"experiment_{test_id}_wavelet_features.png",
            plot_color="teal",
        )
    except Exception:
        traceback.print_exc()


def plot_discriminator_analysis(test_id: int, dataset_path: str) -> None:
    """
    Analyzes the ratio of Faulty Bearing Features to Healthy Bearing Means.
    This helps visualize how distinguishable the failure is.
    """
    print(f"\n===== Discriminator Analysis: Test {test_id} =====")

    failure_map = {1: [3, 4], 2: [1], 3: [3]}
    failed_bearings = failure_map.get(test_id, [])
    if not failed_bearings:
        return
    try:
        df = pd.read_parquet(dataset_path, filters=[("test_id", "==", test_id)])
        if df.empty:
            return
        df = calculate_all_features(df)

        df = df.drop_duplicates(subset=["timestamp", "bearing_id"])

        df_pivot = df.pivot(
            index="timestamp", columns="bearing_id", values=["rms", "kurtosis"]
        )
        df_pivot.columns = [f"{col[0]}_{col[1]}" for col in df_pivot.columns]
        df_pivot = df_pivot.sort_index()
        healthy_ids = [b for b in [1, 2, 3, 4] if b not in failed_bearings]
        for fail_id in failed_bearings:
            df_pivot["h_rms_mean"] = df_pivot[[f"rms_{b}" for b in healthy_ids]].mean(
                axis=1
            )
            df_pivot["h_kurt_mean"] = df_pivot[
                [f"kurtosis_{b}" for b in healthy_ids]
            ].mean(axis=1)

            ratio_rms = df_pivot[f"rms_{fail_id}"] / (df_pivot["h_rms_mean"] + 1e-9)
            ratio_kurt = df_pivot[f"kurtosis_{fail_id}"] / (
                df_pivot["h_kurt_mean"] + 1e-9
            )

            df_ratio = pd.DataFrame(
                {
                    "timestamp": df_pivot.index,
                    "rms_r": ratio_rms.values,
                    "kurt_r": ratio_kurt.values,
                }
            )
            df_ratio = _insert_time_gaps(df_ratio)

            fig, axes = plt.subplots(2, 2, figsize=(18, 10), layout="constrained")

            for b in range(1, 5):
                col_name = f"rms_{b}"
                if col_name in df_pivot.columns:
                    axes[0, 0].plot(df_pivot.index, df_pivot[col_name], label=f"B{b}")
            axes[0, 0].set_title("Raw RMS")
            axes[0, 0].legend()

            axes[0, 1].plot(df_ratio["timestamp"], df_ratio["rms_r"], color="crimson")
            axes[0, 1].axhline(1, color="black", ls="--")
            axes[0, 1].set_title(f"RMS Ratio (Fault B{fail_id} / Healthy Mean)")

            for b in range(1, 5):
                col_name = f"kurtosis_{b}"
                if col_name in df_pivot.columns:
                    axes[1, 0].plot(df_pivot.index, df_pivot[col_name], label=f"B{b}")
            axes[1, 0].set_title("Raw Kurtosis")
            axes[1, 0].legend()

            axes[1, 1].plot(df_ratio["timestamp"], df_ratio["kurt_r"], color="crimson")
            axes[1, 1].axhline(1, color="black", ls="--")
            axes[1, 1].set_title(f"Kurtosis Ratio (Fault B{fail_id} / Healthy Mean)")
            fig.suptitle(
                f"Discriminator Analysis | Test {test_id} | Target: B{fail_id}",
                fontsize=20,
                weight="bold",
            )
            save_path = (
                Config.FIGURES_DIR / f"experiment_{test_id}_disc_bearing_{fail_id}.png"
            )
            fig.savefig(save_path, dpi=300)
            print(f"Saved discriminator plot for Bearing {fail_id}")
            plt.show()
            plt.close(fig)
    except Exception:
        traceback.print_exc()

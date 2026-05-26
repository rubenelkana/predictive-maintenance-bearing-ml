"""Unified feature extraction across NASA IMS Sets 1, 2, 3 — for LOBO experiment in notebook 05.

Walks all 12 bearings, extracts the 6-D physics-informed feature vector per
40 ms window, and derives bearing-level 3-class labels (Normal / Degraded /
Critical) via EWMA on each failure bearing's envelope RMS. Non-failure
bearings are Normal throughout.

Following Darlami & Awasthi 2026 LOBO protocol — uses only the FIRST
accelerometer channel per bearing (Set 1 has dual-channel; Sets 2 & 3 have
single-channel; consistency rule: take channel 1).

Documented failures:
  Set 1 (Oct-Nov 2003, 2156 files, 8 channels):
    B1 normal / B2 normal / B3 inner race / B4 ball element
  Set 2 (Feb 2004, 984 files, 4 channels):
    B1 outer race / B2-B4 normal
  Set 3 (Mar-Apr 2004, 6324 files, 4 channels):
    B1-B2 normal / B3 outer race / B4 normal

Output: data/processed/all_sets_features.parquet — ~946k rows × 14 cols.
"""

from __future__ import annotations

import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import kurtosis


# --- paths ---
DATA_ROOT = Path.home() / ".cache/kagglehub/datasets/vinayak123tyagi/bearing-dataset/versions/1"
SET_DIRS = {
    1: DATA_ROOT / "1st_test" / "1st_test",
    2: DATA_ROOT / "2nd_test" / "2nd_test",
    3: DATA_ROOT / "3rd_test" / "4th_test" / "txt",
}
# Set 1 has 8 cols (2 channels per bearing); we take cols [0,2,4,6] for B1..B4 channel-1.
# Sets 2 and 3 have 4 cols straight: [0,1,2,3] for B1..B4.
CHANNEL_LAYOUT = {
    1: [0, 2, 4, 6],
    2: [0, 1, 2, 3],
    3: [0, 1, 2, 3],
}
N_CHANNELS_RAW = {1: 8, 2: 4, 3: 4}

# Failure bearings per set (1-indexed). Empty list = no failure that set.
FAILURE_BEARINGS = {
    1: [3, 4],
    2: [1],
    3: [3],
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# --- constants ---
SAMPLE_RATE_HZ = 20_000
SAMPLES_PER_FILE = 20_480
WINDOW_SAMPLES = 800
WINDOWS_PER_FILE = SAMPLES_PER_FILE // WINDOW_SAMPLES
BAND_HALFWIDTH_HZ = 15.0
HF_BAND = (2000.0, 10000.0)


# --- bearing fault frequencies (canonical Rexnord ZA-2115 at 2000 RPM) ---
def fault_frequencies(N=16, B_d=0.331, P_d=2.815, phi_deg=15.17, rpm=2000):
    n = rpm / 60.0
    ratio = B_d / P_d
    cos_phi = math.cos(math.radians(phi_deg))
    ftf = 0.5 * (1 - ratio * cos_phi) * n
    return {
        "shaft": n,
        "FTF": ftf,
        "BPFO": N * ftf,
        "BPFI": 0.5 * N * (1 + ratio * cos_phi) * n,
        "BSF": (P_d / (2 * B_d)) * (1 - (ratio * cos_phi) ** 2) * n,
    }


FF = fault_frequencies()
print(f"Fault frequencies: BPFO={FF['BPFO']:.1f} Hz, BPFI={FF['BPFI']:.1f} Hz")


# --- precompute FFT band masks ---
FFT_FREQS = np.fft.rfftfreq(WINDOW_SAMPLES, d=1.0 / SAMPLE_RATE_HZ)


def _mask(low: float, high: float) -> np.ndarray:
    return (FFT_FREQS >= low) & (FFT_FREQS <= high)


BPFO_MASK = _mask(FF["BPFO"] - BAND_HALFWIDTH_HZ, FF["BPFO"] + BAND_HALFWIDTH_HZ)
BPFI_MASK = _mask(FF["BPFI"] - BAND_HALFWIDTH_HZ, FF["BPFI"] + BAND_HALFWIDTH_HZ)
HF_MASK = _mask(*HF_BAND)
print(f"FFT masks: BPFO {BPFO_MASK.sum()} bins, BPFI {BPFI_MASK.sum()} bins, HF {HF_MASK.sum()} bins")


def window_features(x: np.ndarray) -> tuple[float, float, float, float, float, float]:
    env = np.abs(hilbert(x))
    f1 = float(np.sqrt(np.mean(env ** 2)))
    f2 = float(kurtosis(x, fisher=False))
    spec = np.abs(np.fft.rfft(x - x.mean())) ** 2
    f3 = float(np.log1p(spec[BPFO_MASK].sum()))
    f4 = float(np.log1p(spec[HF_MASK].sum()))
    f5 = float(np.log1p(spec[BPFI_MASK].sum()))
    f6 = float(np.sqrt(np.mean(x ** 2)))
    return f1, f2, f3, f4, f5, f6


FEATURE_NAMES = ["env_rms", "kurtosis", "log_e_bpfo", "log_e_hf", "log_e_bpfi", "raw_rms"]


def list_files(set_dir: Path) -> list[Path]:
    files = [p for p in set_dir.iterdir() if not p.name.startswith(".") and p.is_file()]
    files.sort(key=lambda p: p.name)
    return files


def extract_set(set_num: int) -> pd.DataFrame:
    set_dir = SET_DIRS[set_num]
    files = list_files(set_dir)
    cols = CHANNEL_LAYOUT[set_num]
    n_raw = N_CHANNELS_RAW[set_num]

    print(f"\n--- Set {set_num}: {len(files)} files ---")
    rows = []
    t0 = time.time()
    for i, fp in enumerate(files):
        timestamp = datetime.strptime(fp.name, "%Y.%m.%d.%H.%M.%S")
        arr = np.loadtxt(fp)
        if arr.ndim == 1:
            arr = arr[:, None]
        assert arr.shape[1] == n_raw, f"set {set_num} file {fp.name}: expected {n_raw} cols, got {arr.shape[1]}"
        for b_idx, col in enumerate(cols, start=1):
            sig = arr[:, col]
            # slice into WINDOWS_PER_FILE non-overlapping windows
            for w in range(WINDOWS_PER_FILE):
                start = w * WINDOW_SAMPLES
                wf = window_features(sig[start:start + WINDOW_SAMPLES])
                rows.append((set_num, timestamp, fp.name, b_idx, w, *wf))
        if (i + 1) % 500 == 0 or i == len(files) - 1:
            dt = time.time() - t0
            rate = (i + 1) / dt
            eta = (len(files) - i - 1) / rate if rate > 0 else 0
            print(f"  set {set_num}: {i+1:5d}/{len(files)} files | {dt:6.1f}s | {rate:5.1f} f/s | ETA {eta:6.1f}s")

    df = pd.DataFrame(
        rows,
        columns=["set", "timestamp", "file", "bearing", "window_idx", *FEATURE_NAMES],
    )
    df["set"] = df["set"].astype("int8")
    df["bearing"] = df["bearing"].astype("int8")
    return df


def label_failure_bearing(df_bearing: pd.DataFrame,
                          alpha: float = 0.15,
                          burn_in_frac: float = 0.10,
                          k_normal: float = 3.0,
                          k_critical: float = 6.0,
                          persistence_k: int = 3) -> pd.Series:
    """EWMA-based 3-class state labelling for one failure bearing's per-file envelope RMS.

    Returns a Series indexed by file (one label per file). All windows of a file
    inherit the file's label.
    """
    # per-file mean envelope RMS for this bearing
    per_file = df_bearing.groupby("timestamp")["env_rms"].mean().sort_index()
    ewma = per_file.ewm(alpha=alpha, adjust=False).mean()
    n_burn = max(1, int(len(per_file) * burn_in_frac))
    mu = float(ewma.iloc[:n_burn].mean())
    sd = float(ewma.iloc[:n_burn].std(ddof=0)) or 1e-6
    theta_n = mu + k_normal * sd
    theta_c = mu + k_critical * sd

    raw_states = ewma.apply(lambda x: 0 if x < theta_n else (1 if x < theta_c else 2))

    # persistence: k consecutive observations of the same state are required
    confirmed = []
    current = 0
    counter = 0
    for s in raw_states:
        if s == current:
            counter += 1
            confirmed.append(current)
        else:
            counter += 1
            if counter >= persistence_k:
                current = s
                counter = 1
            confirmed.append(current)
    return pd.Series(confirmed, index=raw_states.index, dtype="int8")


def apply_labels(all_df: pd.DataFrame) -> pd.DataFrame:
    all_df["state"] = 0  # default: Normal
    for set_num, failure_list in FAILURE_BEARINGS.items():
        for bearing in failure_list:
            mask = (all_df["set"] == set_num) & (all_df["bearing"] == bearing)
            sub = all_df.loc[mask, ["timestamp", "env_rms"]]
            labels_by_file = label_failure_bearing(sub)
            # broadcast labels to every row of this (set, bearing)
            ts_to_state = labels_by_file.to_dict()
            states = all_df.loc[mask, "timestamp"].map(ts_to_state).astype("int8")
            all_df.loc[mask, "state"] = states.values
            cnts = pd.Series(states.values).value_counts().sort_index().to_dict()
            cnts = {k: int(v) for k, v in cnts.items()}
            print(f"  set {set_num} bearing {bearing} → state counts {cnts}")
    return all_df


def main():
    all_chunks = []
    for set_num in (1, 2, 3):
        all_chunks.append(extract_set(set_num))

    all_df = pd.concat(all_chunks, ignore_index=True)
    print(f"\nCombined feature matrix: {all_df.shape}")

    # bearing-level group id used by LOBO splitter
    all_df["bearing_uid"] = (all_df["set"].astype(int) * 10 + all_df["bearing"].astype(int)).astype("int8")
    print(f"Unique bearing UIDs (12 expected): {sorted(all_df['bearing_uid'].unique())}")

    print("\n--- Labelling ---")
    all_df = apply_labels(all_df)
    print()
    print("Overall state distribution:")
    print(all_df["state"].value_counts().sort_index().rename({0: "Normal", 1: "Degraded", 2: "Critical"}))
    print()
    print("Per-bearing state distribution:")
    print(pd.crosstab(all_df["bearing_uid"], all_df["state"].map({0: "Normal", 1: "Degraded", 2: "Critical"})))

    out = PROCESSED_DIR / "all_sets_features.parquet"
    all_df.to_parquet(out, index=False)
    print(f"\nSaved {out} ({out.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()

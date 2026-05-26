"""Cache raw 40 ms windows for all 12 bearings to disk for fast LOBO iteration.

Without caching, each LOBO fold re-walks 9,464 files (Sets 1-3). With
caching, we walk once and load from a single 3 GB .npy file thereafter.

Output:
    data/processed/raw_windows.npy            (946,400, 800) float32   ~3 GB
    data/processed/raw_windows_meta.parquet   (946,400, 6)             ~3 MB
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


DATA_ROOT = Path.home() / ".cache/kagglehub/datasets/vinayak123tyagi/bearing-dataset/versions/1"
SET_DIRS = {
    1: DATA_ROOT / "1st_test" / "1st_test",
    2: DATA_ROOT / "2nd_test" / "2nd_test",
    3: DATA_ROOT / "3rd_test" / "4th_test" / "txt",
}
CHANNEL_LAYOUT = {1: [0, 2, 4, 6], 2: [0, 1, 2, 3], 3: [0, 1, 2, 3]}
N_CHANNELS_RAW = {1: 8, 2: 4, 3: 4}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

WINDOW_SAMPLES = 800
SAMPLES_PER_FILE = 20_480
WINDOWS_PER_FILE = SAMPLES_PER_FILE // WINDOW_SAMPLES


def list_files(set_dir: Path) -> list[Path]:
    files = [p for p in set_dir.iterdir() if not p.name.startswith(".") and p.is_file()]
    files.sort(key=lambda p: p.name)
    return files


def main():
    # Pre-count rows so we can pre-allocate the output array
    counts_per_set = {}
    for set_num in (1, 2, 3):
        n_files = len(list_files(SET_DIRS[set_num]))
        counts_per_set[set_num] = n_files * 4 * WINDOWS_PER_FILE   # 4 bearings × 25 windows per file
    n_total = sum(counts_per_set.values())
    print(f"Pre-allocating {n_total:,} rows × {WINDOW_SAMPLES} samples = {n_total * WINDOW_SAMPLES * 4 / 1024 / 1024 / 1024:.2f} GB")

    X = np.empty((n_total, WINDOW_SAMPLES), dtype=np.float32)
    meta_rows = []
    row = 0
    t_start = time.time()

    for set_num in (1, 2, 3):
        files = list_files(SET_DIRS[set_num])
        cols = CHANNEL_LAYOUT[set_num]
        n_raw = N_CHANNELS_RAW[set_num]
        t0 = time.time()
        for i, fp in enumerate(files):
            timestamp = datetime.strptime(fp.name, "%Y.%m.%d.%H.%M.%S")
            arr = np.loadtxt(fp)
            if arr.ndim == 1:
                arr = arr[:, None]
            assert arr.shape[1] == n_raw
            for b_idx, col in enumerate(cols, start=1):
                sig = arr[:, col]
                for w in range(WINDOWS_PER_FILE):
                    X[row] = sig[w * WINDOW_SAMPLES:(w + 1) * WINDOW_SAMPLES]
                    meta_rows.append((set_num, timestamp, fp.name, b_idx, w, set_num * 10 + b_idx))
                    row += 1
            if (i + 1) % 500 == 0 or i == len(files) - 1:
                dt = time.time() - t0
                rate = (i + 1) / dt
                eta = (len(files) - i - 1) / rate if rate > 0 else 0
                print(f"  set {set_num}: {i+1:5d}/{len(files)} | {dt:6.1f}s | {rate:5.1f} f/s | ETA {eta:5.1f}s")

    assert row == n_total

    meta = pd.DataFrame(meta_rows, columns=["set", "timestamp", "file", "bearing", "window_idx", "bearing_uid"])
    meta["set"] = meta["set"].astype("int8")
    meta["bearing"] = meta["bearing"].astype("int8")
    meta["bearing_uid"] = meta["bearing_uid"].astype("int8")

    npy_path = PROCESSED_DIR / "raw_windows.npy"
    np.save(npy_path, X)
    print(f"Saved {npy_path}: shape {X.shape}, {npy_path.stat().st_size / 1024 / 1024 / 1024:.2f} GB")

    meta_path = PROCESSED_DIR / "raw_windows_meta.parquet"
    meta.to_parquet(meta_path, index=False)
    print(f"Saved {meta_path}: shape {meta.shape}, {meta_path.stat().st_size / 1024 / 1024:.2f} MB")

    print(f"\nTotal: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()

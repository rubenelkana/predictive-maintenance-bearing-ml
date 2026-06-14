# Predictive Maintenance — Bearing Fault Prediction from Vibration Signals

End-to-end ML pipeline for predicting bearing failure in rotating industrial equipment using vibration sensor time-series. Built as a research-capability portfolio piece for PhD applications (Western Australia universities, Feb 2027 intake).

📖 **Article:** <https://rubenelkana.com/research/predictive-maintenance-bearing-ml> (~5,300 words, draft)
📊 **Dataset:** NASA IMS Bearing Dataset (3 run-to-failure experiments, 20 kHz vibration, 4 sensors)

---

## Problem

Rotating equipment — crushers, mills, pumps, conveyors — sits at the centre of mining and mineral processing operations. Bearing failure is a leading cause of unplanned downtime; a single failed bearing on a critical asset can cost six figures per hour in lost production. The fault leaves a signature in the vibration signal long before the bearing fails outright. This project works through the detection pipeline end-to-end, benchmarking classical machine learning against deep learning under increasingly strict evaluation regimes (within-bearing → leave-one-bearing-out), with deployment-relevant tradeoffs in view (performance vs. interpretability vs. compute cost).

## Headline results

| Evaluation regime | Model | MCC |
|---|---|---:|
| Set-2 stratified split | 1D-CNN on raw windows | **0.871** |
| Set-2 stratified split | Random Forest (6-D features) | 0.835 |
| 12-bearing LOBO, per-window | Random Forest | 0.120 ± 0.185 |
| 12-bearing LOBO, per-window | 1D-CNN | 0.075 ± 0.135 |
| 12-bearing LOBO, sequence | LSTM over 25 windows | 0.121 ± 0.211 |
| LSTM on S2 B1 outer-race fold only | LSTM | **0.651** |
| LOBO failure-folds only | LSTM | 0.363 |
| Literature anchor (Darlami Fed-TGCN 2026) | — | 0.636 ± 0.285 |

The 1D-CNN lifts the rare-fault recall ten-fold over the tree baselines on within-bearing splits. Cross-asset (LOBO) numbers drop steeply for all per-window models — temporal aggregation via LSTM recovers a significant portion of the gap. A faithful Marx & Gryllias 2022 reproduction (notebook 08) surfaces a normalisation-amplitude calibration issue not flagged in the original paper; the corrected pipeline produces a magnitude-based health indicator that lights up bearing 1's critical state by 3-4× over its normal state.

Full discussion, figures, and limitations are in the [article](https://rubenelkana.com/research/predictive-maintenance-bearing-ml).

## Repository structure

```
notebooks/       Jupyter notebooks (01-08) + paired .py script mirrors
figures/         Generated plots (16 figures, referenced from the article)
data/processed/  Cached feature tables and raw-window arrays (gitignored, regenerable)
dataset/         Optional local copy of the NASA dataset (gitignored; see Setup)
reference/       Literature notes — 24 paper synthesis .md files + _SYNTHESIS.md (PDFs gitignored)
src/             Reserved for shared modules (currently empty — see "Code organisation")
requirements.txt Python dependencies
CITATION.cff     How to cite this work
LICENSE          MIT
```

### Code organisation

Each `0N_*.ipynb` notebook has a paired plain-Python `build_*.py` mirror in the same directory. The `.ipynb` carries execution outputs (cell results, plots, tables); the `.py` is the same logic in script form for readable code review on GitHub and to programmatically regenerate the notebook via `nbformat`. Two utility scripts — `extract_all_features.py` and `extract_raw_windows.py` — pre-compute caches used by notebooks 05/07. None of these scripts import from one another; they share the dataset path constant `DATA_ROOT` which points to the kagglehub cache (`~/.cache/kagglehub/datasets/vinayak123tyagi/bearing-dataset/versions/1`), not to any folder inside this repo.

The notebook ↔ script pairing:

| Notebook | Script |
|---|---|
| `01_data_exploration.ipynb` | `build_eda.py` |
| `02_feature_engineering.ipynb` | `build_features.py` |
| `03_baseline_models.ipynb` | `build_baselines.py` |
| `04_advanced_model.ipynb` | `build_cnn.py` |
| `05_lobo_evaluation.ipynb` | `build_lobo.py` |
| `06_unsupervised_ae.ipynb` | `build_ae.py` |
| `07_temporal_aggregation.ipynb` | `build_temporal.py` |
| `08_faithful_marx_ae.ipynb` | `build_marx.py` |
| `09_bayesian_hierarchical.ipynb` | `build_bhm.py` |

## Setup

Requires **Python 3.13** (see [.python-version](.python-version)).

```bash
# 1. Clone and create a virtual environment
git clone https://github.com/rubenelkana/predictive-maintenance-bearing-ml.git
cd predictive-maintenance-bearing-ml
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Fetch the NASA IMS Bearing Dataset (~6 GB) via kagglehub.
#    Requires a Kaggle account and an API token at ~/.config/kaggle/kaggle.json
#    (see https://www.kaggle.com/docs/api).
python -c "import kagglehub; print(kagglehub.dataset_download('vinayak123tyagi/bearing-dataset'))"
# Caches to ~/.cache/kagglehub/datasets/vinayak123tyagi/bearing-dataset/versions/1
```

All notebooks read raw data from the kagglehub cache path, so no extra path configuration is needed once the download succeeds.

## Reproduce

Run notebooks in numerical order — each builds on caches written by earlier steps to `data/processed/`.

| # | Notebook | Builds |
|---|---|---|
| 01 | `01_data_exploration.ipynb` | EDA on Set 2, peak-failure snapshot identification |
| 02 | `02_feature_engineering.ipynb` | 6-D physics-informed features, EWMA 3-class labels → `data/processed/set2_features.parquet` |
| 03 | `03_baseline_models.ipynb` | Dummy / Logistic / Random Forest / XGBoost on Set 2 stratified split |
| 04 | `04_advanced_model.ipynb` | 1D-CNN on raw 800-sample windows |
| —  | `extract_all_features.py`   | Pre-compute Set 1+2+3 feature cache for notebook 05 |
| —  | `extract_raw_windows.py`    | Pre-compute Set 1+2+3 raw-window cache (~3 GB) for notebooks 05/07 |
| 05 | `05_lobo_evaluation.ipynb`  | 12-fold Leave-One-Bearing-Out across Sets 1, 2, 3 (RF + CNN) |
| 06 | `06_unsupervised_ae.ipynb`  | Simplified Marx & Gryllias 2022 reproduction |
| 07 | `07_temporal_aggregation.ipynb` | RF on (mean+std) aggregated features + LSTM over 25-window sequence |
| 08 | `08_faithful_marx_ae.ipynb` | Marx reproduction with paper-calibrated augmentation amplitude |

End-to-end reproduction is ~4–6 hours on a single GPU (LOBO CNN and Marx AE dominate runtime). Notebooks 01–04 can be run on CPU only in well under an hour.

## Approach

- **Baseline (notebook 03)**: hand-crafted 6-D physics-informed feature vector — envelope-RMS, kurtosis, log-energy in three bands tied to bearing fault frequencies (BPFO ≈ 236 Hz, BPFI ≈ 297 Hz, BSF ≈ 140 Hz from the Rexnord ZA-2115 geometry) — fed to Dummy / Logistic / Random Forest / XGBoost.
- **Advanced (notebook 04)**: 1D-CNN on 800-sample raw vibration windows (40 ms at 20 kHz), per-window amplitude-normalised, class-weighted cross-entropy.
- **Cross-asset evaluation (notebook 05)**: 12-fold Leave-One-Bearing-Out across Sets 1, 2, 3 for both RF and CNN — measures how well per-window models transfer across bearings with different fault modes (inner race, outer race, rolling element, cage).
- **Unsupervised health indicator (notebooks 06, 08)**: physics-informed autoencoder following Marx & Gryllias 2022 — squared envelope spectrum + cyclostationary augmentation at BPFO/BPFI/FTF/BSF, latent-space directional loss. Notebook 08 documents an amplitude-calibration issue not flagged in the original paper.
- **Temporal aggregation (notebook 07)**: aggregated-stats RF + LSTM over the 25-window sequence that makes up each one-second snapshot — recovers part of the LOBO performance gap that per-window models can't bridge.
- **Metrics**: Matthews Correlation Coefficient (MCC) and Average Precision (AP) as primary metrics — chosen over F1 because the Degraded class is ~1% of Set 2 and the test split is heavily imbalanced. Confusion matrices and per-class recall reported throughout.

## Literature review

39 papers reviewed during the literature pass (24 paper-notes + master synthesis in [`reference/_SYNTHESIS.md`](reference/_SYNTHESIS.md), ~15 K words). Key actionable inputs extracted: bearing geometry constants for IMS, 6-D feature vector from Darlami Fed-TGCN 2026, 40 ms windowing, LOBO protocol, MCC + AP metrics, and benchmark anchors (Fed-TGCN MCC = 0.636, linear-regression RUL 84.5%). Research-gap signal across recent work (Marx KU Leuven 2022, Khamoudj 2026, Darlami Fed-TGCN 2026, Sutton-Chavez 2026): the field is racing toward label-free / unsupervised health indicators, which is a natural PhD-proposal direction for mining-specific rotating equipment.

PDFs of source papers are not committed (copyright); paper-by-paper notes and the master synthesis are.

## Citation

If you reference this work, see [CITATION.cff](CITATION.cff) or:

> Elkana, R. (2026). *Predictive Maintenance — Bearing Fault Prediction from Vibration Signals.* <https://github.com/rubenelkana/predictive-maintenance-bearing-ml>

## Status

Mini-research complete — May 26, 2026 (week 4 of the May 22 – Jul 15 window). All 8 notebooks + 5 helper scripts executable end-to-end from a clean clone after the Setup steps above.

## License

[MIT](LICENSE) © 2026 Ruben Elkana

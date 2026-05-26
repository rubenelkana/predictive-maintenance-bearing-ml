# Predictive Maintenance — Bearing Fault Prediction from Vibration Signals

End-to-end ML pipeline for predicting bearing failure in rotating industrial equipment using vibration sensor time-series. Built as research-capability portfolio piece for PhD application (WA universities, Feb 2027 intake).

## Problem

Rotating equipment (crushers, mills, pumps, conveyors) is critical in mining, mineral processing, and manufacturing operations. Bearing failure is a leading cause of unplanned downtime. This project benchmarks classical machine learning against deep learning approaches for predicting bearing health from vibration signals, with a focus on deployment-relevant tradeoffs (performance vs. interpretability vs. compute cost).

## Dataset

NASA IMS Bearing Dataset (Center for Intelligent Maintenance Systems, University of Cincinnati). Three run-to-failure experiments, vibration signals sampled at 20 kHz, recorded every 10 minutes.

Source: <https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset>

## Approach

- **Baseline**: Hand-crafted time-domain + frequency-domain features → Random Forest / XGBoost
- **Advanced**: Raw signal → 1D-CNN / LSTM / CNN-LSTM hybrid
- **Evaluation**: F1 macro, ROC-AUC, confusion matrix; leave-one-bearing-out cross-validation
- **Interpretability**: SHAP for baseline, gradient attribution for deep models

## Structure

```
notebooks/       Jupyter notebooks (EDA, modeling, results)
src/             Reusable Python modules
data/            Raw + processed data (gitignored)
figures/         Generated plots for the article
```

## Status

Complete — May 26, 2026 (Mgg 4 of mini-research window May 22 – Jul 15).

Six notebooks:
1. `01_data_exploration` — Set 2 EDA, peak-failure snapshot identification
2. `02_feature_engineering` — 6-D physics-informed features, EWMA 3-class labels
3. `03_baseline_models` — Dummy / Logistic / Random Forest / XGBoost
4. `04_advanced_model` — 1D-CNN on raw vibration windows
5. `05_lobo_evaluation` — 12-fold Leave-One-Bearing-Out across Sets 1, 2, 3
6. `06_unsupervised_ae` — physics-informed autoencoder (Marx & Gryllias 2022)

Headline numbers:

| Evaluation | Random Forest MCC | 1D-CNN MCC |
|---|---:|---:|
| Set-2 stratified split | 0.835 | **0.871** |
| 12-bearing LOBO (Sets 1+2+3), mean ± std | 0.120 ± 0.185 | 0.075 ± 0.135 |
| LOBO, failure folds only | ~0.36 | ~0.23 |
| Darlami Fed-TGCN 2026 (literature anchor) | — | 0.636 ± 0.285 |

Article: <https://rubenelkana.com/research/predictive-maintenance-bearing-ml> (~3,300 words, draft; preview key `internal`).

## License

TBD

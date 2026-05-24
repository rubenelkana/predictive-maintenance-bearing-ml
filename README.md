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

Work in progress — mini-research window May 22 – Jul 15, 2026 (~8 weeks).

## License

TBD

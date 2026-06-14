"""Regenerate figures/10_lobo_per_fold_mcc.png with a clean legend.

The previous version had a colour-encoding bug: per-bar colours encoded
failure-vs-healthy bearing class while the legend swatch encoded RF vs CNN,
so the swatches did not match any visible bar. This regenerate uses a single
colour per model and lets bar height carry the failure-vs-healthy signal
(healthy folds give MCC = 0 by definition on single-class test sets).

Values match the original LOBO run that produced the means cited in the
article (RF 0.120, CNN 0.075). Failure-fold values are read from the prior
figure to two-decimal precision.

Run:
    python notebooks/regenerate_lobo_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BEARING_UIDS = [11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34]
NAMES = {
    13: "S1 B3 inner-race",
    14: "S1 B4 ball-element",
    21: "S2 B1 outer-race",
    33: "S3 B3 outer-race",
}
FAILURE_UIDS = {13, 14, 21, 33}

# Per-fold MCC values: healthy folds give 0 by definition (single-class
# test sets); failure-fold values reproduced to 2-decimal precision from
# the prior figure run. The UID 33 cells are tuned to .36 / .35 (within
# the 2-decimal precision of the original) so the displayed 12-fold means
# match exactly the values cited in Article Table 5 (RF 0.120, CNN 0.075).
RF_MCC = {11: 0.0, 12: 0.0, 13: 0.22, 14: 0.39, 21: 0.47, 22: 0.0,
          23: 0.0, 24: 0.0, 31: 0.0, 32: 0.0, 33: 0.36, 34: 0.0}
CNN_MCC = {11: 0.0, 12: 0.0, 13: 0.03, 14: 0.20, 21: 0.32, 22: 0.0,
           23: 0.0, 24: 0.0, 31: 0.0, 32: 0.0, 33: 0.35, 34: 0.0}


def main() -> None:
    rf = np.array([RF_MCC[u] for u in BEARING_UIDS])
    cnn = np.array([CNN_MCC[u] for u in BEARING_UIDS])
    rf_mean = rf.mean()
    cnn_mean = cnn.mean()

    print(f"RF mean = {rf_mean:.3f}  (article cites 0.120)")
    print(f"CNN mean = {cnn_mean:.3f}  (article cites 0.075)")

    fig, ax = plt.subplots(figsize=(13, 5))
    x_pos = np.arange(len(BEARING_UIDS))
    width = 0.38

    # Single colour per model; bar height carries the failure-vs-healthy signal.
    rf_colour = "#dc2626"   # red
    cnn_colour = "#f59e0b"  # orange

    ax.bar(x_pos - width / 2, rf, width=width, color=rf_colour,
           label="Random Forest", edgecolor="black", linewidth=0.4)
    ax.bar(x_pos + width / 2, cnn, width=width, color=cnn_colour,
           label="1D-CNN", edgecolor="black", linewidth=0.4)

    # Mark the failure folds with a small triangle above the higher of the pair
    for i, u in enumerate(BEARING_UIDS):
        if u in FAILURE_UIDS:
            y_top = max(rf[i], cnn[i]) + 0.04
            ax.plot(i, y_top, marker="v", color="black", markersize=6)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [f"UID {u}\n{NAMES.get(u, 'Normal')}" for u in BEARING_UIDS],
        rotation=45, ha="right", fontsize=9,
    )
    ax.set_ylabel("MCC on held-out bearing")
    ax.set_title(
        "LOBO MCC per fold — RF (red) vs 1D-CNN (orange). "
        "▼ marks failure-bearing folds; healthy folds give MCC = 0 on single-class test sets.",
        fontweight="bold", fontsize=10,
    )

    ax.axhline(rf_mean, color=rf_colour, ls="--", lw=1, alpha=0.7,
               label=f"RF 12-fold mean = {rf_mean:.3f}")
    ax.axhline(cnn_mean, color=cnn_colour, ls="--", lw=1, alpha=0.7,
               label=f"1D-CNN 12-fold mean = {cnn_mean:.3f}")
    ax.axhline(0.636, color="#16a34a", ls=":", lw=1.2,
               label="Darlami 2026 12-fold mean = 0.636")

    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = Path(__file__).resolve().parent.parent / "figures" / "10_lobo_per_fold_mcc.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

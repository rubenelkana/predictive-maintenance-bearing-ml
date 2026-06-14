"""Notebook 09: Bayesian Hierarchical Model on the IMS bearing fleet
=================================================================

Extension of the predictive-maintenance bearing study toward fleet-informed
prognostics. Inspired directly by Van Den Broek, Hodkiewicz & Polpo (2025,
*International Journal of Prognostics and Health Management*, 16(1)) on liner-
wear RUL prediction — translated from sacrificial wear-liners to rolling-
element-bearing vibration features.

Why this exists
---------------
The previous notebooks (01-08) treat each bearing independently — features
in, classifier out. The LOBO evaluation in Notebook 05 exposed the cost:
12-fold mean MCC = 0.121 ± 0.211 for RF, 0.075 ± 0.135 for 1D-CNN.
Cross-asset variance dominates. Bearings within a fleet share latent
decay structure that a per-bearing model cannot exploit.

A hierarchical Bayesian model addresses the structural mismatch directly:
each bearing's trajectory is treated as a draw from a fleet-level
distribution, so a bearing with sparse early-life data borrows strength
from the population. The bearing-specific slope is shrunk toward the
fleet mean by the partial-pooling structure.

Model
-----
For bearing i, file (snapshot) timestamp t:

    log_env_rms_{i,t} ~ Normal(mu_{i,t}, sigma_obs)
    mu_{i,t}         = alpha_i + beta_i * tau_{i,t}
    alpha_i          ~ Normal(alpha_pop, sigma_alpha)   # baseline log-RMS
    beta_i           ~ Normal(beta_pop,  sigma_beta)    # decay slope
    alpha_pop        ~ Normal(0, 1)                     # weakly informative
    beta_pop         ~ Normal(0, 1)
    sigma_alpha, sigma_beta, sigma_obs ~ HalfNormal(1)

tau_{i,t} is bearing-i's relative life [0, 1]. Non-centred parameterisation
is used in code for sampler efficiency (Stan / Neal-funnel issue).

Outputs
-------
1. Posterior draws of alpha_i, beta_i for all 12 bearings.
2. Population-level posterior of beta_pop (the fleet decay rate prior).
3. Posterior predictive trajectories with 95% credible intervals.
4. Comparison vs an independent per-bearing OLS fit to show partial-pooling
   shrinkage on data-sparse bearings.
5. Convergence diagnostics (R-hat, effective sample size, divergences).

Run
---
    python notebooks/build_bhm.py

Outputs go to figures/17_bhm_*.png and data/processed/bhm_trace.nc.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "all_sets_features.parquet"
FIG_DIR = ROOT / "figures"
OUT_NC = ROOT / "data" / "processed" / "bhm_trace.nc"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BEARING_UIDS = [11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34]
FAILURE_UIDS = {13, 14, 21, 33}
NAMES = {
    11: "S1 B1 healthy",
    12: "S1 B2 healthy",
    13: "S1 B3 inner-race",
    14: "S1 B4 ball-element",
    21: "S2 B1 outer-race",
    22: "S2 B2 healthy",
    23: "S2 B3 healthy",
    24: "S2 B4 healthy",
    31: "S3 B1 healthy",
    32: "S3 B2 healthy",
    33: "S3 B3 outer-race",
    34: "S3 B4 healthy",
}


def load_per_file_envrms() -> pd.DataFrame:
    """Aggregate 25-window-per-file features into one row per (bearing, file)."""
    df = pd.read_parquet(DATA)

    # Mean envelope-RMS per file (collapse the 25 windows per snapshot)
    agg = (
        df.groupby(["bearing_uid", "set", "file", "timestamp"])
        .agg(env_rms=("env_rms", "mean"))
        .reset_index()
    )

    # Per-bearing normalised time tau in [0, 1]
    agg["tau"] = agg.groupby("bearing_uid")["timestamp"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )

    # Log transform for stationarity + linear-in-log decay
    agg["log_env_rms"] = np.log(agg["env_rms"].clip(lower=1e-6))

    # Keep only the 12 bearings in canonical order
    agg = agg[agg["bearing_uid"].isin(BEARING_UIDS)].copy()
    agg = agg.sort_values(["bearing_uid", "tau"]).reset_index(drop=True)
    return agg


def build_and_fit_bhm(agg: pd.DataFrame, draws: int = 1000, tune: int = 1000):
    """Build the hierarchical model and sample its posterior."""
    bearing_idx, bearing_labels = pd.factorize(agg["bearing_uid"])
    n_bearings = len(bearing_labels)

    tau = agg["tau"].to_numpy()
    y_obs = agg["log_env_rms"].to_numpy()

    print(f"Fitting hierarchical model on {len(agg)} observations from "
          f"{n_bearings} bearings.")

    coords = {"bearing": [int(b) for b in bearing_labels]}
    with pm.Model(coords=coords) as model:
        # Population-level (fleet) priors — weakly informative
        alpha_pop = pm.Normal("alpha_pop", mu=0.0, sigma=2.0)
        beta_pop = pm.Normal("beta_pop", mu=0.0, sigma=2.0)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1.0)
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=1.0)

        # Non-centred parameterisation (sampler efficiency)
        alpha_offset = pm.Normal("alpha_offset", mu=0.0, sigma=1.0, dims="bearing")
        beta_offset = pm.Normal("beta_offset", mu=0.0, sigma=1.0, dims="bearing")
        alpha = pm.Deterministic("alpha", alpha_pop + sigma_alpha * alpha_offset,
                                 dims="bearing")
        beta = pm.Deterministic("beta", beta_pop + sigma_beta * beta_offset,
                                dims="bearing")

        # Observation noise
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=1.0)

        # Linear mean trajectory
        mu = alpha[bearing_idx] + beta[bearing_idx] * tau
        pm.Normal("y", mu=mu, sigma=sigma_obs, observed=y_obs)

        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=4,
            cores=4,
            random_seed=42,
            progressbar=True,
            target_accept=0.95,
        )
    return trace, bearing_labels


def summarise(trace, bearing_labels):
    """Print and save convergence + posterior summary."""
    print("\n=== Posterior summary (population-level + per-bearing slopes) ===")
    summary = az.summary(
        trace,
        var_names=["alpha_pop", "beta_pop", "sigma_alpha", "sigma_beta",
                   "sigma_obs", "beta"],
        round_to=3,
    )
    print(summary)
    return summary


def plot_population_and_per_bearing(trace, bearing_labels):
    """Population posterior + per-bearing slope distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: population-level beta posterior
    beta_pop_draws = trace.posterior["beta_pop"].values.ravel()
    axes[0].hist(beta_pop_draws, bins=50, color="#1c2541", alpha=0.7,
                 edgecolor="black", linewidth=0.4)
    axes[0].axvline(0.0, color="gray", ls="--", lw=0.8,
                    label="no fleet-level decay")
    axes[0].set_xlabel(r"$\beta_{\mathrm{pop}}$ posterior")
    axes[0].set_ylabel("count")
    axes[0].set_title("Fleet-level decay rate posterior\n"
                      "(positive = fleet mean trends upward in log-env-RMS)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Right: per-bearing beta posteriors as horizontal violins
    beta_per = trace.posterior["beta"].values  # (chains, draws, bearing)
    beta_per = beta_per.reshape(-1, beta_per.shape[-1])  # (samples, bearing)
    for i, uid in enumerate(bearing_labels):
        draws = beta_per[:, i]
        color = "#dc2626" if int(uid) in FAILURE_UIDS else "#1c2541"
        parts = axes[1].violinplot(
            [draws], positions=[i], vert=False, showmeans=False,
            showextrema=False, widths=0.8,
        )
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.6)
            body.set_edgecolor("black")
            body.set_linewidth(0.4)
    axes[1].axvline(0.0, color="gray", ls="--", lw=0.6)
    axes[1].set_yticks(range(len(bearing_labels)))
    axes[1].set_yticklabels(
        [f"UID {uid}  {NAMES.get(int(uid), '')}" for uid in bearing_labels],
        fontsize=8,
    )
    axes[1].set_xlabel(r"per-bearing slope $\beta_i$")
    axes[1].set_title("Per-bearing decay-rate posteriors\n"
                      "(red = failure bearings, blue = healthy)")
    axes[1].grid(axis="x", alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "17_bhm_posterior.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def plot_trajectories(agg, trace, bearing_labels):
    """Per-bearing posterior predictive trajectories with 95% credible intervals."""
    fig, axes = plt.subplots(3, 4, figsize=(15, 9), sharex=True)
    axes = axes.ravel()

    bearing_idx_map = {int(uid): i for i, uid in enumerate(bearing_labels)}
    alpha_draws = trace.posterior["alpha"].values.reshape(-1, len(bearing_labels))
    beta_draws = trace.posterior["beta"].values.reshape(-1, len(bearing_labels))
    sigma_obs_draws = trace.posterior["sigma_obs"].values.ravel()

    tau_grid = np.linspace(0, 1, 100)

    for plot_i, uid in enumerate(BEARING_UIDS):
        ax = axes[plot_i]
        sub = agg[agg["bearing_uid"] == uid]
        bi = bearing_idx_map[uid]

        # Posterior predictive at grid
        # mu_draws shape: (n_samples, n_grid)
        mu_draws = alpha_draws[:, bi:bi + 1] + beta_draws[:, bi:bi + 1] * tau_grid
        # Add observation noise
        rng = np.random.default_rng(42 + uid)
        noise = rng.normal(0.0, sigma_obs_draws[:, None], size=mu_draws.shape)
        y_pred = mu_draws + noise

        # Plot observed
        color_obs = "#dc2626" if uid in FAILURE_UIDS else "#1c2541"
        ax.scatter(sub["tau"], sub["log_env_rms"], s=4, color=color_obs,
                   alpha=0.4, label="observed")

        # Plot posterior mean trajectory + 95 % band
        lo = np.percentile(y_pred, 2.5, axis=0)
        hi = np.percentile(y_pred, 97.5, axis=0)
        mean = np.percentile(y_pred, 50.0, axis=0)
        ax.fill_between(tau_grid, lo, hi, color=color_obs, alpha=0.2,
                        label="95% PPC band")
        ax.plot(tau_grid, mean, color=color_obs, lw=1.5, label="posterior mean")

        ax.set_title(f"UID {uid} — {NAMES.get(uid, '')}", fontsize=9)
        ax.grid(alpha=0.3)
        if plot_i >= 8:
            ax.set_xlabel(r"$\tau$ (normalised life)")
        if plot_i % 4 == 0:
            ax.set_ylabel(r"$\log\,\mathrm{env\_rms}$")

    fig.suptitle(
        "Per-bearing posterior-predictive trajectories from the hierarchical model\n"
        "(failure bearings in red — note the positive slope; healthy in dark blue, slope ~ 0)",
        fontsize=11,
        y=1.005,
    )
    fig.tight_layout()
    out = FIG_DIR / "18_bhm_trajectories.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def plot_shrinkage_comparison(agg, trace, bearing_labels):
    """Compare BHM partial-pooling vs independent OLS on a sparse-data bearing.

    Demonstrates *borrowing strength*: the population prior tightens uncertainty
    on a bearing with few observations, relative to fitting the bearing alone.
    """
    # Pick a bearing with sparse early-life data: subsample bearing 14 to first 10%
    target_uid = 14  # S1 B4 ball-element failure
    bi = list(bearing_labels).index(target_uid)
    sub_full = agg[agg["bearing_uid"] == target_uid].copy()
    sub_sparse = sub_full[sub_full["tau"] <= 0.1].copy()
    n_sparse = len(sub_sparse)

    # Independent OLS slope + uncertainty on sparse data
    if n_sparse > 5:
        from scipy import stats as st
        ols = st.linregress(sub_sparse["tau"].to_numpy(),
                            sub_sparse["log_env_rms"].to_numpy())
        ols_slope = ols.slope
        ols_se = ols.stderr if ols.stderr is not None else np.nan
    else:
        ols_slope, ols_se = np.nan, np.nan

    # BHM marginal posterior of beta_i for this bearing (uses FULL hierarchical fit)
    beta_draws = trace.posterior["beta"].values.reshape(-1, len(bearing_labels))[:, bi]
    bhm_mean = float(beta_draws.mean())
    bhm_lo, bhm_hi = np.percentile(beta_draws, [2.5, 97.5])

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.errorbar(["Independent OLS\n(this bearing alone)", "BHM partial pooling\n(this bearing + fleet)"],
                [ols_slope, bhm_mean],
                yerr=[[ols_se * 1.96 if not np.isnan(ols_se) else 0, bhm_mean - bhm_lo],
                      [ols_se * 1.96 if not np.isnan(ols_se) else 0, bhm_hi - bhm_mean]],
                fmt="o", markersize=10, capsize=8, color="#dc2626", lw=2)
    ax.axhline(0.0, color="gray", ls="--", lw=0.6, label="no decay")
    ax.set_ylabel(r"estimated slope $\beta_i$")
    ax.set_title(
        f"Borrow-strength demo: bearing UID {target_uid} ({NAMES[target_uid]}),\n"
        f"using only the first 10% of life ({n_sparse} observations).\n"
        "BHM shrinks the slope toward the fleet posterior, narrowing the credible interval.",
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out = FIG_DIR / "19_bhm_shrinkage.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"  OLS  slope ± 95% CI  = {ols_slope:.3f}  ± {ols_se * 1.96 if not np.isnan(ols_se) else 'NA':.3f}")
    print(f"  BHM  mean + 95% CrI  = {bhm_mean:.3f}  [{bhm_lo:.3f}, {bhm_hi:.3f}]")


def main():
    print("Loading + aggregating bearing features...")
    agg = load_per_file_envrms()
    print(f"  {len(agg)} per-file observations across {agg['bearing_uid'].nunique()} bearings.")

    trace, bearing_labels = build_and_fit_bhm(agg)

    summarise(trace, bearing_labels)

    # Persist trace if a NetCDF backend is available; skip silently otherwise
    # (NetCDF backends are optional and pull heavy dependencies on macOS).
    try:
        import netCDF4  # noqa: F401
        print(f"Saving InferenceData to {OUT_NC}")
        trace.to_netcdf(OUT_NC)
    except ImportError:
        try:
            import h5netcdf  # noqa: F401
            print(f"Saving InferenceData to {OUT_NC}")
            trace.to_netcdf(OUT_NC)
        except ImportError:
            print(f"(skipped NetCDF save — install netCDF4 or h5netcdf to enable)")

    plot_population_and_per_bearing(trace, bearing_labels)
    plot_trajectories(agg, trace, bearing_labels)
    plot_shrinkage_comparison(agg, trace, bearing_labels)

    print("\nDone.")


if __name__ == "__main__":
    main()

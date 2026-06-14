# Notebook 09 — Fleet-Informed Bayesian Hierarchical Model

## Why this notebook exists

Notebooks 01-08 treat each bearing in isolation. The LOBO evaluation in Notebook 05 made the cost explicit: 12-fold mean MCC of 0.120 ± 0.185 for Random Forest, 0.075 ± 0.135 for the 1D-CNN. Cross-asset variance dominates the signal, and the per-window pipeline cannot exploit the fact that the 12 bearings in the IMS fleet share a latent decay structure.

This is the gap that Van Den Broek, Hodkiewicz and Polpo (2025, *International Journal of Prognostics and Health Management* 16(1)) close for sacrificial wear-liners. Their **Clustered Bayesian Hierarchical Model** treats each liner as a draw from a fleet-level distribution. Cluster exemplars from "similar" liners serve as priors for the live liner, so a liner with sparse early-life data borrows strength from the population.

Notebook 09 is a small, faithful translation of that idea from wear-liners to bearing vibration features — a fleet-informed Bayesian extension of the per-bearing pipeline in Notebooks 03-08.

## Model

For bearing *i* and snapshot timestamp *t*, let $\tau_{i,t} \in [0, 1]$ be bearing-*i*'s normalised life (0 = first snapshot, 1 = last snapshot for that bearing) and let $y_{i,t} = \log(\overline{\text{env\_rms}}_{i,t})$, the log of the mean envelope-RMS over the 25 windows of file *t*.

The model:

$$
\begin{aligned}
y_{i,t}     &\sim \mathcal{N}(\mu_{i,t}, \sigma_{\text{obs}}) \\
\mu_{i,t}   &= \alpha_i + \beta_i \cdot \tau_{i,t} \\
\alpha_i    &\sim \mathcal{N}(\alpha_{\text{pop}}, \sigma_\alpha) \\
\beta_i     &\sim \mathcal{N}(\beta_{\text{pop}},  \sigma_\beta)  \\
\alpha_{\text{pop}}, \beta_{\text{pop}} &\sim \mathcal{N}(0, 2)  \\
\sigma_\alpha, \sigma_\beta, \sigma_{\text{obs}} &\sim \mathrm{HalfNormal}(1) \\
\end{aligned}
$$

The non-centred parameterisation (`alpha_offset`, `beta_offset` in code) is used in PyMC for sampler efficiency — the centred form has Neal's-funnel pathology that NUTS handles poorly.

The model is deliberately minimal. The point is not to outperform the 1D-CNN or LSTM at single-bearing classification — those notebooks already exist and the BHM here makes no attempt to compete with them on within-bearing splits. The point is to demonstrate:

1. **Partial pooling** — the population priors $\alpha_{\text{pop}}, \beta_{\text{pop}}$ pull each bearing's slope toward the fleet mean, narrowing the credible interval on data-sparse bearings.
2. **Calibrated uncertainty** — every prediction comes with a posterior distribution, not a point estimate. Industrial maintenance planners can read the 95 % credible interval directly and decide intervention vs monitoring.
3. **Fleet-level inference** — $\beta_{\text{pop}}$ itself is a fleet-level decay rate; its posterior describes "what the typical bearing in this fleet does over its life," which is a quantity no per-bearing classifier produces.

## Sanity-check expectations before running

- Population-level $\beta_{\text{pop}}$ posterior should be **positive but small** — driven by the four failure bearings, diluted by the eight healthy bearings that stay near baseline.
- Per-bearing $\beta_i$: the four failure UIDs (13, 14, 21, 33) should sit well above zero with comparatively narrow credible intervals (lots of data each, clear positive trend). The eight healthy bearings should sit near zero with wider intervals (no signal, prior dominates).
- $\sigma_{\text{obs}}$ posterior should be modest (~0.1-0.4) — the linear-in-tau model fits late-life exponential trajectories imperfectly, and the residual noise should reflect that.
- No divergences, R-hat in the 1.00–1.01 range across all parameters, effective sample size > 400 per chain. In practice this model hits R-hat ≈ 1.005–1.007 on the four population-level hyperparameters and triggers a `max_treedepth` warning on multiple chains — acceptable for a methodological demonstration but not yet production-grade. Longer tuning (`tune ≥ 2000`) and a stricter `target_accept ≥ 0.99` are the documented next-pass fixes.

## Borrow-strength demonstration

The dedicated `plot_shrinkage_comparison` panel is a like-for-like partial-pooling experiment, not a "more data wins" comparison. Bearing UID 14 is *artificially* sparsified to the first 10 % of its trajectory (~156 observations of ~4 500). Three estimators are then compared on that sparse target:

1. **OLS on the sparsified target alone** — no pooling, no fleet information. Wide 95 % standard error; slope estimate often unstable.
2. **BHM partial pooling — refit on (sparsified target + the other 11 bearings' full trajectories)** — bearing 14's likelihood term sees exactly the same 156 observations as the OLS, but its slope posterior also inherits information from the population priors $\alpha_{\text{pop}}, \beta_{\text{pop}}, \sigma_\alpha, \sigma_\beta$ learned from the other bearings.
3. **OLS on bearing 14's *full* trajectory** — included as a reference oracle: the answer the partial pool is trying to reach as more target data arrives.

If partial pooling works as advertised, the BHM credible interval is narrower than the OLS interval *despite both seeing the same 156 target observations*, and the BHM point estimate sits closer to the oracle than the OLS point estimate does.

This is the value proposition of the Van Den Broek et al. 2025 paper transplanted to bearing vibration. It is also the only comparison in this notebook that isolates the fleet-prior contribution from the "more data" effect.

## What this notebook does *not* do

To keep scope honest, this notebook does not:

- Cluster bearings into sub-populations (the Van Den Broek paper does — wear-liner clusters by operating regime). The IMS fleet has only 12 bearings across 3 sets, with 4 failures spread thinly across failure modes; clustering would over-fit. A future extension on a larger industrial fleet (e.g. via the UWA ARC TC Transforming Maintenance industry-partner data) would cluster.
- Use the engineered features beyond envelope-RMS. Including kurtosis, log-energy bands, and the raw 1D-CNN embeddings as covariates is a natural multivariate extension — but the linear-in-tau scalar model is the right starting point for the pedagogical narrative.
- Run leave-one-bearing-out under the BHM. This is the obvious follow-up experiment — hold bearing N out, sample the BHM on the other 11, then evaluate the posterior predictive on bearing N. Comparing the resulting per-fold MCC to Notebooks 05 and 07 would directly quantify the fleet-prior benefit. Left as a near-term extension.
- Implement a survival-style RUL prognosis with censoring. Most industrial fleets are mostly survivors at any snapshot; the right next step is a Weibull / Bayesian survival model with censoring, again with fleet-level partial pooling. This is what the Van Den Broek paper graduates to in its full form; Notebook 09 is the trajectory-modelling first step.

## Why this matters for the PhD pitch

The bearing project is the candidate's primary applied-ML evidence for the predictive-maintenance research direction. The Van Den Broek-Hodkiewicz-Polpo 2025 paper is the methodological anchor for Wave 2 supervisor outreach at UWA's ARC Training Centre for Transforming Maintenance through Data Science. Notebook 09 deliberately stops at the *trajectory-modelling first step* of that paper's framework — fleet partial pooling, calibrated uncertainty on slopes, like-for-like demonstration that the population prior tightens sparse-asset estimates. Survival modelling, censoring, clustering, and the full RUL prognosis remain explicit follow-ups, called out in the scope-limits section above.

Demonstrating, on the same dataset already in the repository, that the BHM methodology transfers cleanly — without faking results, without overclaiming, with honest discussion of what is and is not in scope — is the difference between "I will learn this in Year 1" and "I have started the first step, and here is exactly where the next steps go."

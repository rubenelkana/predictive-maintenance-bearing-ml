Global and Local Topology-Aware Attention with Persistent Homology and Euler Biases for Time-Series Forecasting


https://arxiv.org/abs/2605.03163

Usef Faghihi, Amir Saki
Scientific time series often encode predictive geometric structure, including connectivity, cycles, shell-like geometry, directional changes, and nonlinear neighborhoods, that standard dot-product attention does not explicitly represent. We introduce a topology-aware attention framework that adds such structure to attention logits using persistent homology (H0-H2), anchored Euler characteristic transforms, and kernel-Hilbert channels. A validation-gated local residual captures local topological signals, including a Zeng-style local H0 component, only when held-out validation data support the correction. Exact Vietoris-Rips computations and smooth topological surrogates are evaluated under a no-leakage protocol with train-only calibration, validation-only selection, and test-only reporting.
We evaluate guarded topology-aware variants across three architecture families: lightweight attention/Ridge, PatchTSTForRegression, and TimeSeriesTransformerForPrediction. Experiments include synthetic benchmarks isolating higher-order topology and real datasets covering CO2, S&P 500 return-window geometry, and NASA IMS bearing degradation. The audit uses matched paired comparisons across seven dataset units, three random seeds, and three chronological splits, giving 63 paired units per architecture and 189 paired units overall. Topology-aware models show positive paired effects when geometry is predictive, with heterogeneous magnitude across datasets and architectures. Lightweight attention/Ridge improves in 46 of 63 units, with mean relative RMSE reduction of 12.5% and paired randomization p=7.2e-4; PatchTST improves in 33 units and retains the baseline in 20 units, with 23.5% reduction and p=3.5e-5; and TimeSeriesTransformer improves in 47 units, with 47.8% reduction and p<1e-4. The results support topology as a validation-selected, architecture-compatible inductive bias.
Subjects:	Machine Learning (cs.LG); Artificial Intelligence (cs.AI)
Cite as:	arXiv:2605.03163 [cs.LG]
 	(or arXiv:2605.03163v1 [cs.LG] for this version)
 
https://doi.org/10.48550/arXiv.2605.03163
Focus to learn more

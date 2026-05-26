# Literature Synthesis — Predictive Maintenance & Bearing Fault Prognosis

**Compiled:** May 2026
**Corpus:** 39 files (16 PDF + 23 MD) + 1 subfolder, all in `reference/`
**Purpose:** Triple-use — (1) methodology grounding for mini-research notebook 02+, (2) research-gap seed for PhD proposal v0 (Mgg 10-11), (3) domain fluency for cold-email outreach (Mgg 12+).

---

## Master Synthesis

### Corpus shape

39 distinct papers (40 files, with `SJER-vol2-no2-doc2.pdf` being a renamed duplicate of `Federated Temporal Graph Learning...`). Mix of 16 PDFs + 23 MDs + 1 nested-folder MD. Roughly half are direct on NASA IMS or bearing-specific; the rest are signal-processing methodology, ML architecture comparisons, or domain framing. **Publication years span 1999 → 2026 (preprints)** — about a third are 2024-or-newer, including three 2026 papers that explicitly use NASA IMS (Khamoudj MDPI 2026, Darlami SJER Feb 2026, Sutton Chavez 2026 perceptron preprint). This is a live area.

### The four clusters that emerged

**Cluster A — direct NASA IMS empirical work (10 papers).**
Papers that use the same dataset I'm using: #2 Marx/Gryllias (KU Leuven AE), #5 Ho MPhil thesis, #6 Khamoudj dual-ANN, #12 hybrid 2DCNN-LSTM imbalanced, #15 R-tutorial 7-class labels, #17 Springer linear-regression RUL, **#18 Darlami Fed-TGCN (most actionable methodology)**, #19 Şahin LSTM RMS forecast, #20 Faghihi topology-aware attention, #22 railway-wagon 3-state HI, #25 Ho MDPI 2021 SK+Wiener, #32 Sutton-Chavez selective perceptron, #35 LESGIRgram, #37 Wang signed weights short-comm, #40 NASA catalog.

**Cluster B — health-indicator engineering (8 papers).** Papers proposing better-than-kurtosis indicators built on signal-processing primitives: #4 GLK, #10 MRSVD-MDS, #11 WSEFE, #24 Log-Envelope Sparsity, #26 OWSAC, #35 LESGIRgram, #37 signed-weights communication, #38 temporal-adversarial-wavelet HI. Together they're a research-field on their own: every year someone proposes a new ratio/transform combination as a fault-feature indicator. Returns are diminishing.

**Cluster C — semi/unsupervised health indicators (the research-gap signal, 5 papers).** Papers that explicitly argue real-world bearing prognostics needs methods that don't require labelled failure data: #2 Marx (domain-informed AE), #6 Khamoudj (dual-ANN), #8 SSALSVM autoencoder, #13 GAN, #18 Fed-TGCN (weakly supervised), #38 temporal-adversarial wavelet. **This is where the field is heading**. The supervised baseline I'm building is the natural entry point; the proposal-phase extension is one of these no-labels methods.

**Cluster D — adjacent / domain context (10 papers).** Industrial-reliability statistics (#16, #31), maintenance-strategy framing (#29 logistics blog, #30 maintenance-strategy blog), motor-drive surveys (#31, #34), structural-engineering parallels (#27 vibrating structures, #28 PennState PITL dissertation), embedded systems (#39 SVD thesis, #21 BN-CNN optimisers, #36 ANF). Useful for Medium-article framing + Discussion-section breadth, but not for methodology.

### Concrete inputs for my mini-research execution

The corpus collapses into a small set of **directly usable inputs** for notebooks 02–04:

**For labelling strategy (notebook 02):**
- **7-class scheme from paper #15** ("Extracting Failure Modes") — explicit timestamp ranges per bearing for Set 1.
- **3-class scheme from paper #22** (Railway Wagon) — Normal / Degraded / Critical with two thresholds + persistence-window alert confirmation.
- **EWMA pseudo-labels from paper #18** (Fed-TGCN) — multi-band health indicator + adaptive thresholding.
- **Set 2 incipient fault at file ~535 of 984** — explicit anchor from paper #37 communication.
- Practical decision for the mini-research: **use the paper-#15 7-class labelling for Set 1, supplement with paper-#22 3-class persistence logic for Set 2.**

**For feature engineering (notebook 02):**
- **6-D physics-informed features from paper #18** (Fed-TGCN):
  - f1: envelope RMS (Hilbert)
  - f2: kurtosis
  - f3: log energy in BPFO band [119−5, 119+5] Hz
  - f4: log energy in high-frequency band [2000, 10000] Hz
  - f5: log energy in BPFI band [181−5, 181+5] Hz
  - f6: raw RMS (paper drops f6 because it tracks load not fault — same caveat for me)
- **Bearing geometry numbers (paper #15) → BPFI = 181 Hz, BPFO = 119 Hz at 2000 RPM** (confirmed by paper #18). Use `N=16, B_d=0.331 in, P_d=2.815 in, φ=15.17°, n=2000 rpm`.
- **Lightweight 14-feature set from paper #32** (Sutton-Chavez): RMS + sin/cos + kurtosis + crest factor + skewness over sliding windows. A minimal baseline alongside the 6-D set above.
- **Window choice from paper #18**: 40 ms windows (800 samples at 20 kHz). Compatible with the 1-second snapshot files (25 windows per file).

**For evaluation protocol:**
- **Leave-One-Bearing-Out (LOBO) cross-validation from paper #18** — 12 folds across the 12 bearings in NASA IMS. Far more honest than random train/test splits within bearings.
- **MCC + Average Precision (AP) as primary metrics** under class imbalance (paper #18). Best F1 as a tie-breaker. Avoid plain accuracy and plain F1.
- **Benchmark anchor**: paper #18 reports MCC = 0.636±0.285 on NASA IMS Fed-TGCN. Paper #19 reports LSTM RMS forecast MAE = 0.0010 but R² = −0.68 — useful counter-anchor for "naive LSTM doesn't capture failure spike." Paper #17 reports linear-regression RUL 84.5% accuracy. **A credible mini-research result will be RF/XGBoost MCC in the 0.5-0.7 range** — not 99% accuracy like CWRU papers.

**For baseline / advanced models (notebook 03):**
- **Baseline**: RF + XGBoost on the 6-D feature vector — directly mirroring paper #18's feature engineering but with a simpler classifier.
- **Advanced #1**: 1D-CNN on raw signal segments — paper #1 thesis + paper #12 hybrid 2DCNN-LSTM are the references.
- **Advanced #2**: LSTM forecasting of RMS — paper #19, with the honest caveat about R² < 0.
- **Optimiser**: Nadam by paper #21's empirical comparison across four bearing datasets.

**For Medium-article structural template:**
- Borrow the **table-1-style "Reference vs Proposed" comparison** from paper #25 (Ho MDPI 2021) — list Qiu 2006 wavelet, Wang EMD, Yu HMM-DPCA, my approach with columns for de-noising / filtering / decomposition / prior-knowledge-needed / automated-detection.
- Borrow the **three-archetype maintenance framing** from paper #30 commercial blog for the opening paragraph.
- Borrow the **30% bearings ≈ all-motor-faults statistic** from paper #31 review (or 40% from paper #16).

### Where the research-gap signal points (for PhD proposal Mgg 10+)

Three credible research-gap angles emerge from the corpus, in increasing ambition:

1. **Cross-dataset generalisation of NASA-IMS-trained models** — Khamoudj #6 and Marx #2 both validate on NASA IMS + PRONOSTIA. The natural extension is "if I train on NASA IMS, how well does the model transfer to FEMTO/PRONOSTIA, and what physics-informed adaptation is needed?" This is a research-question-with-Mining-edge if framed as "transfer from public bearing datasets to a mining-specific bearing dataset."

2. **Label-free or weakly-supervised bearing prognostics** — the strong cluster-C signal. The 2024–2026 papers are racing to replace expert-annotated labels with EWMA pseudo-labels (#18), domain-informed augmentations (#2), unsupervised clustering (#6), variance-maximising autoencoders (#8), or GANs (#13). The PhD-level contribution would be a **physics-informed unsupervised HI** specific to mining/mineral-processing rotating equipment — combining the corpus's two strongest threads.

3. **Federated/privacy-preserving prognostics** — paper #18's Fed-TGCN is the only federated paper in the corpus, and it explicitly names data-privacy as one of four bearing-prognostics open problems. Industry-coupled mining HDRs (Curtin WASM, MRIWA scholarship) would value a federated learning framework that lets mining companies collaborate without sharing raw vibration data. This is closest to a Curtin-DR-ISYS-flavoured framing (digital-transformation + privacy-preserving) wrapped around a Mining-WASM application.

Of the three, **angle 2 (label-free physics-informed HI) is the strongest fit for both the methodology preparation done in the mini-research and the mining-domain edge**. Angle 3 is the strongest fit for Curtin DR-ISYS framing if Mining doesn't pan out.

### For cold-email Wave 1 fluency

To speak credibly with WASM / UWA Liu / MRIWA / Murdoch supervisors, the corpus gives me:

- **Citation fluency**: I should be able to drop Qiu/Lee/Lin 2006, Antoni 2006/2007 (spectral kurtosis), McFadden-Smith 1984 (single-defect vibration model), and Smith-Randall 2015 CWRU benchmark study without looking them up.
- **Method fluency**: kurtogram → infogram → autogram → SKRgram → LESGIRgram is the genealogy of band-selection methods, all asking "where in the FFT does the fault energy live?" Sparsity-measure papers (kurtosis → Gini → Box-Cox → GGI) all generalise this with different weights on the squared envelope.
- **Dataset fluency**: NASA IMS = run-to-failure benchmark, 3 tests with documented fault outcomes. PRONOSTIA = FEMTO platform, accelerated-life-test. CWRU = controlled-condition seeded-fault (most-cited but least realistic). Paderborn = newer controlled. MFPT = small mechanical-failure prevention. C-MAPSS = aircraft engine RUL benchmark.
- **Research-gap fluency**: the field is open about its limitations — labelled failure data is scarce in industry, supervised methods don't transfer across operating conditions, and current HIs lack physical interpretability. Anything I propose along these lines will be received as "you've read the right literature."

### Net assessment of this corpus

The corpus is opinionated toward **(a) signal-processing-heavy methods for bearing fault diagnosis** and **(b) recent (2023–2026) deep-learning approaches to NASA IMS**. Notable absences: no CWRU-benchmark papers (Smith-Randall 2015 is cited by ~5 papers but not in corpus), no review of motor current signature analysis (MCSA) as an alternative to vibration, no papers on dynamic-context (variable speed, variable load) which is the hardest real-world deployment scenario. These are gaps I'd note when discussing scope with a supervisor.

For a mini-research exercise of 6–8 weeks, the corpus is **substantially more than I'll be able to act on**. The right move is to extract the concrete inputs above for notebooks 02–04 and treat the rest as a citation pool for the Medium article and PhD proposal.

---

## Per-Paper Notes

Notes are in alphabetical order of file name (the order they appear in the folder). Each entry follows the same structure:

- **Source / format / size** — file metadata.
- **Authors / venue / year** — who wrote it and where it was published, when discoverable.
- **Type** — empirical / review / survey / methodology / domain context / commercial blog.
- **Relevance** — direct (NASA IMS or bearing-specific), adjacent (signal processing, ML, prognostics general), or context (domain framing without direct technical link).
- **Summary** — what the paper actually claims and demonstrates.
- **Key findings / takeaways** — bullet form, what to remember.
- **Connection to corpus / mini-research** — how it relates to other papers here and to the NASA IMS work.

---

### 1. Microcontroller-Based Real-Time Motor Bearing Fault Detection and Diagnosis Using 1D Convolutional Neural Networks

- **Source / format / size:** `175.pdf`, 3.5 MB, ~94 pages + references.
- **Authors / venue / year:** Sertaç Kılıçkaya, Master's Thesis, Izmir University of Economics, Master Program in Electrical and Electronics Engineering, January 2022. Advisor: Prof. Dr. Türker İnce. TÜBİTAK 2210-A funded.
- **Type:** Empirical thesis — covers literature review, model design, and full embedded deployment.
- **Relevance:** Direct (deep learning for bearing fault diagnosis) + methodology (edge deployment, quantization).
- **Summary:** Establishes that 1D Self-Organized Operational Neural Networks (1D Self-ONNs) and their special case 1D CNNs can replace hand-crafted features for bearing fault diagnosis. The thesis benchmarks 1D CNNs and Self-ONNs (with q-orders 3, 5, 7) on two open datasets — **CWRU** and **University of Ottawa Variable Speed** — then collects in-house 3-axis accelerometer data from two single-phase induction motors with four bearing health conditions, trains a 1D CNN on it, quantizes the model, and deploys to an STM32L4 Discovery Kit IoT Node (Arm Cortex-M4) using STM32Cube.AI and TensorFlow Lite for Microcontrollers. Real-time on-device inference is demonstrated. The thesis structure also surveys autoencoders, GANs, RNNs, 2D CNNs, adaptive 1D CNNs, and Self-ONNs as deep-learning approaches to bearing fault diagnosis (Chapter 3) — useful as a literature scaffolding.
- **Key findings / takeaways:**
  - 1D CNNs fold feature extraction and classification into one model, removing the need for hand-engineered features — important counter-frame for the classical baseline I'll build first.
  - Self-ONNs generalise CNNs by replacing linear neurons with non-linear ones; q-order is a hyperparameter (1 = CNN, higher = more nonlinear expressivity).
  - Post-training quantization vs quantization-aware training are both viable for deployment; tables 15–18 compare TFLM and STM32Cube.AI runtimes.
  - Two benchmark datasets used (CWRU + Ottawa) — **NASA IMS is not used here**, so this paper is a methodology reference, not a direct benchmark comparison for my work.
- **Connection to corpus / mini-research:** Best in-corpus reference for the **advanced 1D-CNN model** in notebook 03. Architectures listed in chapter 4 (Tables 7, 11) give a concrete starting point. The embedded-deployment chapters (5–6) are out of scope for mini-research but worth keeping in mind as a possible "discussion" angle in the Medium article (deployment-aware ML, not just lab accuracy).

---

### 2. Domain Knowledge Informed Unsupervised Fault Detection for Rolling Element Bearings

- **Source / format / size:** `3348-Document Upload-11053-2-10-20220702-2.pdf`, 2.4 MB, 9 pages.
- **Authors / venue / year:** Douw Marx, Konstantinos Gryllias (KU Leuven, Flanders Make). *Proceedings of the 7th European Conference of the Prognostics and Health Management Society* 2022. ISBN 978-1-936263-36-3.
- **Type:** Empirical, methodology paper. Open-access (CC BY 3.0 US).
- **Relevance:** **Direct + high signal** — applies the method to the **NASA IMS bearing dataset** explicitly (Section 4, Tables 2–3, Figure 9). Same dataset I'm using.
- **Summary:** Argues that fully supervised bearing-fault methods require labelled failure data that real industry rarely has; fully unsupervised methods can flag anomalies but not diagnose them. The paper proposes a hybrid: use **domain knowledge** (characteristic bearing fault frequencies — BPFO, BPFI, BSF) to synthetically augment healthy training data into expected faulty states, then train an autoencoder with a specialised loss function that pushes the latent space to deviate in known fault directions for each fault mode. At inference, a new sample is projected onto each expected fault direction; the negative log-likelihood gives a per-fault-mode health indicator. Demonstrated on both a phenomenological McFadden–Smith dataset and the NASA IMS dataset (Tests 1, 2, 3 — using bearing 3 ch5 inner race, bearing 4 ch7 ball, bearing 1 ch1 outer race, bearing 3 ch3 outer race). Outer-race faults clearly separable; inner-race and ball faults less so because the squared envelope spectrum (SES) used as input feature isn't equally sensitive to all fault modes.
- **Key findings / takeaways:**
  - **Squared Envelope Spectrum (SES)** is the input feature, not raw signal. Healthy-data SES is augmented by adding triangular peaks at the four characteristic fault frequencies with decaying harmonics.
  - Loss = reconstruction loss + latent direction loss + latent magnitude loss. The two latent losses enforce that healthy-to-faulty movement happens in mode-specific directions and that distance from healthy cluster is similar across modes.
  - Effectiveness depends critically on how informative the input feature is. SES works for outer-race faults but is weak for inner-race/ball — a limitation the authors call out as future work (use time-frequency maps or raw time-domain).
  - Confirms that the **NASA IMS dataset is treated as four labelled fault datasets** in the standard literature: B3-ch5 (Set1 inner race), B4-ch7 (Set1 ball), B1-ch1 (Set2 outer race), B3-ch3 (Set3 outer race) — matches what I saw in the readme PDF.
- **Connection to corpus / mini-research:** Most directly useful paper in the corpus so far. It tells me (a) the canonical labelled subsets of NASA IMS, (b) that SES is a serious feature-engineering choice for bearing, (c) a credible "unsupervised + domain knowledge" angle that would be a natural Discussion-section topic in my Medium article. The authors are KU Leuven — Gryllias's group is a strong reference target in the European bearing-prognostics community.

---

### 3. A Robust Topological Framework for Detecting Regime Changes in Multi-Trial Experiments With Application to Predictive Maintenance

- **Source / format / size:** Markdown abstract, 1.5 KB. Source: Wiley `Journal of Time Series Analysis`, [10.1111/jtsa.70032](https://onlinelibrary.wiley.com/doi/abs/10.1111/jtsa.70032).
- **Authors / venue / year:** Not captured in MD; full author list available on Wiley.
- **Type:** Methodology — change-point detection with topological data analysis. Validated on NASA bearing dataset.
- **Relevance:** Direct (uses NASA bearing dataset for validation) but methodology comes from a different field (topology of time-frequency spectra).
- **Summary:** Proposes a change-point detection framework that operates **across trials** rather than within a single time series. Uses topological analysis of time-frequency characteristics (spectra and spectrograms). Validated on simulated time-varying AR processes and then on vibration signals from the NASA bearing dataset, where the framework successfully identifies bearing failures via time-frequency analysis. Sells itself as flexible to different stationarity assumptions and statistical conditions.
- **Key findings / takeaways:**
  - Provides a "cross-trial" framing — applicable when failure timing varies between bearings.
  - Topological invariants on spectrograms (persistent homology) is the underlying tool — see also paper #19 in this corpus (Global and Local Topology-Aware Attention).
  - Useful as a *concept* even if I don't implement TDA in mini-research — the early-warning + degradation-curve story in EDA notebook 01 is morally adjacent.
- **Connection to corpus / mini-research:** This and the persistent-homology attention paper (file #19) are the two TDA-flavoured items in the corpus. Worth flagging as "fancy methods that exist for this problem" in a Discussion section, but probably out of scope for mini-research execution.

---

### 4. A New Framework Based on Geometric Partition L-Kurtosis Indicator for Bearing Condition Monitoring and Incipient Fault Detection

- **Source / format / size:** Markdown 13.1 KB. Source: SAGE `Structural Health Monitoring`, [10.1177/14759217241293373](https://journals.sagepub.com/doi/abs/10.1177/14759217241293373).
- **Authors / venue / year:** Not captured.
- **Type:** Methodology — health indicator construction.
- **Relevance:** Direct (bearing fault HI design, run-to-failure validation).
- **Summary:** Proposes a new health indicator called **Geometric Partition L-Kurtosis (GLK)** for bearing condition monitoring. Argues traditional HIs (kurtosis, negentropy, Gini index) have weak spots — sensitive to outliers, not robust to noise. GLK leverages L-moments (linear combinations of order statistics, which are more robust than conventional moments). Combined with empirical wavelet transform, the authors construct a **GLKgram** that selects the optimal demodulation frequency band for fault feature extraction. Claims of advantage: better recognition of repetitive transients, resistance to random shocks and strong noise, and clean visualisation of the full-life degradation trend.
- **Key findings / takeaways:**
  - L-moments are an interesting alternative to ordinary moments for health-indicator design; my EDA notebook uses ordinary kurtosis (k=4) — L-kurtosis could be a robustness extension.
  - The "*gram" naming convention (kurtogram, GLKgram, autogram, infogram, Lkurtogram, sparsogram) recurs throughout the bearing-prognostics literature — these are all 2D maps over frequency-band × bandwidth used to pick the best demodulation band.
  - The reference list (52 entries) is itself a goldmine — includes Antoni 2006/2007 (kurtogram), Wang 2020 (sum of weighted normalised square envelope), the Qiu/Lee 2006 IMS reference paper, and others that recur across the corpus.
- **Connection to corpus / mini-research:** The L-Kurtosis idea connects to the Continuous Monitoring fuzzy-entropy paper (file #11), the Optimized Weights Spectrum paper (file #25), and several others on health-indicator design. Together they form the "**HI engineering**" cluster — papers that work hard on the input features before any ML.

---

### 5. Advanced Digital Signal Processing Technique for Asset Health Monitoring

- **Source / format / size:** `Advanced digital signal processing technique for asset health monitoring.pdf`, 9.5 MB, 84 pages.
- **Authors / venue / year:** Siu Ki Ho, MPhil thesis, Department of Electronic and Electrical Engineering, Brunel University London, Academic Year 2018–2022. Principal supervisor: Prof. Wamadeva Balachandran.
- **Type:** Empirical thesis covering literature review, methodology, and two experimental applications.
- **Relevance:** Direct — Chapter 3.3 and 4.2 are bearing fault early detection using the **IMS bearing dataset** (Figure 3.9 explicitly shows the IMS test setup).
- **Summary:** Two applications under one signal-processing technique. Application 1: bolt-looseness detection on bolted joints (bridge health monitoring framing). Application 2: bearing fault early detection on the open-source IMS dataset. The unifying technique is a **Spectral Kurtosis (SK)–based optimal Wiener filter** combined with Hilbert envelope, peak tracking, and Stationary Wavelet Transform de-noising. The thesis surveys data-acquisition systems, MEMS accelerometer development from 2006–2016, signal-processing feature-extraction criteria, and ML approaches (clustering, CNN, ensemble) in the context of structural health monitoring. The conclusion explicitly notes that the SK-based features can be fed into downstream ML models (SVM, Random Forest) to build an automated classification pipeline, reducing false alarms.
- **Key findings / takeaways:**
  - **Spectral Kurtosis as a feature-extraction-and-filter design** is a strong classical baseline for bearing-fault detection — kurtosis at each spectral bin tells you which frequency bands carry the impulsive (fault-related) energy.
  - **Hilbert envelope** of the SK-filtered signal recovers the modulating fault signature — a standard pipeline.
  - The IMS application uses the *spike subtraction* algorithm and stationary wavelet de-noising as pre-processing — useful technique references for my notebook 02.
  - This thesis is "single-author signal-processing heavy" — gives me a worked example of how a research-degree-level project on bearing fault detection is *structured*, including how the literature review is built and how the experimental section is organised.
- **Connection to corpus / mini-research:** The SK + Wiener filter approach is a credible **secondary baseline** beyond my planned RF/XGBoost on tabular features. If I want a third baseline that is "classical signal-processing benchmark", SK→Hilbert envelope→peak tracking is the canonical move. Also useful as a *structural template* for how to organise the Medium article (introduction → state-of-art survey → methodology → experiments → conclusion).

---

### 6. An Unsupervised Data-Driven Framework for Bearing Failure Prognosis via Health Stage Clustering and Artificial Neural Network-Based Remaining Useful Life Estimation

- **Source / format / size:** `An Unsupervised Data-Driven Framework ... RUL Estimation.pdf`, 1.3 MB, 30 pages.
- **Authors / venue / year:** Charafeddine Khamoudj, Fatima Benbouzid-Si Tayeb, Karima Benatchba (ESI, Algeria); Mohamed Benbouzid (Univ. Brest, France). *MDPI Applied Sciences* 2026, 16, 2472 (Special Issue: Technical Diagnostics and Predictive Maintenance, 2nd Edition). Open access, CC BY.
- **Type:** Empirical methodology paper with broad RUL-estimation literature review (Section 2).
- **Relevance:** **Direct + high signal** — uses both PRONOSTIA and NASA IMS datasets. Recent (2026).
- **Summary:** Proposes a fully unsupervised RUL-estimation framework for induction-machine bearings. Pipeline: (1) advanced signal-preprocessing of vibration + temperature signals; (2) **unsupervised CMO-based clustering** to construct bearing Health Stages (HS) automatically — no manual labelling required; (3) a designed **Health Indicator (HI)** is computed from the latest historical observations; (4) a **forecast-ANN** trained on a sliding window forecasts HI values recursively until detecting the failure threshold; (5) a complementary **adjustment-ANN** is trained on the forecast-ANN's residuals during the test phase to correct prediction error. Validated on PRONOSTIA and **NASA-IMS** datasets. The paper also delivers a meaty literature review of model-based vs data-driven, time-series-analysis vs ML approaches to bearing RUL (Section 2), with a comparative table of ARIMA, SARIMA, LSTM, BiLSTM, GRU, RF, SVM-RVM-RFR-GPR (Sparrow-Search hybrid), CSIDTL transfer-learning, multiscale-CNN, recurrent-CNN, ELM-REX, and many others.
- **Key findings / takeaways:**
  - **HS construction is unsupervised** via CMO clustering — directly addresses the labelling-strategy question that my EDA notebook 01 flagged ("3-class healthy / early-fault / severe-fault" guided by kurtosis crossings).
  - **Dual-ANN architecture** (forecast-ANN + adjustment-ANN) is a novel framing — most papers use single LSTM/BiLSTM. The adjustment-ANN absorbs distribution shift between train and test.
  - Confirms a **strategic conclusion** of the field: "deep-learning-based approaches with direct RUL mapping" are the most promising for unsupervised RUL, but require enough data; ANN-based forecast-then-adjust is a defensible middle path for small-data industrial settings.
  - Reference [42] in this paper is NASA IMS, [41] is PRONOSTIA — the canonical pair of public run-to-failure datasets.
- **Connection to corpus / mini-research:** Best "what comes after my mini-research" reference. Mini-research can stop at classification (healthy/degraded/failure); this paper sketches the full RUL extension that a PhD-level project would do. The Section 2 literature review is genuinely useful as a seed for my proposal's lit review later (Mgg 10+). The Algerian + Brest authorship is a useful diversifier — most NASA IMS papers come from a tight US/EU/Chinese cluster.

---

### 7. An Enhanced Empirical Fourier Decomposition Method for Bearing Fault Diagnosis

- **Source / format / size:** Markdown 7.8 KB. Source: SAGE `Structural Health Monitoring` Volume 23, Issue 2, [10.1177/14759217231178653](https://doi.org/10.1177/14759217231178653).
- **Authors / venue / year:** Danchen Zhu, Guoqiang Liu, Bolong Yin + others. Year not captured (likely 2023 from the volume number).
- **Type:** Methodology — signal decomposition for fault feature extraction. Abstract-only in MD.
- **Relevance:** Adjacent (signal processing, not specific to NASA IMS).
- **Summary:** Bearing fault signals are often masked by background interference from complex transmission paths in real machines. The proposed enhanced Empirical Fourier Decomposition (EFD) handles this with three improvements: (1) **trend-line extraction** pre-processing to suppress signal distortion and background noise; (2) **correlation-coefficient-based decomposition number selection** to avoid spurious modes; (3) a **band-improvement strategy** + **weighted harmonics significant index** to pick the optimal modal components. After decomposition, FFT extracts the fault signatures. Validated on simulation and experimental bearing signals (which experimental dataset is not visible in the abstract).
- **Key findings / takeaways:**
  - EFD belongs to the same family as EMD (Empirical Mode Decomposition), VMD (Variational Mode Decomposition), EWT (Empirical Wavelet Transform) — these are all methods that decompose a non-stationary signal into intrinsic mode functions. EFD is Zhou et al. 2022's variant; the present paper enhances it.
  - The references list (in the body of the MD file) is again a survey of the same "*gram + decomposition + filter" cluster — kurtogram, harmogram, TIEgram, autogram, infogram, etc.
- **Connection to corpus / mini-research:** Adjacent — this is a classical-signal-processing competitor to deep learning. Worth knowing exists for the Medium "Related work" paragraph. Probably not implementing.

---

### 8. An Interpretable Health Indicator for Bearing Condition Monitoring Based on Semi-Supervised Autoencoder Latent Space Variance Maximization

- **Source / format / size:** Markdown 2.3 KB (abstract + metadata only). Source: IOP `Measurement Science and Technology` 34(12) 125135 (2023), [10.1088/1361-6501/acf515](https://iopscience.iop.org/article/10.1088/1361-6501/acf515/meta). 77 Dimensions citations, 389 total downloads — well-cited in its niche.
- **Authors / venue / year:** Xieyi Chen, Yi Wang, Lihua Meng, Yi Qin, Baoping Tang. Special issue "Fault Diagnosis and Prognosis of Railway Vehicle System". 2023.
- **Type:** Empirical methodology — deep-learning health indicator construction.
- **Relevance:** Direct (bearing HI construction, autoencoder, run-to-failure validation on two datasets).
- **Summary:** Argues that current deep-learning health indicators predict RUL well but lose anomaly-detection ability and are not interpretable. Proposes a **Semi-Supervised Autoencoder with Latent Space Variance Maximization (SSALSVM)** built on a deep convolutional encoder + single fully-connected decoder. To capture the **early degradation point (EDP)** in the latent space, an auxiliary layer is added to the encoder output. A variance-maximization constraint on the latent space increases sensitivity to abnormality and separates healthy from degraded states more cleanly. The training process is presented via projected variance evolution over epochs. Validated on two datasets (not named in abstract).
- **Key findings / takeaways:**
  - **Early degradation point (EDP) detection** is a distinct sub-problem from RUL prediction and binary fault detection — close to what my EDA notebook 01 calls "early-fault" labelling.
  - Variance-maximization in latent space is a simple but effective regulariser — could inform an advanced-model design for the Medium article ("we tried adding a latent-variance term to the loss").
  - Semi-supervised setting = a few labelled examples + lots of unlabelled — realistic for industrial settings where labelled failure data is scarce. Mirrors the unsupervised theme in papers #2 and #6.
- **Connection to corpus / mini-research:** Three-of-a-kind with paper #2 (Marx & Gryllias — KU Leuven autoencoder with augmented data) and paper #6 (Khamoudj — ANN forecast on unsupervised health stages). Together they form the **"semi/unsupervised health indicator"** cluster — papers that share a common bet that the next research frontier is learning health indicators without labels.

---

### 9. Bearing Fault Diagnosis Based on Multiscale Permutation Entropy and Support Vector Machine

- **Source / format / size:** `Bearing Fault Diagnosis ... MPE and SVM.pdf`, 0.3 MB, 14 pages.
- **Authors / venue / year:** Shuen-De Wu, Po-Hung Wu, Chiu-Wen Wu, Jian-Jiun Ding, Chun-Chieh Wang. Affiliations: National Taiwan Normal University (Mechatronic Technology), National Taiwan University (EE), and ITRI Mechanical & Systems Research Lab, Taiwan. *Entropy* 2012, 14, 1343–1356. ISSN 1099-4300. Open Access.
- **Type:** Empirical methodology — classical-ML approach to bearing fault diagnosis.
- **Relevance:** Adjacent. Uses **CWRU** (not NASA IMS), but the feature-extraction + SVM pipeline is exactly the baseline structure I'm planning to build for NASA IMS.
- **Summary:** Standard three-step pipeline: (1) **Multiscale Permutation Entropy (MPE)** feature extraction from 2048-point windows of vibration signal; (2) one-vs-one multi-class SVM classifier (LIBSVM); (3) accuracy reporting across four CWRU shaft speeds (1730/1750/1772/1797 RPM) and four health states (normal / ball fault / inner race / outer race). Compares MPE against three baselines: time-domain-and-frequency-domain statistical formulas (TDFDSFs, 16 features), single-scale Permutation Entropy (PE), and Multiscale Entropy (MSE). MPE wins decisively — 99%+ accuracy at all four speeds, robust across training-set sizes from 10% to 50% of total samples. Even using only 5 of the 20 multi-scale features still yields >99% accuracy.
- **Key findings / takeaways:**
  - **PE alone gives only 74–85% accuracy; multiscale aggregation lifts it to >99%.** Multiscale = coarse-grain the signal at scales 1..S via averaging, then compute PE at each scale.
  - **2048-point windows** — that's ~100 ms at 20 kHz. A pragmatic chunk size for windowed feature extraction.
  - **One-vs-one SVM** is the standard multi-class strategy; with c classes you train c(c-1)/2 binary classifiers and vote.
  - The paper is a 2012 *Entropy*-journal classic — useful as a citation for the "classical features + SVM" baseline reference in my Medium article.
- **Connection to corpus / mini-research:** Closest in spirit to **my planned baseline notebook 02–03** — hand-crafted features + classical ML classifier. Even if I use RF/XGBoost instead of SVM, the pipeline structure (window-extract-classify-report-accuracy-per-class-and-per-condition) is transferable. The MPE feature is a candidate to add to my feature set alongside RMS / kurtosis / FFT bins.

---

### 10. Construction of a Comprehensive Degradation Index for Rolling Bearings Based on Feature Weighting and Multidimensional Scaling Analysis

- **Source / format / size:** Markdown 1.5 KB (abstract only).
- **Authors / venue / year:** Not captured in MD. Likely *Structural Health Monitoring* journal based on the format of sibling papers.
- **Type:** Methodology — degradation index construction.
- **Relevance:** Direct (rolling-bearing health monitoring, degradation index).
- **Summary:** Pipeline: (1) extract time-domain features from vibration signal; (2) Pearson-correlation feature selection; (3) **Multi-Resolution Singular Value Decomposition (MRSVD)** for deep-level feature extraction; (4) **Gini-index + Box–Cox** sparsity weighting to amplify sensitive features; (5) **Multidimensional Scaling (MDS)** for dimensionality reduction; (6) integrate the resulting features into a single comprehensive degradation index. Experimental results claim "significantly improved accuracy and timeliness" of fault detection on bearing run-to-failure data — specific dataset not visible in abstract.
- **Key findings / takeaways:**
  - Combines three things that don't usually appear together: MRSVD (decomposition), Gini/Box-Cox weighting (sparsity), MDS (dimensionality reduction).
  - **MDS** is a manifold-learning step that I hadn't been planning to use — could be a useful alternative to PCA for compressing my feature matrix before classifier.
  - Gini index here is used the same way as Box–Cox sparsity in the wider corpus (papers #4 GLK, #25 optimized weights spectrum, etc.) — a sparsity measure on the squared envelope spectrum.
- **Connection to corpus / mini-research:** Adjacent — abstract-level reference for the "**HI engineering**" cluster. If I want to write the Medium "Related work" section, this is one of the multi-step pipelines I'd cite as a comparison to my much simpler RMS+kurtosis approach.

---

### 11. Continuous Monitoring of Rolling Element Bearing Health by Nonlinear Weighted Squared Envelope-Based Fuzzy Entropy

- **Source / format / size:** Markdown 10.2 KB. Source: SAGE `Structural Health Monitoring` 23(1), [10.1177/14759217231163090](https://doi.org/10.1177/14759217231163090).
- **Authors / venue / year:** Khandaker Noman, Yongbo Li (Northwestern Polytechnical University, China), Shun Wang + others. 2024 issue.
- **Type:** Methodology — nonlinear entropy-based health indicator.
- **Relevance:** Direct (REB health monitoring, run-to-failure experimental validation).
- **Summary:** Standard Fuzzy Entropy (FE) measures the irregularity of a vibration signal as a proxy for fault severity — but under heavy noise the transient impulses of an incipient fault are buried, and FE fails to detect either onset or progression. The proposed **Weighted Square Envelope-based Fuzzy Entropy (WSEFE)** first computes the squared envelope of the signal, weights it to suppress background noise, then computes Fuzzy Entropy. Validated on one simulated case + two run-to-failure experimental datasets (one is **NASA IMS**, citation [37]: Lee, Qiu, Yu — IMS Univ Cincinnati 2007). Outperforms original FE, conventional Permutation Entropy, and Multiscale FE (MFE) for continuous monitoring.
- **Key findings / takeaways:**
  - **Squared envelope = pre-amplification before entropy.** Same idea as paper #2 (Marx/Gryllias) using SES as the input feature to the autoencoder — squared envelope concentrates fault energy.
  - Confirms (again) that NASA IMS is the standard validation dataset for new health-indicator proposals.
  - The reference list (39 entries in MD body) is a parallel-structure goldmine: includes McFadden–Smith 1984 (the modeling foundation for bearing fault frequencies), Qiu/Lee 2006 (the IMS reference paper), Bandt-Pompe 2002 (permutation entropy seminal), Yan-Liu-Gao 2012, and many others.
- **Connection to corpus / mini-research:** **Direct sibling to paper #4 (GLK indicator)** — both papers compete on building better-than-kurtosis health indicators on NASA IMS-style data. If I want a "what's the modern alternative to kurtosis?" framing in the Medium article, WSEFE and GLK are the two natural answers. Probably out of scope for implementation in mini-research (each is a small paper-worth of work to reproduce); useful as Discussion-section citations.

---

### 12. Deep Learning Neural Networks with Input Processing for Vibration-Based Bearing Fault Diagnosis Under Imbalanced Data Conditions

- **Source / format / size:** Markdown 13.8 KB (abstract + full reference list).
- **Authors / venue / year:** Not captured (authors hidden behind paywall metadata). Source: SAGE `Structural Health Monitoring`, [10.1177/14759217241246508](https://journals.sagepub.com/doi/abs/10.1177/14759217241246508). 2024.
- **Type:** Empirical — comparison study of DL architectures for bearing fault diagnosis under class imbalance.
- **Relevance:** **Direct + high signal** — explicitly uses three datasets including **NASA IMS bearing dataset with five fault classes** (alongside CWRU and Paderborn).
- **Summary:** Real industrial datasets are imbalanced — many normal samples, few labelled failure samples. The paper investigates how four DL architectures behave under imbalance: LSTM, 1D-CNN, 2D-CNN, and a novel hybrid **2DCNN-LSTM**. Compares two input modes: direct raw vibration vs. signal-processed inputs (Fourier transform, continuous wavelet transform). The hybrid 2DCNN-LSTM with multi-channel input (raw signal + mean channel + variance channel) outperforms standalone CNN or LSTM, with or without input processing. **NASA IMS is treated as a 5-class classification dataset** — likely the four labelled fault subsets plus healthy. The paper's reference list includes the **Gousseau/Antoni 2016** and **Cavalaglio Camargo Molano 2019** analyses of NASA IMS specifically — two papers worth tracking down separately for IMS-specific feature-engineering wisdom.
- **Key findings / takeaways:**
  - **Imbalanced data is the real-world default** — a credible Discussion-section angle for the Medium article ("our baseline assumes a balanced dataset; in practice…").
  - **Hybrid 2DCNN-LSTM with augmented channels (raw + mean + variance)** beats single-stream models — a concrete model design idea if I want to push notebook 03 further.
  - The dataset access notes in the abstract (manufacturingnet.io/html/datasets.html as a CWRU+Paderborn mirror) are useful pointers.
- **Connection to corpus / mini-research:** **The single most directly applicable paper for notebook 03 (advanced model).** Multi-channel input idea + hybrid CNN-LSTM are both candidates worth trying. Also confirms my "Set 2 has imbalanced failure samples" intuition from EDA notebook 01 — this paper says imbalance is the norm, not the exception.

---

### 13. Deep Semi-Supervised Generative Adversarial Fault Diagnostics of Rolling Element Bearings

- **Source / format / size:** Markdown 6.5 KB (abstract + reference list).
- **Authors / venue / year:** Verstraete D, Droguett EL, Meruane V + others. *Structural Health Monitoring* 19(2): 390–411, 2020, [10.1177/1475921719850576](https://journals.sagepub.com/doi/10.1177/1475921719850576).
- **Type:** Empirical — GAN-based fault diagnostics.
- **Relevance:** Adjacent. Uses two public vibration datasets but neither named in abstract; almost certainly CWRU is one (based on reference [14] = Loparo CWRU page).
- **Summary:** Argues that labelling all data in a big-machinery setting is cost-prohibitive, so proposes both a fully unsupervised and a semi-supervised **GAN-based** method for fault diagnostics. The unsupervised method clusters latent representations from the GAN; the semi-supervised method uses a small labelled subset to anchor a classifier on top of GAN features. Validates on two public bearing datasets. Reference list includes Goodfellow 2014 (vanilla GAN), Radford-Metz-Chintala 2016 (DCGAN), Salimans 2016 (improved training of GANs), Gulrajani 2017 (WGAN), Metz 2016 (unrolled GAN), InfoGAN, and Smith & Randall 2015 (the benchmark study of CWRU).
- **Key findings / takeaways:**
  - **Generative-then-classify** is one of three families of "no-labels-needed" methods in this corpus — alongside autoencoder-based (paper #2, #8) and unsupervised-clustering-based (paper #6).
  - The paper is well-cited (2020) and would be a natural Discussion citation when arguing that supervised methods like mine have a labelling constraint that GANs sidestep.
- **Connection to corpus / mini-research:** Same "semi/unsupervised cluster" as papers #2, #6, #8. Together they're the **research-gap signal** — the field is openly drifting toward unsupervised because labelling industrial run-to-failure data is hard. My mini-research's supervised approach (using the timestamps as labels) is the easy entry point; the proposal-level extension would be one of these no-labels-needed methods. **Good source of "next steps" framing for the Medium conclusion + the PhD proposal positioning.**

---

### 14. Exploration in Using the Weibull Distribution for Characterizing Trends in Bearing Failure Operational Changes

- **Source / format / size:** Markdown 1.8 KB (abstract only).
- **Authors / venue / year:** ASME IMECE 2022 Conference, paper V02BT02A026. Authors not captured.
- **Type:** Empirical — distributional / reliability modelling.
- **Relevance:** Direct — explicitly compares laboratory data to **NASA/IMS bearing run-to-failure dataset**.
- **Summary:** A laboratory bearing test stand is used to generate failure data under controlled fatigue and contamination failure modes. The resulting failure trajectories are compared to NASA IMS run-to-failure data. A **Weibull distribution** is fitted to the data of both sources; the resulting distribution parameters (shape, scale) are tracked across "damage stages." The Weibull parameters trend in a similar way for the lab data and NASA IMS, validating the lab setup as a controlled-environment proxy. The motivation is improving the *end-of-RUL* calculation by understanding the underlying distribution of bearing life under varying operating conditions.
- **Key findings / takeaways:**
  - **Weibull distribution** is the canonical reliability-engineering choice for bearing life modelling — useful framing for the Medium article's "what is RUL really?" framing.
  - Confirms the IMS dataset has a recognisable Weibull-shape degradation trajectory under fatigue mode — possibly an angle for my Discussion section ("the kurtosis spike on B1 corresponds to the Weibull shape parameter shift").
  - The "Purposeful Failure Methodology" mentioned in the abstract is a laboratory protocol worth knowing about as a domain-fluency point.
- **Connection to corpus / mini-research:** Adjacent. The Weibull angle isn't where my mini-research lives, but it's a credible Discussion-section bridge between classification (what my mini-research does) and reliability engineering (what an industrial deployment would need).

---

### 15. Extracting Failure Modes from Vibration Signals (NASA IMS tutorial)

- **Source / format / size:** Markdown 40.1 KB — substantial document, R-based step-by-step tutorial.
- **Authors / venue / year:** Not captured in document (likely a blog post or tutorial from the bearing-prognostics community). Source URL not in MD.
- **Type:** **Tutorial / worked example** — full code-and-prose walk-through of NASA IMS Set 1 from raw files to a classifier.
- **Relevance:** **Direct + extremely high signal** — this is essentially the same project I'm building, with concrete labelling decisions and feature engineering already worked out.
- **Summary:** Walks through the NASA IMS dataset Set 1 (the 8-channel test with B1+B2 not actually failing, B3 inner race, B4 ball/rolling element) end-to-end in R. Proposes a **7-class labelling scheme** with explicit timestamp ranges per bearing:
  - **Bearing 1** — early (12.06.24 → 09.14.13), suspect, normal, suspect again, imminent failure
  - **Bearing 2** — early, normal, suspect, imminent failure
  - **Bearing 3** — early, normal, suspect, inner race failure
  - **Bearing 4** — early, normal, suspect, rolling element failure, stage 2 failure
  - The seven distinct states across all four bearings: *Early, Normal, Suspect, Imminent Failure, Inner Race Failure, Rolling Element Failure, Stage 2 Failure.*

  Provides exact **bearing geometry parameters** for the Rexnord ZA-2115 used in the experiment: N=16 rolling elements, B_d=0.331 in (rolling element diameter), P_d=2.815 in (pitch diameter), φ=15.17° (contact angle), n=2000 rpm. With these, the four bearing fault frequencies BPFI / BPFO / BSF / FTF can be computed exactly — concrete numbers to look for in the FFT spectrum.

  Lists the canonical feature set: time-domain (RMS, kurtosis, skewness, peak-to-peak, Shannon entropy, AR(8) coefficients, shape/crest/impulse/margin factors) and frequency-domain (vibration levels at characteristic frequencies, frequency centre, mean-square frequency, spectral skewness/kurtosis/entropy, higher-order spectra). Notes a subtle data quirk: 20,480 samples per file at 20 kHz = 1.024 seconds, not exactly 1 second.

- **Key findings / takeaways:**
  - **Use the proposed 7-class labelling scheme directly in notebook 02.** It's a much more sophisticated story than my notebook 01's "healthy / early-fault / severe-fault" placeholder, and it's authoritative for Set 1.
  - **Bearing geometry parameters are gold** — I can compute BPFI/BPFO/BSF/FTF exactly and overlay them on FFT plots in notebook 02. This is the single most valuable concrete data point in the entire corpus.
  - **AR(8) coefficients** are a feature category I hadn't planned to include — but the tutorial mentions them, and they're standard in this literature.
  - The note about 20,480 ≠ 20,000 samples per second is exactly the kind of subtle dataset detail that makes a Medium article more credible.
- **Connection to corpus / mini-research:** **The single most directly useful document in the corpus.** Treats Set 1; my notebook 01 focused on Set 2. So this tutorial actually pushes me to **also process Set 1**, with its richer 8-channel data and the explicit 7-class labelling. Should be the primary reference for notebook 02's labelling section. Worth tracking down the original source URL to cite properly.

---

### 16. Failure Identification and Analysis for High-Voltage Induction Motors in the Petrochemical Industry

- **Source / format / size:** Markdown 3.3 KB (abstract + reference list).
- **Authors / venue / year:** O. V. Thorsen, M. Dalva. *IEEE Transactions on Industry Applications* 35(4): 810–818, July–Aug 1999, [10.1109/28.777188](https://ieeexplore.ieee.org/document/777188).
- **Type:** Survey / empirical — industrial reliability statistics.
- **Relevance:** Context. **Not a bearing-specific paper** — it's an industrial-reliability survey of failure modes in high-voltage induction motors used in petrochemical plants (100–1300 kW). Useful as domain framing.
- **Summary:** Catalogues failure modes, initiators, contributors, and underlying causes across 483 motor units = 6,135 unit-years of operating experience in petrochemical industry. Compares against the earlier 1985 IEEE Large Motor Reliability Survey. The point of the paper: bearings are among the largest sources of motor failures (the 1985 survey found ~40% of motor failures are bearing-related); maintenance strategy and protection methods matter for failure rates.
- **Key findings / takeaways:**
  - **Industry-level confirmation that bearing failures dominate rotating-machine failure statistics.** Useful one-line opener for the Medium article: "across the petrochemical industry, bearings account for ~40% of motor failures (Thorsen & Dalva 1999, IEEE)."
  - The 1985 IEEE survey is the canonical citation for "bearings are the leading failure mode" — Thorsen & Dalva 1999 is the modern update.
- **Connection to corpus / mini-research:** Pure framing paper. Use in the Medium article's "why this matters" introduction. The Mining/WASM cold-email framing is even stronger when the same statistic applies to mineral processing rotating equipment as it does to petrochem motors.

---

### 17. Fault Prognosis and Predictive Maintenance via Big Data Analysis for Aircraft Maintenance

- **Source / format / size:** Markdown 2.9 KB (abstract only). Source: Springer book chapter, [10.1007/978-981-96-6235-7_37](https://link.springer.com/chapter/10.1007/978-981-96-6235-7_37).
- **Authors / venue / year:** Not captured. Book chapter in a 2025 Springer collection.
- **Type:** Empirical — applied condition monitoring on the IMS Rexnord ZA-2115 bearings.
- **Relevance:** Direct — explicitly uses the **NASA IMS Rexnord ZA-2115 bearings**.
- **Summary:** Frames bearing condition monitoring under an "aircraft maintenance / big data" banner but the underlying dataset is the IMS Rexnord ZA-2115 bearings. The first simulation test shows vibration variance is stable (normal operation); the second and third show variance spiking (degradation). Estimates Remaining Useful Life via **linear regression** in scikit-learn — reports **time to failure of 284.19 hours at ~84.5% accuracy** vs the actual failure time. References point to the Miltos-90 GitHub repo for "Failure_Classification_of_Bearings" on NASA IMS — a public implementation worth looking up.
- **Key findings / takeaways:**
  - **Linear regression on degradation feature (RMS variance) → RUL = 284.19 h, ~84.5% accuracy** — a credible baseline number to cite when discussing what a "naive RUL" approach looks like.
  - Confirms (again) that NASA IMS uses **Rexnord ZA-2115 double-row bearings**.
  - Points to the [Miltos-90 GitHub repo](https://github.com/Miltos-90/Failure_Classification_of_Bearings) — almost certainly the source for the tutorial in paper #15 ("Extracting Failure Modes"). I should check this repo for code I can reuse.
- **Connection to corpus / mini-research:** Adjacent — provides a quotable "84.5% accuracy with linear regression" benchmark for the Medium article's results section. Strongest signal: the GitHub repo link is worth following up.

---

### 18. Federated Temporal Graph Learning for Weakly Supervised Bearing Anomaly Detection

- **Source / format / size:** `Federated Temporal Graph Learning ... .pdf`, 3.8 MB, ~16 pages.
- **Authors / venue / year:** Khagendra Darlami, Lalit Awasthi. School of Computer Science and Technology + School of Artificial Intelligence, Nanjing University of Information Science and Technology, China. *Scientific Journal of Engineering Research* 2(2), 24 Feb 2026, e-ISSN 3109-1725.
- **Type:** Empirical methodology — federated learning + graph neural networks on NASA IMS.
- **Relevance:** **Direct + high signal** — recent 2026 paper using **NASA IMS 12 bearings** under leave-one-bearing-out evaluation.
- **Summary:** Names four open problems in industrial bearing prognostics: (1) high class imbalance, (2) absence of fault-type annotations in industry, (3) data-privacy constraints that prevent centralised aggregation, (4) non-IID degradation across geographically dispersed assets. Proposes **Fed-TGCN** — a weakly supervised, federated Temporal Graph Convolutional Network. Each client = one leave-one-bearing-out fold over NASA IMS's 12 bearings (3 tests × 4 bearings). Each bearing's 1-second snapshots are segmented into 40 ms windows (800 samples), from which **six physics-informed statistical features** are computed: (1) envelope RMS f1, (2) kurtosis f2, (3) log-band energy in BPFO=119Hz±5 band f3, (4) log-band energy 2000–10000 Hz f4, (5) log-band energy in BPFI=181Hz±5 band f5, (6) raw RMS f6. The RMS f6 is later excluded as it tracks load not fault. Pseudo-labels come from **EWMA thresholding on a multi-band health indicator** (a weighted sum of features 1–5 with weights [0.30, 0.25, 0.20, 0.15, 0.10] derived from a 10% burn-in window). A hybrid spatio-temporal graph is built per bearing (nodes = 40ms windows, edges = sequential + k-NN by feature similarity). A T-GCN model (2 GCN + 1 GRU + 2 Linear) trains per client; the server aggregates via **FedAvg** under strict **Leave-One-Bearing-Out (LOBO)** evaluation across all 12 bearings. Results: Fed-TGCN achieves AP=0.675±0.276 and MCC=0.636±0.285, outperforming centralised T-GCN, GNN, LSTM, 1D-CNN, and Isolation Forest baselines.
- **Key findings / takeaways:**
  - **Concrete bearing fault frequencies for NASA IMS at 2000 RPM: BPFO ≈ 119 Hz, BPFI ≈ 181 Hz.** These match paper #15's geometry formulas — same physical setup, computed values. **Use these in notebook 02 directly** when overlaying fault frequencies on the FFT spectrum.
  - **40 ms windows (800 samples at 20 kHz) → 6-D feature vector per window**, then mean/max aggregated per file. This is a concrete feature-engineering recipe to copy.
  - **EWMA-based adaptive thresholding for pseudo-labels** is a clean weakly-supervised labelling alternative to my paper-#15-derived timestamp labels. Worth comparing both.
  - **LOBO evaluation** = train on 3 bearings of a test, hold out 1 — the right cross-validation protocol for this dataset. Significantly more honest than random train/test splits within bearings.
  - **MCC** (Matthews Correlation Coefficient) and AP (Average Precision) are the right metrics under class imbalance — the paper makes a strong case against plain accuracy/F1 for this problem.
  - The dataset description (Table 1) confirms my counts: Set1=2156 files, Set2=984 files, Set3=6324 files (matching my own EDA notebook 01 finding that Set 3 has more than the documented 4448).
- **Connection to corpus / mini-research:** **The single most concretely actionable paper in the corpus for notebook 02 and beyond.** Gives me: (1) exact fault frequencies, (2) a 6-D feature recipe to copy, (3) an EWMA pseudo-labelling alternative, (4) LOBO evaluation protocol, (5) MCC + AP metric choices. If I were going to mirror one paper's methodology section in my Medium write-up, this would be the one. The federated-learning angle is out of scope for mini-research, but everything before that (features → labels → metrics) is directly applicable.

---

### 19. Future Vibration Estimation Using LSTM for Condition-Based Maintenance of Aircraft Systems

- **Source / format / size:** `Future-Vibration-Estimation-Using-LSTM ... .pdf`, 0.7 MB, 7 pages.
- **Authors / venue / year:** Hüseyin Şahin, Ömer Faruk Göktaş. Vocational School of Technical Science, Ankara Yıldırım Beyazıt University, Turkey. *ICEEECS 2025* (2nd International Conference on Advances in Electrical, Electronics, Energy, and Computer Sciences), pp.169–175. CC BY-NC-ND 4.0.
- **Type:** Empirical — LSTM-based forecasting of vibration RMS on NASA IMS.
- **Relevance:** Direct (uses **NASA IMS bearing dataset**; framed as aircraft CBM but the data is the same Rexnord ZA-2115 IMS test rig).
- **Summary:** A short conference paper (~7 pages) that walks through the entire CBM workflow: motivation, related work table (benefits/challenges of RUL forecasting), LSTM architecture, methodology, results. Methodology: load NASA IMS Set 1 vibration → Min-Max normalisation → window into sequences → compute RMS per window as the modelling target → train LSTM to forecast future RMS values → evaluate with MAE and RMSE. Results: MAE = **0.0010** (on min-max normalised data), RMSE near zero, but R² = **−0.6838** — meaning the model captures short-term trend but fails to explain variance in the highly non-linear failure regime. Honest about the limitation. Concludes that LSTM is good for CBM but the long-term stochasticity of bearing failure isn't fully captured.
- **Key findings / takeaways:**
  - **LSTM on min-max normalised RMS** is a working pipeline — but R² = −0.68 is a red flag that the model is essentially predicting the recent mean.
  - The paper's own Figure 5 visualises that LSTM with 100% training data fails to catch the final RMS spike — exactly the same Bearing 1 outer race failure jump I see in my notebook 01.
  - **A useful counter-example for the Medium article**: "naive LSTM forecasting on raw RMS doesn't capture the failure spike — the failure event is by definition out of the training distribution." Good Discussion-section angle.
- **Connection to corpus / mini-research:** Confirms the LSTM-on-RMS approach as a credible baseline but flags its honest limitation. Combined with paper #18 (Fed-TGCN's MCC=0.636), this gives me a realistic anchor: "current state-of-art on NASA IMS is MCC ~0.6, not 0.99." If my Medium article reports a baseline RF/XGBoost at ~0.7–0.8 F1 I'm in a credible range, not magically beating literature.

---

### 20. Global and Local Topology-Aware Attention with Persistent Homology and Euler Biases for Time-Series Forecasting

- **Source / format / size:** Markdown 2.3 KB. Source: arXiv:2605.03163 (note: future-dated, likely placeholder).
- **Authors / venue / year:** Usef Faghihi, Amir Saki. arXiv preprint.
- **Type:** Methodology — augments transformer attention with topology-derived inductive biases.
- **Relevance:** Direct (one of the validation datasets is **NASA IMS bearing degradation**).
- **Summary:** Proposes adding **persistent-homology features (H0–H2)** and **Euler-characteristic transforms** as inductive biases to standard dot-product attention. The topology features are computed via exact Vietoris-Rips on chunks of the time series. A "validation-gated local residual" learns when to use the topology bias based on held-out validation data. Three architecture families tested: lightweight Ridge attention, PatchTSTForRegression, TimeSeriesTransformerForPrediction. Real-world datasets used in evaluation: CO2, S&P 500 returns, and **NASA IMS bearing degradation**. Across the 12 paired tests on PatchTST, the model improves on baseline in 33 of 63 units with a mean relative RMSE reduction of 23.5% (p=3.5e-5). TimeSeriesTransformer improves in 47 of 63 units (47.8% RMSE reduction, p<1e-4).
- **Key findings / takeaways:**
  - **Topology-aware attention helps on NASA IMS too**, not just synthetic data — credible evidence that the degradation curve has geometric/topological structure beyond first/second moments.
  - Two of the corpus papers (this one and #3 "Robust Topological Framework") form a TDA-flavoured sub-cluster. The TDA angle is academically respectable but probably not where my mini-research lives.
- **Connection to corpus / mini-research:** Adjacent. Worth a citation in the Medium "Related work" paragraph as evidence that the IMS dataset has been used to validate exotic methods, but not implementing TDA in this project.

---

### 21. Gradient Descent-Based Optimization Algorithms for Batch-Normalized Convolutional Neural Networks: A Comparative Performance Analysis Using FEMTO, NASA, CWRU and MFPT Bearing Datasets

- **Source / format / size:** Markdown 4.0 KB (abstract + DOI metadata). Source: Springer LNNS Vol 711, SAI 2023 conference, [10.1007/978-3-031-37717-4_25](https://link.springer.com/chapter/10.1007/978-3-031-37717-4_25).
- **Authors / venue / year:** C. Usigbe, X. Perry. SAI 2023 (Intelligent Computing conference). September 2023.
- **Type:** Comparative methodology — optimiser comparison for batch-normalised CNN on bearing fault data.
- **Relevance:** Direct — explicitly uses **NASA** alongside FEMTO/CWRU/MFPT.
- **Summary:** Trains a standard Batch-Normalized CNN architecture on four bearing datasets (FEMTO, NASA, CWRU, MFPT) using nine stochastic gradient-based optimisers: SGD, SGDm, SGDm+Nesterov, RMSProp, Adam, AdaGrad, AdaDelta, Adamax, **Nadam**. Reports convergence speed, accuracy, and loss across all four datasets and nine optimisers. **Nadam consistently best across all four datasets; AdaDelta worst.**
- **Key findings / takeaways:**
  - **Use Nadam as the default optimiser** for the deep-learning model in notebook 03 — empirically validated across four bearing datasets including NASA.
  - **Avoid AdaDelta** — worst performer in this benchmark.
  - The paper is a useful "boring methodology baseline" citation — the kind of comparative study that's important to know exists but isn't a research contribution by itself.
- **Connection to corpus / mini-research:** Direct methodology input: optimiser choice for the deep model in notebook 03. Other than that, low novelty — the paper itself is more of a tutorial-style comparative study.

---

### 22. Intelligent Railway Wagon Health Assessment Using IoT Sensors and Predictive Analytics for Safety-Critical Applications

- **Source / format / size:** `Intelligent Railway Wagon Health Assessment ... .pdf`, 0.6 MB, ~21 pages.
- **Authors / venue / year:** Shiva Kumar Mysore Gangadhara, Krishna Alabhujanahalli Neelegowda, Anitha Arekattedoddi Chikkalingaiah, Naveena Chikkaguddaiah. SJB Institute of Technology + Government SKSJ Technological Institute, Karnataka, India. *MDPI IoT* 2026, 7, 32. CC BY.
- **Type:** Empirical — sensor-based health assessment framework with decision logic.
- **Relevance:** Direct (validated on a publicly available run-to-failure bearing dataset whose degradation characteristics match railway axle bearings — likely NASA IMS based on context).
- **Summary:** A sensor-based health assessment framework for railway wagons. Pipeline: multi-sensor acquisition → systematic signal preprocessing → feature-based health indicator construction → temporal degradation analysis → **safety-oriented decision logic** that classifies operating conditions into three discrete health states (Normal: h≥θn, Degraded: θc≤h<θn, Critical: h<θc) and confirms alerts through a **persistence window** to reduce false alarms from transient disturbances. Algorithm 1 formalises the procedure: maintain a health history buffer of length L per component, compute degradation trend Δh, compute weighted risk r=w(1-h), require η confirmations within a validation window ΔT before issuing an alert. Validated on a public run-to-failure bearing dataset; reports improved classification accuracy, higher detection reliability, lower false alarm rates, and lower detection latency vs representative baselines.
- **Key findings / takeaways:**
  - **Three-class labelling (Normal / Degraded / Critical)** with two thresholds θn and θc — almost identical to the labelling scheme I floated in notebook 01.
  - **Persistence-window alert validation** is the right way to reduce false alarms — a useful tweak in any downstream "should we trigger maintenance?" decision logic, even though my mini-research stops at classification.
  - The paper's formal optimisation framing (Equations 4–8: minimise weighted health-degradation risk subject to maintenance capacity constraints) reads like an operations-research framing of predictive maintenance — useful for the proposal's "why this matters operationally" angle.
- **Connection to corpus / mini-research:** Direct labelling-strategy validation: three-class state classification with persistence-based alerts is exactly the framing I'm building toward. Algorithm 1 is essentially a pseudo-code blueprint I can adapt.

---

### 23. Investigation on Early Fault Classification for Rolling Element Bearing Based on the Optimal Frequency Band Determination

- **Source / format / size:** Markdown 7.1 KB. Source: Springer `Journal of Intelligent Manufacturing` 26: 189–198, 2015 (published April 2013), [10.1007/s10845-013-0772-8](https://link.springer.com/article/10.1007/s10845-013-0772-8).
- **Authors / venue / year:** Hongkun Li, Xiaoting Lian, Cheng Guo, Pengshi Zhao. 2013/2015. 841 accesses, 29 citations.
- **Type:** Methodology — optimal demodulation frequency band selection for envelope analysis.
- **Relevance:** Adjacent — REB fault diagnosis, specific dataset not in abstract.
- **Summary:** Standard envelope-analysis pipeline for bearing diagnosis depends critically on picking the right frequency band before demodulation. The paper proposes using a **reference signal** (a known healthy condition) to compute the variance ratio per band, then selecting the band that maximises the variance increase from healthy to faulty. Validated on simulation + test-rig + practical monitored bearings. References include the canonical Antoni 2006 (Spectral Kurtosis), Antoni 2007 (kurtogram), Barszcz & Jablonski 2011 (optimal band selection), Randall & Antoni 2011 (the bearing diagnostic tutorial).
- **Key findings / takeaways:**
  - **Reference-based variance ratio** is a simple, interpretable alternative to the kurtogram for optimal band selection. Worth knowing as a baseline.
  - The reference list is a who's-who of classical bearing-diagnostic signal processing — every key citation in this corpus's "*gram" family appears here.
- **Connection to corpus / mini-research:** Adjacent. Use as background reference when discussing why FFT-band features are reasonable inputs to ML, even though I'm not implementing reference-based band selection.

---

### 24. Log-Envelope Sparsity Measures in Machine Condition Monitoring: Insights into Gaussian Distribution Variance

- **Source / format / size:** `Log-Envelope Sparsity Measures ... .pdf`, 8.4 MB, preprint (not yet peer-reviewed). SSRN abstract 6294734.
- **Authors / venue / year:** Jiamei Li, Dong Wang, Zhike Peng. Ningxia University + Shanghai Jiao Tong University. Submitted to *Mechanical Systems and Signal Processing*, January 2026.
- **Type:** Theoretical + empirical — proposes Log-Envelope (LE) as a superior input to sparsity measures vs Squared Envelope (SE).
- **Relevance:** Direct — bearing full-life-cycle dataset (likely IMS though not explicitly named in the read pages).
- **Summary:** Sparsity measures (SMs) — kurtosis, negative entropy, L2/L1 norm, Hoyer measure, Gini index, GI2, GI3, GGI, BCSM — are widely used for early fault detection. The traditional input to these SMs is the **Squared Envelope (SE = E²)**; this paper argues replacing it with the **Log-Envelope (LE = ln(SE) = 2 ln(E))** improves early fault detection. Derives theoretical values of SMs under Gaussian noise for both SE and LE inputs, showing that LE-based SMs have values that depend on Gaussian variance σ² (and are monotonic in it) — making them inherently sensitive to subtle changes in noise level that mask early faults. SE-based SMs are theoretically variance-independent and thus less sensitive. Validation: simulated experiments + actual bearing full-life-cycle data confirm LE-based SMs detect incipient faults earlier and produce more monotonic degradation curves.
- **Key findings / takeaways:**
  - **LE = ln(SE) is the proposed input transform** — a small but powerful change. The theoretical analysis (LE follows Log-Gamma distribution with scale 2σ²) is the paper's mathematical core.
  - **Box–Cox Sparsity Measure (BCSM)** is the unifying family of sparsity measures — kurtosis = BCSM with m=1, NE = BCSM with m=0, plus continuous interpolations.
  - **Gini index family** (GI, GI2, GI3, GGI) all use sorted envelope values; differences are in the weighting kernel.
  - Useful for understanding what the "sparsity measure" literature actually means when papers in this corpus (e.g., #4 GLK, #10 MRSVD, #11 WSEFE) cite "Gini index" or "Box-Cox" weighting — they all derive from this BCSM family.
- **Connection to corpus / mini-research:** Adjacent — too theoretical for inclusion in mini-research, but provides the conceptual scaffold for understanding why papers in the corpus obsess over sparsity measures and envelope transformations. If I want a deeply credible Discussion-section paragraph about "how the literature is evolving from SE to LE inputs," this paper is the citation.

---

### 25. Monitoring of Industrial Machine Using a Novel Blind Feature Extraction Approach

- **Source / format / size:** `Monitoring of Industrial Machine ... .pdf`, 1.4 MB, 11 pages.
- **Authors / venue / year:** Siu Ki Ho, Harish Chandra Nedunuri, Wamadeva Balachandran, Jamil Kanfoud, Tat-Hean Gan. College of Engineering, Design and Physical Sciences, Brunel University London. *MDPI Applied Sciences* 11(13): 5792, 22 June 2021, [10.3390/app11135792](https://doi.org/10.3390/app11135792).
- **Type:** Empirical methodology — Spectral-Kurtosis-based blind feature extraction on NASA IMS.
- **Relevance:** **Direct + high signal** — explicitly uses NASA IMS dataset; same author (Siu Ki Ho) as paper #5 (Brunel MPhil thesis), this is essentially the journal-paper version.
- **Summary:** Pipeline: (1) **Spike subtraction** (median filter over fixed window) to remove outlier spikes that bias Spectral Kurtosis. (2) **Stationary Wavelet Transform** de-noising to lift SNR. (3) **Wiener filter constructed from Spectral Kurtosis estimates** to perform Blind Source Separation that isolates fault-relevant signals from healthy carrier signals. (4) Automated **change detection on SK time-series** as the early-fault alarm. Validated against the IMS published ground truth (Qiu et al. 2006). Table 1 explicitly contrasts the proposed pipeline against three earlier IMS-based methods (Qiu wavelet, Wang EMD, Yu HMM-DPCA) on six dimensions: de-noising, filtering, decomposition, prior knowledge required, automated defect detection. Only the proposed method has "Yes" across all four positive criteria including automated detection. Confirms Rexnord ZA-2115, 2000 RPM, 2721 kg radial load, PCB 353B33 accelerometers x+y per bearing, 20 kHz sampling, 20,480 samples/file, 10 min intervals.
- **Key findings / takeaways:**
  - **Spike subtraction with median filter is a useful preprocessing step** to remove non-fault outliers before SK estimation — a small but credible addition to my notebook 02 preprocessing.
  - The Qiu/Wang/Yu comparison table is exactly the right shape for a "Related work" comparison table in my Medium article.
  - "Detect defect pattern on day 27 — a week earlier than the final run-to-failure inspection day" is a quotable result from Qiu et al. 2006 for the Set 2 outer-race failure timeline.
- **Connection to corpus / mini-research:** Directly augments my notebook 02 preprocessing options (spike subtraction + SWT de-noising). The author overlap with paper #5 means together they give me one of the strongest "Spectral Kurtosis on NASA IMS" worked examples in the corpus.

---

### 26. Optimized Weights Spectrum Autocorrelation: A New and Promising Method for Fault Characteristic Frequency Identification for Rotating Machine Fault Diagnosis

- **Source / format / size:** Markdown 21.5 KB (substantial introduction + reference list).
- **Authors / venue / year:** Bingchang Hou, Xiao Feng, Jin-Zhen Kong, Zhike Peng, Kwok-Leung Tsui, Dong Wang. Shanghai Jiao Tong University + collaborators. *Mechanical Systems and Signal Processing* 2023, [10.1016/j.ymssp.2023.110200](https://www.sciencedirect.com/science/article/abs/pii/S0888327023001073). 28 citations as of harvesting.
- **Type:** Methodology — frequency-domain demodulation for fault frequency identification.
- **Relevance:** Adjacent — bearing + gearbox fault diagnosis, dataset not specifically NASA IMS in this paper but the technique is generic.
- **Summary:** Mainstream FCF identification via Hilbert-transform-based Squared Envelope Spectrum (HT-SES) and Spectral-Coherence-based SES (SC-SES) both demodulate signals from the time domain and depend critically on fault-signature extraction methods (fast kurtogram, blind deconvolution, VMD, etc.) — all of which are sensitive to random impulsive noise. The proposed **OWSAC** demodulates from the frequency domain instead: (1) start from a recently developed **Optimised Weights Spectrum (OWS)** — a convex-optimisation-based purified version of the Fourier spectrum that eliminates interference spectral lines, (2) apply an **adaptive threshold** to suppress noise spectral lines, (3) compute **autocorrelation of the resulting purified spectrum** — the autocorrelation peaks correspond to FCF and harmonics. Demonstrated superior to five baselines (HT-SES, SC-SES, fast-kurtogram-guided SES, optimised SES, plain Fourier autocorrelation) on two bearing/gearbox case studies. Importantly, OWSAC **does not need fault-signature extraction methods** as preprocessing — a major workflow simplification.
- **Key findings / takeaways:**
  - The classical FCF identification chain is: Hilbert envelope → FFT → look for BPFI/BPFO/BSF/FTF peaks. This paper proposes skipping the Hilbert envelope step by working directly on the spectrum's autocorrelation.
  - **Frequency-domain demodulation** is a recently developed angle — useful framing context for the Medium article's signal-processing section.
  - Related to paper #24 (Log-Envelope Sparsity) — both Wang-group SJTU papers, both proposing alternatives to traditional SES.
- **Connection to corpus / mini-research:** Adjacent. Probably not implementing. Worth knowing exists for "Related work" — OWSAC is a recent (2023) competitor to my plan of "do FFT + read off the four bearing fault frequencies."

---

### 27. Parametric Time-Domain Methods for the Identification of Vibrating Structures — A Critical Comparison and Assessment

- **Source / format / size:** Markdown 7.1 KB. Source: Elsevier *Mechanical Systems and Signal Processing* 15(6): 1031–1060, 2001.
- **Authors / venue / year:** S.D. Fassois et al. Greek authors (Patras school of modal analysis). 2001.
- **Type:** Review / comparative assessment of parametric system identification methods.
- **Relevance:** Adjacent — bearing diagnosis specifically is not the topic, but the AR(8) feature in paper #15's tutorial comes from this family.
- **Summary:** Compares four stochastic methods (PEM, 2SLS, LMS, IV) and three deterministic methods (LS, Prony, ERA) for parametric time-domain identification of vibrating structures from random excitation and noise-corrupted response signals. Monte-Carlo experiments on a 6-DOF structural model with closely spaced and highly damped modes show: stochastic methods (PEM, LMS, IV) win on noisy data; deterministic methods (Prony, ERA) suffice for low-noise cases; LS is the worst across the board. Specific findings: weak closely-spaced modes are hard to identify (impossible for 2SLS and deterministic); highly damped modes are also hard; false modes appear especially in LS and ERA; natural frequencies estimate better than damping ratios, which estimate better than mode shapes; user expertise is necessary.
- **Key findings / takeaways:**
  - **Auto-Regressive (AR) models** — the foundation of paper #15's AR(8) coefficient feature — fit into this paper's broader framework. AR models alone (e.g., least-squares fit) are at the weaker end; modal-analysis methods (PEM, LMS, IV) are more robust.
  - The damping-ratio versus natural-frequency accuracy hierarchy is a useful nuance: when paper #15 uses AR(8) coefficients, what's effectively being captured is more about poles (frequencies) than about residues (mode shapes).
- **Connection to corpus / mini-research:** Background reading for understanding the AR(8) feature in paper #15. Mostly out of scope for implementation — adding multiple parametric models is way beyond mini-research scope.

---

### 28. Physics-Informed Transfer Learning Scenarios for Structural Health Monitoring

- **Source / format / size:** `PHYSICS-INFORMED TRANSFER LEARNING SCENARIOS ... .pdf`, 23.5 MB. PhD dissertation, ~300 pages.
- **Authors / venue / year:** Trent S. Furlong. *Doctor of Philosophy* dissertation in Acoustics, The J. Jeffrey and Ann Marie Fox Graduate School, The Pennsylvania State University, December 2025. Advisor: Karl M. Reichard (Applied Research Lab). Committee includes Daniel C. Brown, Gregory A. Banyay, Daning Huang (Aerospace).
- **Type:** PhD dissertation — methodology + experiments on physics-informed transfer learning for SHM.
- **Relevance:** Adjacent (Structural Health Monitoring, not bearing-specific) but **enormously valuable as a structural reference** — this is what a recently-defended PhD dissertation in a closely-related domain looks like in 2025.
- **Summary:** The motivating problem: failure data for SHM is expensive to acquire because physical structures must be damaged to generate it. ML alone struggles with limited training data and generalisation. Physics-driven models (FEM) can simulate failure data but are computationally expensive and can have modelling error. The dissertation proposes **Physics-Informed Transfer Learning (PITL)** scenarios: use traditional + novel physics-driven models to generate synthetic data tailored for specific TL scenarios, use this synthetic data to train source models, then transfer to limited available experimental data. The dissertation's Chapter 3 enumerates the *transfer learning scenarios* applicable to SHM with high granularity: Sensor Transfer (sample rate, sensitivity, S/N, location, number, modality, drift, input features), Operating Condition Transfer (domain augmentation), Fault Transfer (localisation, extent/type, type augmentation/change), Structure Transfer (dissimilar structures, inter-structure with boundary or property transfer), Physics Transfer (homogeneous/inhomogeneous, coordinate-system).
- **Key findings / takeaways:**
  - **A PhD dissertation structural template directly applicable to my eventual PhD proposal**: 1 = Introduction (motivation, background, contributions, organisation); 2 = Methodology + Background literature; 3 = Methodology specifics enumerated as fine-grained scenarios; later chapters = experiments. This is the right structural reference for a Mining/Material PhD proposal.
  - **Transfer Learning is a credible PhD-level extension** of what mini-research builds. Could be a Mining-focused proposal angle: "transfer learning from a published bearing dataset (NASA IMS) to a mining-specific bearing dataset that doesn't yet exist."
  - The committee structure (advisor + 3 committee + adjunct affiliate + special signatory) is the US PhD model. Worth noting as a contrast point — the Australian PhD model (typically primary + 1-2 co-supervisors) is leaner.
- **Connection to corpus / mini-research:** **Most useful as a PhD-proposal template, not as a mini-research methodology source.** The PITL angle is years beyond what mini-research will demonstrate, but the very existence of a 2025 dissertation in this space confirms the area is publication-active.

---

### 29. Predictive Analytics in Logistics: Applications & Use Cases

- **Source / format / size:** Markdown 10.0 KB. Source: <https://www.transmetrics.ai/blog/predictive-analytics-in-logistics/> — a commercial blog post by Transmetrics, a logistics-AI company.
- **Authors / venue / year:** Not captured (blog post, no author).
- **Type:** **Commercial blog post** — domain framing, not academic content.
- **Relevance:** Context. Mentions "Predictive Maintenance" as one of many predictive-analytics use cases in logistics, but the body is mostly about supply-chain forecasting, last-mile delivery, transportation management systems, DHL/Maersk/UPS examples.
- **Summary:** Frames predictive analytics in logistics across six use cases: better supply chain visibility, forecasting, transportation management systems, **predictive maintenance** (one paragraph: "detect failure patterns and anomalies, learn from those patterns, then predict future failures of machine components so that they can be replaced before they fail"), last-mile delivery, sustainability. Includes a stat: "96% of 3PLs and 86% of shippers have migrated to the cloud while 80% of 3PLs and 77% of shippers are investing in predictive analytics."
- **Key findings / takeaways:**
  - Useful **non-academic framing language** for the Medium article's opening — "predictive maintenance sits within a broader predictive-analytics shift across logistics, supply chain, and manufacturing."
  - Quotable industry statistics for the Medium opener (CSCMP study).
  - Confirms that the predictive-maintenance framing is *also* used in logistics — useful when the cold-email Wave 1 supervisor is industry-embedded (Curtin WASM industry-funded HDR, MRIWA scholarship).
- **Connection to corpus / mini-research:** Pure framing input. Use one quote from this in the Medium intro. Nothing technical.

---

### 30. Preventive, Predictive, and Corrective Maintenance

- **Source / format / size:** Markdown 53.7 KB — substantial. Source: <https://fastercapital.com/topics/preventive,-predictive,-and-corrective-maintenance.html> — commercial content-marketing blog.
- **Authors / venue / year:** Not captured (no individual authorship visible).
- **Type:** **Commercial blog** aggregating multiple short sections about maintenance strategies.
- **Relevance:** Context. Defines the three maintenance archetypes (preventive = scheduled, predictive = data-driven, corrective = after-failure) at a beginner level.
- **Summary:** Defines preventive, predictive, and corrective maintenance with examples. Preventive = scheduled regardless of condition, best for critical assets. Predictive = data-driven, monitors equipment to detect issues before failure, best for assets at high failure risk. Corrective = repair after failure, best for non-critical replaceable assets. Then dozens of sections of essentially the same content repeated under different framings (Physical asset value, manufacturing plants, asset-aware HVAC examples, retail HVAC corrective example, etc.).
- **Key findings / takeaways:**
  - The **three-archetype frame is the right one-line opener for the Medium article**: "Maintenance strategies fall into three families: preventive (scheduled), corrective (after failure), and predictive (data-driven). This article is about the predictive family — specifically using vibration signals to forecast bearing failure."
  - The "predictive maintenance for airlines watching engines" example in section 1 is a useful concrete framing — even though my application is to mining/mineral processing.
- **Connection to corpus / mini-research:** Use ~1 paragraph for the Medium article's introductory framing. Then immediately move on to the academic content.

---

### 31. Review of Fault Detection and Diagnosis Techniques for AC Motor Drives

- **Source / format / size:** `Review of Fault Detection ... .pdf`, 1.3 MB, 22 pages.
- **Authors / venue / year:** Muhammed Ali Gultekin, Ali Bazzi. Electrical and Computer Engineering Department, University of Connecticut. *MDPI Energies* 16(15): 5602, 25 July 2023, [10.3390/en16155602](https://doi.org/10.3390/en16155602).
- **Type:** **Survey / review article** — categorises FDD methods for AC motor drives.
- **Relevance:** Adjacent (AC motor drives ⊃ induction motors with bearings; bearings are one of four fault categories covered).
- **Summary:** Categorises faults in AC motor drives into four types: **machine faults** (bearing, stator, rotor), **power electronics faults** (IGBT, gate driver), **DC link capacitor faults**, **sensor faults**. For each, categorises FDD methods into statistical, ML-based, and DL-based. Notes: **bearing faults are the most common machine fault — ~30% of failures**. Bearings, broken rotor bars, and stator winding shorts together account for >75% of motor faults (EPRI + IEEE studies, consistent with paper #16 Thorsen & Dalva). Section 2.2.1 confirms the bearing-failure narrative: "When a surface defect occurs on a bearing element, it causes periodic impact forces that can be detected through the vibration signal during operation. By analyzing the frequency components of the machine vibration, faults in the bearing components can be identified."
- **Key findings / takeaways:**
  - **"Bearings account for ~30% of motor failures"** — a quotable statistic for the Medium opener (consistent with the 40% figure cited in paper #16's discussion).
  - **EPRI + IEEE motor reliability studies** are the canonical industrial-statistics references.
  - Section 2.2.1 is a tight 1-paragraph description of why vibration signatures encode bearing faults — useful template language for the Medium article's "background" section.
  - Open-source data repositories table (Table 4 mentioned in text but not in pages I read) likely lists NASA IMS, CWRU, Paderborn, FEMTO, MFPT — the canonical five public datasets.
- **Connection to corpus / mini-research:** Useful survey-level reference for "Related work" in the Medium article. Provides the industrial-statistics quotables and a clean four-category taxonomy of motor drive faults that I can reference when introducing the bearing-fault sub-problem.

---

### 32. Revisiting Rosenblatt's Perceptron: Robust High-Entropy Classification via Uncertainty Margins

- **Source / format / size:** `Revisiting Rosenblatt's Perceptron ... .pdf`, 0.3 MB, ~12 pages (preprint).
- **Authors / venue / year:** Dylan Sutton Chavez. Independent Researcher, Mexico City. Preprint, SSRN abstract 6804468. Not yet peer-reviewed.
- **Type:** Methodology — modifies Rosenblatt's perceptron with uncertainty margins, validated on three datasets including NASA IMS.
- **Relevance:** Direct — explicitly validates on **NASA IMS Bearing Dataset** (binary healthy/faulty classification).
- **Summary:** Proposes a modified linear perceptron with a **geometric abstention region [−ε, ε]** around the decision hyperplane. The ε parameter is **derived from statistical properties (empirical standard deviation σ)** estimated during training and calibration; remains fixed during inference. This is positioned as a low-resource alternative to deep models (LSTMs, transformers) for embedded edge applications. Three test datasets: (1) nonstationary financial time series (TSLA, GOOGL, LLY, AAPL), (2) SemEval sentiment analysis, (3) **NASA IMS Bearing Dataset binary classification** (healthy=1, faulty=0). For NASA IMS, **low-dimensional input vector d=14 of root-mean square (RMS), harmonic oscillators (sin, cos), kurtosis, crest factor, and skewness over sliding windows**. Reports inference memory footprint ≈1 KB and orders of magnitude lower computational overhead than RNNs. Achieves >90% precision at reduced coverage. Selective prediction = the model can refuse to predict if confidence is low (Δ parameter trades coverage for accuracy: Δ=0.5 → 38% coverage at 82% accuracy; Δ=2.0 → 9% coverage at 99% accuracy).
- **Key findings / takeaways:**
  - **Concrete feature set for NASA IMS binary classification: d=14 = RMS + sin/cos + kurtosis + crest factor + skewness over sliding windows.** This is a useful "minimal feature set" reference — much smaller than the multi-domain feature sets in paper #15 or paper #18.
  - **Selective prediction = abstain on low-confidence samples** is a credible deployment idea worth mentioning in the Medium Discussion section.
  - Confirms the binary classification framing works on NASA IMS with very simple models if features are right. Useful counter-point to deep-learning-on-everything.
- **Connection to corpus / mini-research:** Direct methodology reference for a minimal-feature baseline. The d=14 features are easy to implement and benchmark — could be a "lightweight baseline" alongside RF/XGBoost. The selective-prediction idea is a Discussion-section angle for the Medium article.

---

### 33. SJER-vol2-no2-doc2.pdf

- **Source / format / size:** `SJER-vol2-no2-doc2.pdf`, 3.8 MB, ~16 pages.
- **Type:** **DUPLICATE.** This is the identical file to paper #18 — *Federated Temporal Graph Learning for Weakly Supervised Bearing Anomaly Detection*, Darlami & Awasthi, *Scientific Journal of Engineering Research* 2(2), Feb 2026. The filename `SJER-vol2-no2-doc2.pdf` is the SJER's own publishing identifier (Scientific Journal of Engineering Research, Vol 2, No 2, Doc 2).
- **Action item:** **Rename or delete** one of the two copies to avoid double-counting. Recommend keeping the more descriptive filename (`Federated Temporal Graph Learning ... .pdf`) and removing `SJER-vol2-no2-doc2.pdf`.
- **Summary / takeaways:** See paper **#18 above** — no new content.

---

### 34. The Application of High-Resolution Spectral Analysis for Identifying Multiple Combined Faults in Induction Motors

- **Source / format / size:** Markdown 8.5 KB. Source: IEEE `Transactions on Industrial Electronics` 58(5): 2002–2010, 2011. DOI [10.1109/TIE.2010.2051390](https://ieeexplore.ieee.org/document/5747206).
- **Authors / venue / year:** Garcia-Perez et al. 2011.
- **Type:** Methodology — high-resolution spectral analysis for combined fault diagnosis.
- **Relevance:** Adjacent — induction motors, not specifically bearing or NASA IMS, but bearings are one of the relevant fault classes.
- **Summary:** Motivates the **multiple-combined-faults** problem in induction motors — real-world failures rarely come singly, and combined faults excite vibration/current at fault frequencies that are linearly or nonlinearly combined with each other. Most diagnostic literature handles single faults only. The proposed method combines a **finite impulse response filter bank with high-resolution spectral analysis based on Multiple Signal Classification (MUSIC)** to accurately identify the frequency-related faults. Validated on induction motor data; reports good performance on multi-fault detection when fault frequencies are close to the analytically expected values.
- **Key findings / takeaways:**
  - **Multiple-fault detection** is a credible research-gap angle for a PhD proposal — most NASA IMS work treats faults as single (B1 outer race, B3 inner race, B4 ball). Real-world deployment scenarios in mining could see multiple bearings degrading simultaneously.
  - **MUSIC algorithm** is a frequency-domain super-resolution method that beats FFT resolution in noisy conditions — useful background tool to know about.
- **Connection to corpus / mini-research:** Adjacent. Pure citation reference. The multi-fault framing could be a Discussion-section angle in the Medium article.

---

### 35. The LESGIRgram: A New Method to Select the Optimal Demodulation Frequency Band for Rolling Bearing Faults

- **Source / format / size:** `The LESGIRgram ... .pdf`, 7.5 MB, ~20 pages.
- **Authors / venue / year:** Tian Tian, Guiji Tang, Xiaolong Wang. School of Energy, Power and Mechanical Engineering, North China Electric Power University, Baoding 071000, China. *MDPI Machines* 11(12): 1052, 27 Nov 2023, [10.3390/machines11121052](https://doi.org/10.3390/machines11121052).
- **Type:** Methodology — yet another *-gram for optimal demodulation band selection.
- **Relevance:** Direct — explicitly validated on **NASA IMS bearing dataset** (Section 4).
- **Summary:** Adds another entry to the *-gram family for selecting the optimal demodulation frequency band (ODFB): the **LESGIRgram** = ratio of Logarithmic Envelope Spectrum Gini Indices (LESGI) between fault and healthy baseline signals. Background reviews the predecessors: **Fast Kurtogram (FK)** uses spectral kurtosis on a 1/3-binary tree of frequency bands; **Autogram** uses unbiased autocorrelation kurtosis; **SKRgram** uses the ratio of fault-to-healthy SK matrices; **Infogram**, **Fast Entrogram**, **Hoyergram**, etc., progressively swap the band-screening metric. The LESGIRgram's contribution: use **Gini index of the Logarithmic Envelope Spectrum** as the screening metric, computed as a ratio against a healthy baseline. The ratio "LESGI_measured / LESGI_baseline" amplifies the difference, locating bands that contain the most fault information. Validated on simulated signals and real vibration signals from Jiangnan University + **NASA IMS bearing dataset**.
- **Key findings / takeaways:**
  - **The *-gram literature converges to "ratio against healthy baseline"** — a robust pattern: use a healthy reference signal to compute a relative-purity metric per band, then pick the band that's most different from healthy.
  - **Log Envelope Spectrum (LESGI)** is sibling to paper #24's Log-Envelope Sparsity Measures — both apply log transform to the envelope before computing a sparsity-like statistic.
  - Section 2's review of FK/Autogram/SKRgram/Infogram/Fast Entrogram/Hoyergram/Power Spectrum Screening Combination-gram is essentially **a survey of all the band-selection methods my mini-research isn't using** — useful one-paragraph "what we're not doing" framing in the Medium article.
- **Connection to corpus / mini-research:** Direct on NASA IMS. Probably not implementing LESGIRgram, but the Section-2 survey gives me clean Related-Work language for the Medium article: "while there is a rich literature on optimal-demodulation-band selection (FK, Autogram, SKRgram, Infogram, LESGIRgram), this work treats the FFT band features as input to a downstream classifier rather than as the diagnostic itself."

---

### 36. Time-Domain Signal Analysis Using Adaptive Notch Filter

- **Source / format / size:** Markdown 4.6 KB (abstract + reference list).
- **Authors / venue / year:** Mojiri, Karimi-Ghartemani, Bakhshai. *IEEE Transactions on Signal Processing* 55(1): 85–93, Jan 2007, [10.1109/TSP.2006.885686](https://ieeexplore.ieee.org/document/4034203).
- **Type:** Methodology — adaptive notch filter for sinusoidal-component extraction.
- **Relevance:** Adjacent. Not bearing-specific. Cited by paper #12 (Deep learning under imbalanced data) — likely as a baseline preprocessing tool.
- **Summary:** Proposes an **Adaptive Notch Filter (ANF)** to extract a single sinusoid of possibly time-varying frequency from a noise-corrupted signal. Extends to a chain of filters that can estimate the fundamental frequency of a multi-harmonic signal and decompose it into constituent sinusoids. Order = 2n+1 where n = number of sinusoids. Stability analysis via local averaging theory under slow-adaptation assumption. Provides **instantaneous values** of constituent components (unlike Fourier analysis which is windowed). Adaptive to the fundamental frequency.
- **Key findings / takeaways:**
  - **ANF as an alternative to FFT for time-varying signals.** Useful when the rotational speed isn't constant (which NASA IMS *is* at 2000 RPM, but real industrial bearings often aren't).
  - The "instantaneous values" property is a useful framing distinction — FFT gives you a frequency-domain snapshot averaged over a window; ANF gives you time-resolved per-frequency component.
- **Connection to corpus / mini-research:** Background reference. Not implementing.

---

### 37. Understanding Importance of Positive and Negative Signs of Optimized Weights Used in the Sum of Weighted Normalized Fourier Spectrum / Envelope Spectrum for Machine Condition Monitoring

- **Source / format / size:** Subfolder `Understanding importance of positive and negative signs.../` containing single MD `envelope spectrum for machine condition monitoring.md`. Source: Elsevier *Mechanical Systems and Signal Processing*, [10.1016/j.ymssp.2022.109605](https://www.sciencedirect.com/science/article/abs/pii/S088832702200262X). 69 citations.
- **Authors / venue / year:** D. Wang group at SJTU (same authors as papers #24, #26 — Bingchang Hou, Dong Wang). Earlier paper that #26 builds on.
- **Type:** **Short communication** — extends Wang's "sum of weighted normalised square envelope" framework with new propositions about sign-aware weights.
- **Relevance:** Direct — uses **NASA IMS bearing dataset** (Section 4's second illustrative example, 984 files = Set 2).
- **Summary:** In Wang's prior work, the sum of weighted normalised square envelope was proposed as a generalised framework that subsumes kurtosis, negative entropy, smoothness index, and Gini index (all are special cases with different weights). A convex-optimisation problem can be solved to **automatically design weights data-driven-ly**. Earlier observation: solving for the optimal weights on the Fourier or envelope spectrum produces informative frequency bands / fault characteristic frequencies — but only with positive weights. This short communication **revisits the sign question** and mathematically proves that BOTH positive AND negative optimised weights are important: positive weights flag fault-generated frequency components, negative weights distinguish fundamental frequency components (shaft rotational harmonics, etc.). Three new propositions are proposed and verified on a gearbox dataset + NASA IMS bearing dataset.
- **Key findings / takeaways:**
  - **Bearing fault diagnosis = separating fundamental frequency components (shaft rotation, 1X/2X/3X) from fault-generated frequency components (BPFI, BPFO, BSF, FTF and harmonics).** The optimised-weights framework can do both, with sign distinguishing the role of each frequency component.
  - The framework is positioned as the foundation for **"physics-informed machine learning"** for machine condition monitoring — connects to paper #28 (PITL dissertation) and paper #2 (KU Leuven domain-knowledge AE).
  - **Key citation in the "Cited by" list**: "an incipient bearing fault happened around the 535th file" of NASA IMS Set 2 — i.e., file 535 of 984 is the documented incipient-fault inception in Set 2. Useful for my labelling strategy.
- **Connection to corpus / mini-research:** **Quotable result**: NASA IMS Set 2 incipient fault inception around file 535 of 984 — useful labelling anchor. The sum-of-weighted-normalised framework is the unifying theoretical scaffold for many of the corpus's sparsity-measure papers.

---

### 38. Unsupervised Machinery Health Indicator Construction Based on a Temporal Adversarial Neural Wavelet Model

- **Source / format / size:** Markdown 2.6 KB (abstract only). Source: IOP `Measurement Science and Technology`, accepted manuscript 13 May 2026, [10.1088/1361-6501/ae6d28](https://iopscience.iop.org/article/10.1088/1361-6501/ae6d28/pdf).
- **Authors / venue / year:** Yujun Zhou, Tangbin Xia, Yuhui Xu, Wujun Si, Dong Wang, Lifeng Xi. SJTU. 2026 (recent).
- **Type:** Methodology — unsupervised HI via temporal adversarial neural wavelet model.
- **Relevance:** Direct (bearing-prognostic HI; industrial dataset validation).
- **Summary:** Yet another unsupervised deep-learning health indicator. Pipeline: (1) **Neural wavelet decomposition** with *learnable* wavelet bases performs multi-resolution sub-band feature extraction; (2) Trained with a **temporal adversarial learning scheme** that suppresses the confounding influence of operating time and enhances the intrinsic correlation between extracted features and physical degradation behavior; (3) Each sub-band's features are characterised by an **alpha-stable distribution**; (4) A **dual-domain HI** combines the probability density and the distributional characteristic of each sub-band; (5) An **adaptive correlation-aware fusion mechanism** integrates the sub-band HIs into a unified HI. Validated on multiple industrial datasets.
- **Key findings / takeaways:**
  - **Learnable wavelet bases** = a recent twist; replaces fixed-basis wavelet decomposition with a trainable filterbank that adapts to the dataset.
  - **Alpha-stable distribution** = generalises Gaussian to heavy-tailed distributions; useful for impulsive bearing-fault statistics that violate Gaussian assumptions.
  - **Temporal adversarial training** = a key trick for separating "what's degrading" from "how long the machine has been running" — addresses the trivial-confounder problem in run-to-failure HI design.
- **Connection to corpus / mini-research:** Direct sibling to papers #2, #6, #8 (the unsupervised HI cluster). Together they're the strongest signal for the proposal's "research gap" angle: unsupervised HI construction with physical interpretability.

---

### 39. Untangling and Incremental Methods for Consistent, Low-Latency Embedded Singular Value Decomposition

- **Source / format / size:** `Untangling and Incremental Methods ... .pdf`, 0.2 MB, ~52 pages. M.S. thesis (preview/draft).
- **Authors / venue / year:** Brian Shimabukuro. Master of Science in Electrical and Computer Engineering, University of California, Davis, 2025. Committee: Zhi Ding (Chair), Bernard Levy, Zhaojun Bai.
- **Type:** **M.S. thesis** — embedded-systems engineering, not bearing-specific.
- **Relevance:** Context. Off-topic for bearing fault diagnosis directly, but two of the application domains mentioned are **joint accelerometer suites + vibration arrays** — meaning the SVD pipelines proposed here could be applied to bearing vibration analysis in embedded sensors.
- **Summary:** Embedded sensing systems (MCUs, FPGAs, ASICs) require SVD-grade signal processing for denoising, compression, and subspace tracking under tight latency, power, and memory budgets. Two problems: (1) **Recompute cost** — running a full SVD every time window doesn't scale. (2) **Tangled sample paths** — between successive SVDs, sign/permutation ambiguity causes temporal incoherence that destabilises downstream logic. The thesis contributes: a lightweight fixed-point **untangler** with bounded latency (greedy max-diagonal correlation assignment with sign/phase canonicalisation), fixed-point **incremental Brand-style iSVD**, rank-thresholding policies for reduced-rank maintenance, drift-aware forgetting strategy without explicit downdates. MATLAB → C++ → Vitis flow for FPGA. Approach evaluated on synthetic and representative sensor streams.
- **Key findings / takeaways:**
  - The "tangled sample path" problem is real — repeated SVDs on overlapping time windows produce sign-ambiguous orthonormal bases that need to be aligned for downstream models to use them coherently.
  - **Embedded SVD = a niche but real research area** that connects to PCA, low-rank approximations, subspace tracking — all of which appear in bearing-fault literature (paper #10 uses MDS, papers cite PCA for feature reduction).
  - This thesis is a structural model for "applied embedded systems master's thesis" — useful template style if I ever wanted to write one.
- **Connection to corpus / mini-research:** Out of scope for mini-research. Tangentially relevant — if I ever wanted to deploy a bearing-fault model to a sensor microcontroller (Liu CFD-ML or industrial PdM deployment), this kind of embedded-SVD primitive is in the stack.

---

### 40. NASA Prognostics Data Repository — Data Sets Catalog (dataset.md)

- **Source / format / size:** Markdown 37.9 KB. Source: NASA Prognostics Data Repository catalog page.
- **Authors / venue / year:** NASA Prognostics Center of Excellence + Prognostics & Health Management Society. Updated continuously.
- **Type:** **Catalog / metadata listing** — not a research paper.
- **Relevance:** Direct — describes 21 NASA prognostic datasets including **#4 Bearings (the IMS dataset I'm using)**.
- **Summary:** Lists 21 publicly available NASA Prognostics Data Repository datasets with download URLs and citation strings. Includes: (1) Algae Raceway, (2) CFRP Composites, (3) Milling, **(4) Bearings — IMS Univ Cincinnati, the dataset I'm using**, (5) Batteries, (6) Turbofan Engine Degradation (C-MAPSS), (7) PHM08 Challenge, (8) IGBT Accelerated Aging, (9) Trebuchet, (10) FEMTO Bearing (Bearing dataset #2 = PRONOSTIA), (11) Randomized Battery Usage, (12) Capacitor Electrical Stress, (13) MOSFET Thermal Overstress Aging, (14) Capacitor Electrical Stress-2, (15) HIRF Battery, (16) Small Satellite Power Simulation, (17) Turbofan Engine Degradation Simulation-2, (18) Fatigue Crack Growth in Aluminum Lap Joint, (19) CNC Milling Machine, (20) Anemometer, (21) Accelerated Battery Life Testing.
- **Key findings / takeaways:**
  - **Citation string for IMS Bearings** (item #4): *"J. Lee, H. Qiu, G. Yu, J. Lin, and Rexnord Technical Services (2007). IMS, University of Cincinnati. 'Bearing Data Set', NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA."* — use this verbatim in the Medium article and GitHub repo README.
  - **Reference paper for IMS Bearings**: Hai Qiu, Jay Lee, Jing Lin (2006), "Wavelet Filter-based Weak Signature Detection Method and its Application on Roller Bearing Prognostics," *Journal of Sound and Vibration* 289: 1066–1090. — this is the foundational paper that every NASA IMS publication cites.
  - **The FEMTO Bearing dataset (#10) is a credible second dataset** for cross-validation: it's the PRONOSTIA platform used in paper #6 (Khamoudj) alongside NASA IMS. If mini-research wants a "does our pipeline generalise?" Discussion-section angle, FEMTO is the natural extension.
  - **C-MAPSS Turbofan (#6, #17)** is the most famous NASA prognostic dataset overall — the RUL-prediction benchmark used in tens of thousands of papers. Not bearing-specific but a sibling dataset family.
- **Connection to corpus / mini-research:** **The authoritative citation reference** for the Medium article's "Dataset" section. The Qiu/Lee/Lin 2006 reference paper is the single most important citation I'll use. Also useful as a "what other datasets could extend this work" pointer (FEMTO is the answer).

---

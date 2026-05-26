Unsupervised machinery health indicator construction based on a temporal adversarial neural wavelet model

https://iopscience.iop.org/article/10.1088/1361-6501/ae6d28/pdf

Unsupervised machinery health indicator construction based on a temporal adversarial neural wavelet model
Yujun Zhou, Tangbin Xia, Yuhui Xu, Wujun Si, Dong Wang and Lifeng Xi

Accepted Manuscript online 13 May 2026 • © 2026 IOP Publishing Ltd. All rights, including for text and data mining, AI training, and similar technologies, are reserved.

What is an Accepted Manuscript?

DOI 10.1088/1361-6501/ae6d28
Authors
Article metrics
4 Total downloads

Submit
Submit to this Journal
Permissions
Get permission to re-use this article

Share this article
Article information
Abstract
Unsupervised deep learning-based (DL-based) health indicator (HI) exhibits broad application potential in machinery condition monitoring of long-lifespan and high-value equipment, as it enables degradation-discriminative feature analysis without requiring end-of-life data. However, the evolution patterns of discriminative features extracted by traditional DL workflows still lack physical consistency with actual degradation behavior, which limits the credibility and performance of the constructed HI. For this issue, this study proposes a novel HI construction model that extracts discriminative features to appropriately represent the physical degradation behavior and constructs an HI endowed with an explicit statistical expression. Firstly, a neural wavelet decomposition framework performs multi-resolution sub-band feature extraction on degradation signals using learnable wavelet bases. It is trained with a designed temporal adversarial learning scheme to suppress the confounding influence of operating time and enhance the intrinsic correlation between degradation-discriminative features and the actual physical degradation behavior. Subsequently, each sub-band is independently characterized using the alpha-stable distribution. A dual-domain HI construction strategy, which measures the probability density and the distributional characteristic, constructs independent HIs for each sub-band. It provides explicit statistical analysis of degradation-discriminative features’ evolution based on the theoretical foundation of fault characteristic representation in signal morphology. Finally, an adaptive correlation-aware fusion mechanism integrates these sub-band HIs into a unified HI. Validation results on multiple industrial datasets demonstrate the superior performance of the proposed HI.
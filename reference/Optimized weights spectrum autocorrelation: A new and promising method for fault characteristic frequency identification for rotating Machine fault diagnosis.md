Optimized weights spectrum autocorrelation: A new and promising method for fault characteristic frequency identification for rotating Machine fault diagnosis


https://www.sciencedirect.com/science/article/abs/pii/S0888327023001073

Optimized weights spectrum autocorrelation: A new and promising method for fault characteristic frequency identification for rotating Machine fault diagnosis

Bingchang Hou a b
, 
Xiao Feng a b
, 
Jin-Zhen Kong a b
, Zhike Peng c
, Kwok-Leung Tsui d
, Dong Wang a b

Show more

Cite

Add to Mendeley

Share
10.1016/j.ymssp.2023.110200
Purchase PDF
Article preview
Recommended articles
Cited by (28)
Metrics
Article preview
Abstract
Introduction
Section snippets
References (38)
Highlights
•
Optimized weights spectrum autocorrelation (OWSAC) is newly proposed to identify fault characteristic frequencies.
•
OWSAC demodulates signals from the frequency domain.
•
OWSAC is autocorrelation of a purified spectrum in which interferential and noise spectral lines are automatically eliminated.
•
Two case studies demonstrate the superiority of the proposed OWSAC to five existing methods.
Abstract
Since fault characteristic frequencies (FCFs) and their harmonics are closely connected with specific fault types of rotating machines, identification of FCFs and their harmonics is a very crucial step for signal processing-based rotating machine fault diagnosis. Nowadays, Hilbert transform (HT) based square envelope spectrum (SES) and spectral coherence (SC) based SES are two main tools for FCF identification. Since the HT demodulates signals by calculating envelope signals in the time domain and the SC is based on the temporal instantaneous autocorrelation function of demodulated signals, the temporal waveform of a demodulated signal must be purified by using fault signature extraction methods, e.g., fast kurtogram, blind deconvolution, and their variants, etc. However, it has been extensively reported that these fault signature extraction methods are prone to be affected by interference components such as impulsive noise. Unlike the HT and SC which demodulate from the time domain, this paper aims to demodulate signals from the frequency domain to identify FCFs and their harmonics for rotating machine fault diagnosis. Firstly, it is demonstrated that Fourier spectrum autocorrelation is possible to indicate FCFs and their harmonics. Nevertheless, a troublesome problem is that interference spectral lines and noise spectral lines of real-world signals in the frequency domain will severely affect the performance of the Fourier spectrum autocorrelation. To solve such problem, this paper introduces a recently developed optimized weights spectrum (OWS) and innovatively designs an adaptive threshold method to respectively eliminate an influence of interference spectral lines and noise spectral lines. Thus, a new FCF identification method named optimized weights spectrum autocorrelation (OWSAC) is accordingly proposed. One main merit of the proposed OWSAC is that it does not need any fault signature extraction methods to do signal preprocessing. Two experimental case studies respectively on incipient bearing and gearbox diagnosis validate the effectiveness of the proposed OWSAC. The proposed OWSAC can achieve satisfactory performance respectively in two case studies and it is superior to five methods including HT-based SES, SC-based SES, optimized SES, fast kurtogram guided SES, and Fourier spectrum autocorrelation.
Keywords
Fault characteristic frequency; Fault diagnosis; Optimized weights spectrum; Autocorrelation; Envelope spectrum analysis; Spectral coherence; Fast kurtogram
Previous article in this issue
Next article in this issue
Introduction
Machine fault diagnosis [1] aims to diagnose machine fault types for subsequent maintenances, which is beneficial to preventing unexpected accidents and gaining more profits. Because rotating parts such as bearings and gears [2], [3] are used to transmit force and moment, they are prone to have faults [4]. Thus, fault diagnosis of these rotating parts has received much attention [5]. Existing rotating machine fault diagnosis methods can be mainly classified into signal processing-based and machine learning-based methods. Compared with machine learning-based methods which usually lack interpretable fault features, signal processing-based methods are more reliable because these methods can ascertain fault types by identifying fault characteristic frequencies (FCFs), which are fully interpretable fault features. Here, an FCF refers to an emergence frequency of a kind of repetitive fault transients. Since different fault types have different and unique FCFs, identification of FCFs and their harmonics is a very promising rotating machine fault diagnosis approach.
Up to now, Hilbert transform [6] (HT) based envelope demodulation is a mainstream method for FCF identification. By successively using HT and Fourier transform (FT), a vibration fault signal can be transformed into an envelope spectrum or a square envelope spectrum (SES), from which FCFs and their harmonics can be identified. Because the HT is used to directly demodulate vibration signals from the time domain by obtaining an envelope/square envelope signal, a successful application of SES for FCF identification heavily relies on fault signature extraction methods which recover the waveform of repetitive fault transients. Many fault signature extraction methods including fast kurtogram [7], blind deconvolution [8], variational mode decomposition [9], and their variants [10], [11], [12], [13], [14], [15], [16] are intrinsically blind bandpass filter-based methods, which are guided by statistical parameters such as kurtosis, correlated kurtosis, Gini index, and entropy defined in the time domain or the frequency domain. Informative frequency bands (IFBs) are actually searching objects of these methods because repetitive fault transients usually exist in specific narrow frequency bands [17]. However, it has been extensively reported that these methods might be affected by random impulsive noise or low-frequency components [8], [10], [18]. To enhance the SES, Hou et al. [19] proposed an optimized SES calculated by a convex optimization model of the SES. The optimized SES is interpretable optimized weights [20] based on the sum of weighted normalized square envelope spectrum (SWNSES) [21]. It was validated in some case studies that clean FCFs can be observed on the optimized SES compared with an original HT-based SES. However, since the optimized SES is based on the SES, its performance will be constrained by the SES, which means that it had better purify analyzed vibration signals before SESs are calculated as input samples for the convex optimization model.
Since repetitive fault transients are a kind of cyclostationary signal, spectral correlation [22], which is defined as a two-dimensional Fourier transform of the instantaneous autocorrelation function of a signal, can be used to analyze repetitive fault transients. The spectral correlation is a bi-spectral map of spectral frequency and cyclic frequency. Here, the spectral frequency has the meaning of the frequency of the Fourier spectrum of a signal, and the cyclic frequency can be used to identify FCFs and their harmonics for rotating machine fault diagnosis. Moreover, normalized spectral correlation, whose value is rescaled to a range of [0, 1], was called spectral coherence (SC) [23]. Because a direct computation cost of the spectral correlation is expensive, Antoni et al. [24] proposed a fast spectral correlation and its normalized version named fast SC. Based on the short-time Fourier transform, the fast SC is a substantial improvement in decreasing computation costs for practical applications. Further, Borghesani and Antoni [25] designed a faster SC algorithm, which is 2–3 orders of magnitude faster than the fast SC in the same case studies. Since the SC is a bi-spectral tool and complicated to directly analyze and interpret, it is suggested to integrate the SC from the spectral frequency axis to obtain a new spectrum for analysis [26]. Thanks to deep investigations given by Randall et al. [26], it was proved that the new spectrum is equivalent to SES calculated by HT-based envelope demodulation. Therefore, the new spectrum is called SC-based SES. To enhance the SC-based SES for FCF identification, Antoni et al. [24] recommended integrating SC in a specific spectral IFB rather than a whole spectral frequency axis to generate an enhanced envelope spectrum (EES). Here, a selection of IFBs is equivalent to using fault signature extraction methods, i.e., if a vibration signal is purified before SC calculation, the selection of IFBs for enhancing SC based SES may not be needed.
In a word, HT-based SES and SC-based SES are two equivalent methods for FCF identification, while the former is more computationally convenient. Moreover, it is noteworthy that these two methods heavily rely on correct extraction of repetitive fault transients. Teager Kaiser Energy Operator (TKEO) [27], which was originally established for speech analysis [28], has been used to identify FCFs of rotating machine faults during recent years [29], [30], [31], [32]. A main merit of the TKEO is its real-time processing because it can be calculated from three adjacent sample points [33]. However, Randall and Smith [33] pointed out that real-time processing is not necessary for machine fault diagnosis and will cause a phase distortion problem because of the use of causal filters. Moreover, the TKEO was originally defined on a signal dominated by a mono-component carrier frequency, which was not applied to machine signals for the existence of multiple harmonics of shaft speeds. Thanks to a great effort made by Randall and Smith [33], now it is clarified that there is a misuse of the TKEO for FCF identification, and many claimed merits of TKEO for rotating machine fault diagnosis are false.
Based on the above literature review, it can be known that the HT-based SES and SC-based SES are two equivalent methods for FCF identification, and TKEO for FCF identification is inferior to the former two methods. However, the performance of the two former methods severely depends on correct applications of fault signature extraction methods. Hence, it is necessary to explore new excellent FCF identification methods which are less dependent on fault extraction methods. In fact, both the HT and SC demodulate signals directly from the time domain (i.e., the HT-based envelope demodulation obtains a demodulated envelope signal by calculating the absolute value of an analytic temporal signal, and the SC is based on the instantaneous autocorrelation function of a temporal signal). Different from the classic HT and SC, this paper attempts to demodulate vibration signals from the frequency domain rather than the time domain for FCF identification. Thus, a new method named optimized weights spectrum autocorrelation (OWSAC) is proposed to identify FCFs for rotating machine fault diagnosis. The core of the proposed OWSAC is the optimized weights spectrum (OWS) [20], which is based on the sum of weighted normalized Fourier spectrum (SWNFS). The main contributions of this paper are summarized as follows.
(1) It is demonstrated that intervals of main spectral lines of ideal repetitive fault transients are equal to their corresponding FCFs. Thus, it is possible to calculate a spectrum autocorrelation to identify FCFs.
(2) To respectively eliminate negative influence of interference spectral lines and noisy spectral lines, a recently developed OWS is used as a replacement of Fourier spectra of real-world vibration signals, and an adaptive threshold method is designed to subsequently preprocess the OWS. Therefore, a new method named OWSAC, which is autocorrelation of a preprocessed OWS, is proposed to identify FCFs and their harmonics for rotating machine fault diagnosis. The proposed OWSAC owns a solid mathematical basis and does not need fault signature extraction methods to purify a signal before its implementation.
(3) Two case studies respectively on bearings and gear fault diagnosis have verified the effectiveness of the proposed OWSAC. Compared with five methods including HT-based SES, SC-based SES, fast kurtogram guided SES, optimized SES, and direct Fourier spectrum autocorrelation, the proposed OWSAC can achieve the best performance in these two case studies. This paper demonstrates that designing new methods by demodulating signals from the frequency domain is feasible for FCF identification.
The rest of this paper is organized as follows. Section 2 theoretically and experimentally investigates the feasibility and difficult points of identifying FCFs by analyzing the Fourier spectrum of a fault signal. Section 3 firstly introduces the OWS, which is utilized as a replacement for the Fourier spectrum of a raw faulty signal; then, a novel adaptive threshold method for eliminating noise spectral lines in an OWS is detailed; finally, the technical route of the proposed OWSAC is elaborated. Two case studies and some related discussions on experimental results are presented in Section 4. Conclusions are summarized in Section 5.
Organizational access
Get full-text access by signing in with your organisation
Other access options
Purchase PDF
Need help with access?
Section snippets
Investigations on Fourier spectrum autocorrelation for FCF identification
Once a gear or bearing has a localized fault, its faulty area will strike other parts and generate an exponential decaying impulsive transient, which can be modeled as 
, where 
 and 
 is a time variable; 
 and 
 is a decaying parameter; 
 is an oscillation frequency. With the repetitive striking happening, the impulsive transient will cyclically emerge, so repetitive fault transients are introduced in vibration signals. Assume an approximate repetitive period of the
Optimized weights spectrum autocorrelation for FCF identification
This section will firstly introduce a recently developed OWS, which can be used to replace the Fourier spectrum of a raw signal for eliminating interference spectral lines. Further, an adaptive threshold method is designed to process the OWS for removing noise spectral lines. Finally, a technical route of the proposed OWSACC is given to identify FCFs.
First case study on a NASA bearing run-to-failure dataset
Firstly, a public NASA IMS bearing run-to-failure dataset [35] is used to validate the effectiveness of the proposed method. The test rig is shown in Fig. 6. The dataset contained 984 signal files, and each file was collected under a sampling frequency of 20 kHz and a length of 20,480 points. A bearing outer race fault whose FCF was 236.4 Hz was ascertained at the end of the experiment. A previous study [38] has verified that the bearing incipient fault time was around file 533.
This dataset is
Conclusions
In many signal processing-based rotating machine fault diagnosis methods, FCF identification is the most important step because FCFs and their harmonics are connected with specific fault types of rotating machines. Existing HT-based SES and SC-based SES are two equivalent methods for FCF identification, but their good performance heavily relies on a correct implementation of fault feature extraction methods. Considering that both the HT and SC demodulate signals from the time domain, this paper 
CRediT authorship contribution statement
Bingchang Hou: Conceptualization, Methodology, Software, Writing – original draft, Validation, Formal analysis. Xiao Feng: Validation, Formal analysis. Jin-Zhen Kong: Validation, Formal analysis. Zhike Peng: Validation, Formal analysis, Funding acquisition. Kwok-Leung Tsui: Validation, Formal analysis. Dong Wang: Supervision, Methodology, Writing – review & editing, Formal analysis, Funding acquisition.
Declaration of Competing Interest
The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.
Acknowledgment
This research work was fully supported by the National Natural Science Foundation of China (Project No. 51975355 and Grant No: 12121002), the “Zhiyuan Honors Program for Ph.D. students” at SJTU. We would like to thank anonymous reviewers for their valuable and constructive comments on our manuscript.
References (38)
Y. Lei et al.
Machinery health prognostics: A systematic review from data acquisition to RUL prediction
Mech. Syst. Signal Process.
(2018)
R.B. Randall et al.
Rolling element bearing diagnostics-A tutorial
Mech. Syst. Signal Process.
(2011)
Y. Lei et al.
Condition monitoring and fault diagnosis of planetary gearboxes: A review
Meas. J. Int. Meas. Confed.
(2014)
M. Feldman
Hilbert transform in vibration analysis
Mech. Syst. Signal Process.
(2011)
J. Antoni
Fast computation of the kurtogram for the detection of transient faults
Mech. Syst. Signal Process.
(2007)
Y. Miao et al.
A review on the application of blind deconvolution in machinery fault diagnosis
Mech. Syst. Signal Process.
(2022)
J. Antoni
The infogram: Entropic evidence of the signature of repetitive transients
Mech. Syst. Signal Process.
(2016)
M. Buzzoni et al.
Blind deconvolution based on cyclostationarity maximization and its application to fault identification
J. Sound Vib.
(2018)
G.L. McDonald et al.
Multipoint optimal minimum entropy deconvolution and convolution fix: application to vibration fault detection
Mech. Syst. Signal Process.
(2017)
Q. Ni et al.
A fault information-guided variational mode decomposition (FIVMD) method for rolling element bearings diagnosis
Mech. Syst. Signal Process.
(2022)

View more references
View full text
Recommended articles
Based on reading popularity
Research article
Resonance in dangerous mode and chaotic dynamics of a rotating pre-twisted graphene reinforced composite blade with variable thickness
Yan Niu, …, Qiliang Wu
Composite Structures • Volume 288 • 2022 • Article 115422
Research article
Reorganization of Septins Modulates Synaptic Transmission at Neuromuscular Junctions
Leniz F. Nurullin, …, Olga Vagin
Neuroscience • Volume 404 • 2019 • Pages 91-101
Research article
Fuel conditioning facility inert gas filled reprocessing hot cell leak rate measurement
Chad L. Pope, …, Jason P. Andrus
Annals of Nuclear Energy • Volume 111 • 2018 • Pages 676-682
Research article
Difference mode decomposition for adaptive signal decomposition
Bingchang Hou, …, Kwok-Leung Tsui
Mechanical Systems and Signal Processing • Volume 191 • 2023 • Article 110203
Research article
Measuring the no-load running torque of RV reducer based on the SVD and MCSA
Zhen Yu, …, Lianyu Zhao
Measurement • Volume 190 • 2022 • Article 110697
Review article
Constructing conditional symmetry in symmetric chaotic systems
Chunbiao Li, …, Zuohua Liu
Chaos, Solitons & Fractals • Volume 155 • 2022 • Article 111723
Cited by (28)
Research article
A new difference feature extraction method of slewing bearings in wind turbines via optimization bispectrum domain model
Yang M., …, Xu Y.
Expert Systems with Applications • Volume 278 • 2025 • Article 127325

Show abstract
Research article
Self-supervised graph feature enhancement and scale attention for mechanical signal node-level representation and diagnosis
Zhang X., …, Lu Y.
Advanced Engineering Informatics • Volume 65 • 2025 • Article 103197

Show abstract
Research article
Open access
A fault diagnosis method for analog circuits based on EEMD-PSO-SVM
Zhao S., …, Chen J.
Heliyon • Volume 10 • 2024 • Article e38064
Citation Excerpt:
Based on the specific preprocessing, six feature selection methods are selected to process the data-set. Including ReliefF algorithm, Recursive Feature Elimination algorithm [29](RFE), Maximum Information Coefficient algorithm [30](MIC), maximum Relevance Minimum Redundancy algorithm [31](mRMR), Infinite Latent Feature Selection algorithm [32](ILFS) and Fast Correlation Feature Selection algorithm [33,34](FCFS). A comparison and analysis of these feature selection methods were conducted to evaluate their efficacy in enhancing the classification model.


Show abstract
Research article
Dconformer: A denoising convolutional transformer with joint learning strategy for intelligent diagnosis of bearing faults
Li S., …, Wang Y.
Mechanical Systems and Signal Processing • Volume 210 • 2024 • Article 111142

Show abstract
Review article
Role of image feature enhancement in intelligent fault diagnosis for mechanical equipment: A review
Sun Y., Wang W.
Engineering Failure Analysis • Volume 156 • 2024 • Article 107815

Show abstract
Research article
A graph-guided collaborative convolutional neural network for fault diagnosis of electromechanical systems
Xu Y., …, Chen H.
Mechanical Systems and Signal Processing • Volume 200 • 2023 • Article 110609
Citation Excerpt:
The failure of rotating machinery can result in the shutdown of the electromechanical system and cause unexpected catastrophic incidents and economic loss. Therefore, it is vital to implement timely and accurate condition monitoring and fault diagnosis for electromechanical equipment, ensuring its safe and efficient operation [2]. Research on electromechanical system health monitoring primarily covers model-based approaches and data-driven approaches.
Link = https://journals.sagepub.com/doi/abs/10.1177/14759217241246508
Abstract
Deep learning (DL) networks, such as convolutional neural networks (CNNs) and long short-term memory (LSTM), have gained popularity for bearing fault diagnosis utilizing raw vibration signals. However, their accuracy and stability are compromised when facing imbalanced real-world datasets. This research investigates the impact of imbalanced datasets and explores the potential of signal processing techniques on network inputs compared to the direct use of raw vibration signals. The DL techniques studied include LSTM, one-dimensional CNN, and two-dimensional (2D) CNN, and a novel hybrid 2DCNNLSTM algorithm, incorporating signal processing methods such as Fourier transform and continuous wavelet transform while maintaining nearly equal parameters and the same base architecture. The proposed hybrid 2DCNNLSTM algorithm combines the strengths of LSTM and CNN, allowing for improved bearing diagnosis by capturing both spatial and temporal information in vibration signals. The proposed 2DCNNLSTM algorithm also considers multi-channel input augmenting raw vibration signal, mean, and variance channels to extract meaningful features and enhance classification efficiency. The publicly available Case Western Reserve University benchmark-bearing test rig dataset with ten fault classes, the Paderborn University dataset with three fault classes, and NASA Centre for Intelligent Maintenance Systems bearing datasets with five fault classes are utilized to test the proposed deep learning networks’ accuracy, effectiveness, robustness, and stability. The studies reveal that the hybrid 2DCNNLSTM-based networks outperform both CNN and LSTM networks, even without input processing. Further, utilizing multi-channel input by augmenting the 2D raw signal with mean and variance value channels proves to be more efficient in handling imbalanced and complex datasets while employing a 2DCNNLSTM-based network.
Get full access to this article
View all access and purchase options for this article.

Data availability statement
The data of CWRU and PU datasets can be both accessed: Jan 2023 [Online], from http://manufacturingnet.io/html/datasets.html
Case Western Reserve University (CWRU) Dataset Accessed: Dec. 2018, [Online], Available: https://csegroups.case.edu/bearingdatacenter/pages/welcome-casewestern-reserve-university-bearing-data-center-website. Paderborn University Dataset Accessed: Dec. 2018, [Online]. https://mb.unipaderborn.de/kat/forschung/datacenter/bearing-datacenter. NASA IMS Dataset Accessed: Jan 2023, [Online]. https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
References
1. Wu SD, Wu PH, Wu CW, et al. Bearing fault diagnosis based on multiscale permutation entropy and support vector machine. Entropy 2012; 14(8): 1343–1356.
Crossref
Web of Science
Google Scholar
2. Li H, Lian X, Guo C, et al. Investigation on early fault classification for rolling element bearing based on the optimal frequency band determination. J Intell Manuf 2015; 26: 189–198.
Crossref
Web of Science
Google Scholar
3. Thorsen OV, Dalva M. Failure identification and analysis for high-voltage induction motors in the petrochemical industry. IEEE Trans Ind Appl 1999; 35(4): 810–818.
Crossref
Web of Science
Google Scholar
4. Verstraete DB, Droguett EL, Meruane V, et al. Deep semi-supervised generative adversarial fault diagnostics of rolling element bearings. Struct Health Monit 2020; 19(2): 390–411.
Crossref
Web of Science
Google Scholar
5. Garcia-Perez A, de Jesus Romero-Troncoso R, Cabal-Yepez E, et al. The application of high-resolution spectral analysis for identifying multiple combined faults in induction motors. IEEE Trans Ind Electron 2011; 58(5): 2002–2010.
Crossref
Web of Science
Google Scholar
6. He M, He D. Report of large motor reliability survey of industrial and commercial installations, Part I. IEEE Trans Ind App 1985; 21(4): 853–864.
Google Scholar
7. Mojiri M, Karimi-Ghartemani M, Bakhshai A. Time-domain signal analysis using adaptive notch filter. IEEE Trans Signal Process 2006; 55(1): 85–93.
Crossref
Web of Science
Google Scholar
8. Petsounis KA, Fassois SD. Parametric time-domain methods for the identification of vibrating structures—a critical comparison and assessment. Mech Syst Signal Process 2001; 15(6): 1031–1060.
Crossref
Web of Science
Google Scholar
9. Noman K, Li Y, Wen G, et al. Continuous monitoring of rolling element bearing health by nonlinear weighted squared envelope-based fuzzy entropy. Struct Health Monit 2024; 23(1): 40–56.
Crossref
Web of Science
Google Scholar
10. Zhu D, Liu G, Wu X, et al. An enhanced empirical Fourier decomposition method for bearing fault diagnosis. Struct Health Monit 2024; 23(2): 903–923.
Crossref
Web of Science
Google Scholar
11. Yan X, Yan W, Yuen KV, et al. An adaptive variational mode extraction method based on multi-domain and multi-objective optimization for bearing fault diagnosis. Struct Health Monit 2023; 22(4): 2708–2733.
Crossref
Web of Science
Google Scholar
12. Ali JB, Fnaiech N, Saidi L, et al. Application of empirical mode decomposition and artificial neural network for automatic bearing fault diagnosis based on vibration signals. Appl Acoust 2015; 89: 16–27.
Crossref
Web of Science
Google Scholar
13. Zhang S, Zhang S, Wang B, et al. Deep learning algorithms for bearing fault diagnostics—a comprehensive review. IEEE Access 2020; 8: 29857–29881.
Crossref
Web of Science
Google Scholar
14. Chen X, Zhang B, Gao D. Bearing fault diagnosis base on multiscale CNN and LSTM model. J Intell Manuf 2021; 32: 971–987.
Crossref
Web of Science
Google Scholar
15. Li X, Zhang W, Ding Q, et al. Intelligent rotating machinery fault diagnosis based on deep learning using data augmentation. J Intell Manuf 2020; 31: 433–452.
Crossref
Web of Science
Google Scholar
16. Tian Y, Liu X. A deep adaptive learning method for rolling bearing fault diagnosis using immunity. Tsinghua Sci Technol 2019; 24(6): 750–762.
Crossref
Web of Science
Google Scholar
17. Yu L, Qu J, Gao F, et al. A novel hierarchical algorithm for bearing fault diagnosis based on stacked LSTM. Shock Vibr 2019; 2019: 1–10.
Web of Science
Google Scholar
18. Shao S, Sun W, Wang P, et al. Learning features from vibration signals for induction motor fault diagnosis. In: 2016 International symposium on flexible automation (ISFA), Ohio, 1 Aug 2016, pp. 71–76. IEEE.
Google Scholar
19. Li J, Liu Y, Li Q. Intelligent fault diagnosis of rolling bearings under imbalanced data conditions using attention-based deep learning method. Measurement 2022; 189: 110500.
Crossref
Web of Science
Google Scholar
20. Wang YR, Sun GD, Jin Q. Imbalanced sample fault diagnosis of rotating machinery using conditional variational auto-encoder generative adversarial network. Appl Soft Comput 2020; 92: 106333.
Crossref
Web of Science
Google Scholar
21. Fan Y, Cui X, Han H, et al. Chiller fault diagnosis with field sensors using the technology of imbalanced data. Appl Thermal Eng 2019; 159: 113933.
Crossref
Web of Science
Google Scholar
22. Peng P, Zhang W, Zhang Y, et al. Cost sensitive active learning using bidirectional gated recurrent neural networks for imbalanced fault diagnosis. Neurocomputing 2020; 407: 232–245.
Crossref
Web of Science
Google Scholar
23. Mao W, Liu Y, Ding L, et al. Imbalanced fault diagnosis of rolling bearing based on generative adversarial network: a comparative study. IEEE Access 2019; 7: 9515–9530.
Crossref
Web of Science
Google Scholar
24. Dong X, Gao H, Guo L, et al. Deep cost adaptive convolutional network: a classification method for imbalanced mechanical data. IEEE Access 2020; 8: 71486–71496.
Crossref
Web of Science
Google Scholar
25. Duong BP, Kim JY, Jeong I, et al. A deep-learning-based bearing fault diagnosis using defect signature wavelet image visualization. Appl Sci 2020; 10(24): 8800.
Crossref
Web of Science
Google Scholar
26. Verstraete D, Ferrada A, Droguett EL, et al. Deep learning enabled fault diagnosis using time-frequency image analysis of rolling element bearings. Shock Vibr 2017: 5067651.
Crossref
Web of Science
Google Scholar
27. Xiao Q, Li S, Zhou L, et al. Improved variational mode decomposition and CNN for intelligent rotating machinery fault diagnosis. Entropy 2022; 24(7): 908.
Crossref
PubMed
Web of Science
Google Scholar
28. Pan H, He X, Tang S, et al. An improved bearing fault diagnosis method using one-dimensional CNN and LSTM. Strojniski Vestnik/J Mech Eng 2018; 64: 443–452.
Web of Science
Google Scholar
29. Wang Y, Zhu C, Wang Q, et al. Research on fault detection of rolling bearing based on CWTDCCNN-LSTM. Eng Lett 2023; 31(3): 1–14.
Google Scholar
30. Tian H, Fan H, Feng M, et al. Fault diagnosis of rolling bearing based on HPSO algorithm optimized CNN-LSTM neural network. Sensors 2023; 23(14): 6508.
Crossref
Web of Science
Google Scholar
31. Qiao M, Yan S, Tang X, et al. Deep convolutional and LSTM recurrent neural networks for rolling bearing fault diagnosis under strong noises and variable loads. IEEE Access 2020; 8: 66257–66269.
Crossref
Web of Science
Google Scholar
32. Sun H, Fan Y. Fault diagnosis of rolling bearings based on CNN and LSTM networks under mixed load and noise. Multimedia Tools Appl 2023; 82: 1–25.
Crossref
Web of Science
Google Scholar
33. Peng H, Li H, Zhang Y, et al. Multi-sensor vibration signal based three-stage fault prediction for rotating mechanical equipment. Entropy 2022; 24(2): 164.
Crossref
PubMed
Web of Science
Google Scholar
34. Case Western Reserve University (CWRU) Bearing Data Center [Online], https://csegroups.case.edu/bearingdatacenter/pages/welcome-casewestern-reserve-university-bearing-data-center-website (2018) (accessed 5 August 2022).
Google Scholar
35. Bearing DataCenter, Paderborn University, https://mb.unipaderborn.de/kat/forschung/datacenter/bearing-datacenter (2018) (accessed 5 August 2022).
Google Scholar
36. Datasets, https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
Google Scholar
37. Datasets, http://manufacturingnet.io/html/datasets.html
Google Scholar
38. Gousseau W, Antoni J, Girardin F, et al. Analysis of the rolling element bearing data set of the center for intelligent maintenance systems of the University of Cincinnati. In: CM2016, 2016.
Google Scholar
39. Cavalaglio Camargo Molano J, Strozzi M, Rubini R, et al. Analysis of NASA bearing dataset of the University of Cincinnati by means of Hjorth’s parameters. In: Proceedings of the International Conference on Structural Engineering Dynamics, ICEDyn 2019, 2019.
Google Scholar
40. Choudhury A, Tandon N. Vibration response of rolling element bearings in a rotor bearing system to a local defect under radial load. Tribol Int 2006; 128: 252–261.
Crossref
Web of Science
Google Scholar
41. Kaewkongka T, Joe YH, Rakowski RT, et al. A comparative study of short time Fourier transform and continuous wavelet transform for bearing condition monitoring. Int J COMADEM 2003; 6(1): 41–48.
Google Scholar
42. Fast Fourier Transform - an overview|ScienceDirect Topics, https://www.sciencedirect.com/topics/engineering/fast-fouriertransform (1999, accessed 10 September 2022).
Google Scholar
43. Wavelets and Wavelet Transform Systems and Their Applications|SpringerLink, https://link.springer.com/book/10.1007/978-3-030-87528-2 (2022, accessed 12 September 2022).
Google Scholar
44. Qiu H, Lee J, Lin J, et al. Wavelet filter-based weak signature detection method and its application on rolling element bearing prognostics. J Sound Vibr 2006; 289(4–5): 1066–1090.
Crossref
Web of Science
Google Scholar
45. Girshick R, Donahue J, Darrell T, et al. Rich feature hierarchies for accurate object detection and semantic segmentation. In: Proceedings of the IEEE conference on computer vision and pattern recognition, Columbus, OH, USA, 2014, pp. 580–587.
Google Scholar
46. Turay T, Vladimirova T. Toward performing image classification and object detection with convolutional neural networks in autonomous driving systems: a survey. IEEE Access. 2022; 10: 14076–14119.
Crossref
Web of Science
Google Scholar
47. Alzubaidi L, Zhang J, Humaidi AJ, et al. Review of deep learning: concepts, CNN architectures, challenges, applications, future directions. J Big Data 2021; 8: 1–74.
Crossref
PubMed
Web of Science
Google Scholar
48. IBM Cloud Education, “What are Convolutional Neural Networks?,” Convolutional Neural Networks. https://www.ibm.com/cloud/learn/convolutional-neural-networks (2021, accessed 12 October 2022).
Google Scholar
49. Jin X, Xu C, Feng J, et al. Deep learning with s-shaped rectified linear activation units. In: Proceedings of the AAAI conference on artificial intelligence, 2016, Vol. 30, No. 1.
Google Scholar
50. Srivastava N, Hinton G, Krizhevsky A, et al. Dropout: a simple way to prevent neural networks from overfitting. J Mach Learn Res 2014; 15(1): 1929–1958.
Google Scholar
51. Kohar CP, Connolly DS, Liusko T, et al. Using artificial intelligence to aid vehicle lightweighting in crashworthiness with aluminum. In: MATEC Web of conferences, France, 2020, vol. 326, p. 01006. EDP Sciences.
Crossref
Google Scholar
52. Hochreiter S, Schmidhuber J. Long short-term memory. Neural Comput 1997; 9(8): 1735–1780.
Crossref
PubMed
Web of Science
Google Scholar
53. Greff K, Srivastava RK, Koutník J, et al. LSTM: a search space odyssey. IEEE Trans Neural Networks Learn Syst 2016; 28(10): 2222–2232.
Crossref
PubMed
Web of Science
Google Scholar
54. Sak H, Senior A, Beaufays F. Long short-term memory based recurrent neural network architectures for large vocabulary speech recognition. arXiv preprint: 1402.1128, 2014.
Google Scholar
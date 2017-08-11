band-pass filter between 8–30 Hz, which covers the motor-related mu and beta rhythm.

==========----------========== bci3d3a
(330, 1, 60, 750) (315, 1, 60, 750)

3 seconds, 250 HZ, 4 class. 
60 chn
1/4 trials rejected. 

: bad don't use ! - - - - - - - - - - -  - -  - - - -

A Self-Regulated Interval Type-2 Neuro-Fuzzy Inference System for Handling Nonstationarities in EEG Signals for BCI 16
	conv to: hand (class 1) and right hand (class 2)
	ONLY 2 classes.!
	per subj: 100%. 78.83%. 100%.


Competition:
kappa 	 K3 	 K6 	 L1 	
0.7926 	0.8222 	0.7556 	0.8000 	


===========----------========= bci3d4a
(2280, 1, 118, 750) (3404, 1, 118, 750)

2 classes, 118 EEG channels (0.05-200Hz), 1000Hz sampling rate, left hand, foot
motor imagery
5  subject.
3.5 seconds, 3500 sp. 
label: 1:1 , 2:2, 
1= right, 2=foot, or 0 for test trials


Competition:
acc 	 aa 	 al 	 av 	 aw 	 ay 	
94.17% 	95.5% 	100.0% 	80.6% 	100.0% 	97.6% 

Interpretable deep neural networks for single-trial EEG classification 16
	layer-wise relevance propagation (LRP), compare the classification performance of DNN to that of CSP-LDA
	bandpass filtered in the range of 9–13Hz. 
	only giving individual separate classification results. 
	subj so diff that, inter subj 68.4% 
	BCI 3    4a, R/L, 62%-93% per subj.

 csp/lda dnn
aa 66 62 
al 100 93 
av 70 66 
aw 99 77 
ay 55 60


Filter Bank Common Spatial Pattern (FBCSP) in Brain-Computer Interface
	For each individual only, per subj.
	BCI 3    4a, per subj: aa 86, al 98, av 76, aw 96, ay 94, 

""
Boosting bit rates in non-invasive EEG single-trial classifications by feature combination and multi-class paradigms


===========----------========= bci3d4c
(840, 1, 118, 750) (1120, 1, 118, 750)

2 classes, 118 EEG channels (0.05-200Hz), 1000Hz sampling rate, left hand, foot
one healthy subject.
3 seconds, 3000 sp. 
label: 1:1 , -1:2, 
-1 for left , 1 for foot
test data for 4b is not good, no submission. use 4c test data instead. 
4c test data contains 3 classes, we only use 2. not valid for competition.

- no related work used this dataset....


Competition, mean square error
-1 for class left, 0 for relax and 1 for foot
min mse winner: 0.30



===========----------========= bci3d5
(1350, 1, 32, 750) (448, 1, 32, 750)

Are low cost Brain Computer Interface headsets ready for motor imagery applications?
	every 1s, Surface Laplacian (SL), Power Spectral Density (PSD), requency bands from 8 to 30 Hz when using a bandwidth of 2 Hz, fuzzy logic.
    only for individual, acc per subj:
    87.67%.  82.26%.  58.72%.
    BCI 3    5, 32chn 


Feature Selection Applying Statistical and Neurofuzzy Methods to EEG-Based BCI
	surface Laplacian, a Power Spectral Density (PSD), 8 and 30 Hz with a resolution of 2 Hz, S-dFasArt architecture proposed by Cano-Izquierdo
    BCI 3    5, 32chn 
    per subj:
    87.64%.  81.57%.  59.4%. 


Competition:
12 band * 8 channel = 96 features.
per subj:
79.6%.  70.31%.  56.02%.  

""
On the need for on-line learning in brain-computer interfaces


===========----------========= bci4d2a
(2610, 1, 25, 750) (2592, 1, 25, 750)

3 seconds, 250 HZ, 4 class. 


A Self-Regulated Interval Type-2 Neuro-Fuzzy Inference System for Handling Nonstationarities in EEG Signals for BCI 16
	BCI 4    2a :
	pairwise, binary, -89%
	4 classes, 67%


Deep learning with convolutional neural networks for brain mapping and decoding of movement-related information from the human EEG
	BCI 4    2a, 73.7%, 


Competition:
kappa, 	 1 		 2 		 3 		 4 		 5 		 6 	 	7 		 8 		 9 	
0.57, 	0.68 	0.42 	0.75 	0.48 	0.40 	0.27 	0.77 	0.75 	0.61

acc = 0.6775

""
Spatial filtering and selection of optimized components in four class motor imagery data using independent components analysis

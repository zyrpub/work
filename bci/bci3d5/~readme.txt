bci 3 d 5



Are low cost Brain Computer Interface headsets ready for motor imagery applications?
    only for individual, acc per subj:
    87.67%.  82.26%.  58.72%.
    BCI 3    5, 32chn 


Feature Selection Applying Statistical and Neurofuzzy Methods to EEG-Based BCI
    BCI 3    5, 32chn 
    per subj:
    87.64%.  81.57%.  59.4%. 


Competition:
12 band * 8 channel = 96 features.
per subj:
79.6%.  70.31%.  56.02%.  

----------------
output every 0.5 seconds, 256sp.

Data set V ‹mental imagery, multi-class›

Data set provided by IDIAP Research Institute (Silvia Chiappa, José del R. Millán)

Correspondence to José del R. Millán ⟨jose.millan@idiap.ch⟩

Experiment
This dataset contains data from 3 normal subjects during 4 non-feedback sessions. The subjects sat in a normal chair, relaxed arms resting on their legs. There are 3 tasks:

    Imagination of repetitive self-paced left hand movements, (left, class 2),
    Imagination of repetitive self-paced right hand movements, (right, class 3),
    Generation of words beginning with the same random letter, (word, class 7).

All 4 sessions of a given subject were acquired on the same day, each lasting 4 minutes with 5-10 minutes breaks in between them. The subject performed a given task for about 15 seconds and then switched randomly to another task at the operator's request. EEG data is not splitted in trials since the subjects are continuously performing any of the mental tasks. The algorithm should provide an output every 0.5 seconds using the last second of data (see clarification in the paragraph 'Requirements and Evaluation'.) Data are provided in two ways:

    Raw EEG signals. Sampling rate was 512 Hz.

    Precomputed features. The raw EEG potentials were first spatially filtered by means of a surface Laplacian. Then, every 62.5 ms --i.e., 16 times per second-- the power spectral density (PSD) in the band 8-30 Hz was estimated over the last second of data with a frequency resolution of 2 Hz for the 8 centro-parietal channels C3, Cz, C4, CP1, CP2, P3, Pz, and P4. As a result, an EEG sample is a 96-dimensional vector (8 channels times 12 frequency components).

Format of the Data
For each subject there are 3 training files and 1 testing file (the last recording session). Training files are labelled while testing files are not. Data are provided in ASCII format.

    Precomputed features: files contain a PSD sample per row (i.e., the first 12 components are the PSD in the band 8-30 Hz at channel C3, and so on, for a total of 96 components). The number of PSD samples are:
    	training 	testing
    Subject 1   	3488/3472/3568 	3504
    Subject 2 	3472/3456/3472 	3472
    Subject 3 	3424/3424/3440 	3488
    In the training files, there is a 97th component indicating the class label.
    Raw EEG signals: each line of the files contains the 32 EEG potentials acquired at a given time instant in the order: Fp1, AF3, F7, F3, FC1, FC5, T7, C3, CP1, CP5, P7, P3, Pz, PO3, O1, Oz, O2, PO4, P4, P8, CP6, CP2, C4, T8, FC6, FC2, F4, F8, AF4, Fp2, Fz, Cz. In the training files, each line has a 33rd component indicating the class label.

Requirements and Evaluation
Please provide your estimated class labels (2, 3, or 7) for every input vector of the 3 test files (one per subject). The labels must be estimated in the following way:

    Precomputed features: Since input vectors are computed 16 times per second, provide the average of 8 consecutive samples (so that to get a response every 0.5 seconds). Other (i.e. also past) samples must not be used in order to guarantee a fast response times of the system, although for the competition test data set averaging over more samples could be of benefit.
    
    Raw signals: Compute vectors 16 times per second using the last second of data. Then provide the average of 8 consecutive samples (so that to get a response every 0.5 seconds). Other (i.e. also past) samples must not be used in order to guarantee a fast response times of the system, although for the competition test data set averaging over more samples could be of benefit.

Also give a description of the used algorithm. The performance measure is the classification accuracy (correct classification divided by the total number of samples) averaged over the 3 subjects.
There will be a special prize to the best algorithm working with the precomputed samples (in the case it does not achieve the best absolute result).

Technical Information
EEG signals were recorded with a Biosemi system using a cap with 32 integrated electrodes located at standard positions of the International 10-20 system. The sampling rate was 512 Hz. Signals were acquired at full DC. No artifact rejection or correction was employed.

Reference

    Millán, J. del R.. On the need for on-line learning in brain-computer interfaces Proc. Int. Joint Conf. on Neural Networks., 2004.

[ BCI Competition III ]


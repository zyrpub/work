# Copied from IBM git:

# Brain wave classification project

# About this repo

EEG classification code. Data set is BCI competition IV data set 2a. Code uses Tensorflow.

# Install

1. Install Anaconda with Python 2.7 (https://www.continuum.io/downloads)
3. Install memcache module `pip install python-memcached`. You need to install Memcached and Libmemcached first. Memcache allows persistent in-memory data set storage, for faster training cycles.
4. Install tensorflow 1.2.0.
5. Code on CCC is located at 
```
srallap@dccxl001.pok.ibm.com:/u/srallap/eeg/syncdir/
```

# Data set

The dataset is from BCI Competition IV Data set 2a (http://www.bbci.de/competition/iv/#dataset2a). Raw data is in the folder `/dccstor/srallap1/eeg/bci/bci4d2a/`:
```
raw/A01T.txt
raw/A02T.txt
raw/A03T.txt
raw/A04T.txt
raw/A05T.txt
raw/A06T.txt
raw/A07T.txt
raw/A08T.txt
raw/A09T.txt
raw/A01E.txt
raw/A02E.txt
raw/A03E.txt
raw/A04E.txt
raw/A05E.txt
raw/A06E.txt
raw/A07E.txt
raw/A08E.txt
raw/A09E.txt
```

This data set is for motor imagery classification (left hand, right hand, feet, tongue), with 9 subjects each perform 288 trials, each trial lasts for about 3 seconds. There are 22 EEG channels (0.5-100Hz; notch filtered) plus 3 EOG channels, in 250Hz sampling rate. Each file contains all trial samples for the same person. Each line has the following format:
```
CH1, CH2, ... CH25, label
```
Where label is one of [1,2,3,4], and 1 is left, 2 is right, 3 is feet, 4 is tongue. 

# Edit configuration

Most configurations are located in `/u/srallap/eeg/syncdir/<dir>/conf.txt` file, and are read by a function in `mytools/readconf.py`. Dataset specific configurations are in `./conf/conf-bci4d2a.txt`.

# Processing

Data preprocessing involves the following steps:
1. The sample rate is reduced from 250 Hz to 125 Hz, this also splits the data and doubles the size. 
2. Notch filter to suppress at 50 Hz, also high pass filter to suppress frequency below 0.5 Hz. 
3. Normalization can be done but un-normalized raw data can achieve slightly better results.
4. Each trial is 3 seconds, so each frame/image is 25 by 375. 

# Training

The training uses CNN only. The CNN for time domain first uses 1D filters along the time axis, then uses NumChannel-by-1 filters to capture the cross channel relationship. The last layer is a fully connected layer followed by a softmax layer with 4 classes. The input files ending with "T" are used in training, and files ending in "E" are used for testing/validation. 

0. Do this every time you login:
```
cd /u/srallap/eeg/syncdir/<dir>
source /u/srallap/eeg/anaconda2/bin/activate tf12
export PYTHONPATH=/u/srallap/eeg/anaconda2/envs/tf12/lib/python2.7/site-packages
```

1. Cutting the raw data into segments with same label (run only once for all):
```
jbsub -interactive -queue x86_24h -mem 10g python 0separateSameLabel.py 
```

2. Run some cache servers, and read and store processed data in memcached (run only once every 5 days):
```
jbsub -interactive -queue x86_7d -mem 40g sh /u/srallap/eeg/memcached/start.sh 0
jbsub -interactive -queue x86_7d -mem 40g sh /u/srallap/eeg/memcached/start.sh 1
jbsub -interactive -queue x86_7d -mem 40g sh /u/srallap/eeg/memcached/start.sh 2
jbsub -interactive -queue x86_6h -mem 20g python 0cache_proc_data.py 4
```

3. Run training:
```
jbsub -interactive -queue x86_6h -cores 2+1 -mem 20g -require '(cpuf>=40) && (ncpus>=24)' python 4train.py 
```


# Results

With 90 epoch, the accuracy is around 79.4%, which is better than current state of the art (73.7% [2]).
```
...

train loss 0.302786 train accuracy 0.975
epoch 85.3846153846 batch 11100 lr [0.00058564596]
train loss 0.616864 train accuracy 0.85
epoch 85.7692307692 batch 11150 lr [0.00057597971]
bci4d2a 4 --- test l= 0.623275 a= 0.797307692308
train loss 0.49529 train accuracy 0.9
epoch 86.1538461538 batch 11200 lr [0.00057120656]
train loss 0.309001 train accuracy 1.0
epoch 86.5384615385 batch 11250 lr [0.00056177872]
train loss 0.700679 train accuracy 0.875
epoch 86.9230769231 batch 11300 lr [0.00055712328]
bci4d2a 4 --- test l= 0.620316 a= 0.795961538462
train loss 0.339879 train accuracy 0.95
epoch 87.3076923077 batch 11350 lr [0.0005479278]
train loss 0.378124 train accuracy 0.95
epoch 87.6923076923 batch 11400 lr [0.0005433872]
bci4d2a 4 --- test l= 0.629585 a= 0.791538461538
train loss 0.389713 train accuracy 0.925
epoch 88.0769230769 batch 11450 lr [0.00053441845]
train loss 0.532559 train accuracy 0.875
epoch 88.4615384615 batch 11500 lr [0.00052998972]
train loss 0.38348 train accuracy 0.975
epoch 88.8461538462 batch 11550 lr [0.00052124209]
bci4d2a 4 --- test l= 0.622512 a= 0.795576923077
train loss 0.484849 train accuracy 0.875
epoch 89.2307692308 batch 11600 lr [0.00051692262]
train loss 0.428859 train accuracy 0.925
epoch 89.6153846154 batch 11650 lr [0.00050839072]
train loss 0.539974 train accuracy 0.85
epoch 90.0 batch 11700 lr [0.00049999962]
bci4d2a 4 --- test l= 0.639625 a= 0.794230769231
bci4d2a 4 ----test loss 0.639625 accuracy 0.794230769231
```

# Useful papers

1. DEEP FEATURE LEARNING FOR EEG RECORDINGS

2. Deep learning with convolutional neural networks for brain mapping and decoding of movement-related information from the human EEG

3. LEARNING REPRESENTATIONS FROM EEG WITH DEEP RECURRENT-CONVOLUTIONAL NEURAL NETWORKS

4. Deep Learning EEG Response Representation for Brain Computer Interface






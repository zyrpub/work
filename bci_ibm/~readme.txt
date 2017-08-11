all bci data sets,

1train: ensemble, 
2train: ensemble, all chn dim. input projection to smaller dim.

3train: cnn2d.
4train: cnn2d. input projection to smaller dim.

train-all: all datasets. input projection to smaller dim.


------- best -------
0.791923076923
9.47.193.185 Saved: ./model.ckpt
4train.py bci4d2a [0, 1, 2, 3, 4, 5, 6, 7, 8] [0, 1, 2, 3, 4, 5, 6, 7, 8]
log/log-4train.py193.185-0-2017-08-10-17-22-23.txt
(tf12) [shu@dccxl011 bci_ibm]$ cat log/log-4train.py193.185-0-2017-08-10-17-22-23.txt
4train.py193.185
9.47.193.185
2017-08-10 17:22:23
1502400143.85

4train.py 
regularize_coef= 0.0006 
batch_size= 40 
stateStep= 1 
nUseChannels= 25 
secondPerFrame= 3.0 
width= 375 
height= 25 
convNum1= 64 
convNum2= 64 
convNum3= 64 
convNum4= 64 
input_proj_dim= 30 
CONV_KEEP_PROB= 0.5 
fully_out_dim= 95 
total_parameters= 191335 
init_lr= 0.01 
end_lr= 0.001 
nepoch= 50 
total_steps= 6500.0 
decay_steps= 32.5 
weight_decay_rate= 0.988553094657 
epoch= 1.0 trainloss= 1.70918 trainacc 0.3 testloss 21.2751 testacc 0.250192307692 
epoch= 2.0 trainloss= 1.72021 trainacc 0.2 testloss 21.5313 testacc 0.25 
epoch= 3.0 trainloss= 1.80916 trainacc 0.25 testloss 3.77903 testacc 0.250192307692 
epoch= 4.0 trainloss= 1.41509 trainacc 0.425 testloss 1.47075 testacc 0.365769230769 
epoch= 5.0 trainloss= 1.45958 trainacc 0.425 testloss 1.13269 testacc 0.493653846154 
epoch= 6.0 trainloss= 1.25838 trainacc 0.475 testloss 1.06225 testacc 0.544615384615 
epoch= 7.0 trainloss= 1.27011 trainacc 0.525 testloss 0.983647 testacc 0.595192307692 
epoch= 8.0 trainloss= 1.17417 trainacc 0.7 testloss 0.979971 testacc 0.586346153846 
epoch= 9.0 trainloss= 1.16733 trainacc 0.625 testloss 0.906985 testacc 0.625769230769 
epoch= 10.0 trainloss= 1.49517 trainacc 0.5 testloss 0.870119 testacc 0.653269230769 
epoch= 11.0 trainloss= 0.979623 trainacc 0.675 testloss 0.842333 testacc 0.660192307692 
epoch= 12.0 trainloss= 1.10194 trainacc 0.625 testloss 0.801896 testacc 0.692307692308 
epoch= 13.0 trainloss= 1.17672 trainacc 0.7 testloss 0.801917 testacc 0.680576923077 
epoch= 14.0 trainloss= 1.02366 trainacc 0.725 testloss 0.843621 testacc 0.660576923077 
epoch= 15.0 trainloss= 1.25553 trainacc 0.55 testloss 0.821913 testacc 0.674807692308 
epoch= 16.0 trainloss= 1.07918 trainacc 0.675 testloss 0.837092 testacc 0.659230769231 
epoch= 17.0 trainloss= 1.0071 trainacc 0.775 testloss 0.753942 testacc 0.696923076923 
epoch= 18.0 trainloss= 1.14295 trainacc 0.65 testloss 0.737594 testacc 0.699423076923 
epoch= 19.0 trainloss= 0.943856 trainacc 0.75 testloss 0.741165 testacc 0.720769230769 
epoch= 20.0 trainloss= 1.18344 trainacc 0.65 testloss 0.725653 testacc 0.725961538462 
epoch= 21.0 trainloss= 0.790598 trainacc 0.825 testloss 0.729649 testacc 0.721346153846 
epoch= 22.0 trainloss= 0.980398 trainacc 0.75 testloss 0.745104 testacc 0.712884615385 
epoch= 23.0 trainloss= 1.05237 trainacc 0.7 testloss 0.681423 testacc 0.745576923077 
epoch= 24.0 trainloss= 0.926032 trainacc 0.775 testloss 0.686507 testacc 0.737692307692 
epoch= 25.0 trainloss= 1.29006 trainacc 0.575 testloss 0.688181 testacc 0.74 
epoch= 26.0 trainloss= 0.715863 trainacc 0.875 testloss 0.701838 testacc 0.724423076923 
epoch= 27.0 trainloss= 0.807639 trainacc 0.8 testloss 0.710138 testacc 0.708846153846 
epoch= 28.0 trainloss= 0.835425 trainacc 0.75 testloss 0.686619 testacc 0.730192307692 
epoch= 29.0 trainloss= 0.827286 trainacc 0.8 testloss 0.689214 testacc 0.729038461538 
epoch= 30.0 trainloss= 1.09222 trainacc 0.675 testloss 0.64425 testacc 0.754230769231 
epoch= 31.0 trainloss= 0.833172 trainacc 0.8 testloss 0.660554 testacc 0.736730769231 
epoch= 32.0 trainloss= 0.790214 trainacc 0.725 testloss 0.625801 testacc 0.770769230769 
epoch= 33.0 trainloss= 0.953199 trainacc 0.725 testloss 0.624437 testacc 0.760576923077 
epoch= 34.0 trainloss= 0.567212 trainacc 0.9 testloss 0.608099 testacc 0.765576923077 
epoch= 35.0 trainloss= 0.921865 trainacc 0.775 testloss 0.59892 testacc 0.776153846154 
epoch= 36.0 trainloss= 0.525091 trainacc 0.95 testloss 0.59237 testacc 0.780384615385 
epoch= 37.0 trainloss= 0.814863 trainacc 0.8 testloss 0.596676 testacc 0.780576923077 
epoch= 38.0 trainloss= 0.694483 trainacc 0.875 testloss 0.607873 testacc 0.778653846154 
epoch= 39.0 trainloss= 0.664746 trainacc 0.85 testloss 0.595891 testacc 0.784038461538 
epoch= 40.0 trainloss= 1.0276 trainacc 0.725 testloss 0.576652 testacc 0.783846153846 
epoch= 41.0 trainloss= 0.742612 trainacc 0.875 testloss 0.586398 testacc 0.78 
epoch= 42.0 trainloss= 0.780059 trainacc 0.725 testloss 0.580853 testacc 0.785961538462 
epoch= 43.0 trainloss= 0.740646 trainacc 0.85 testloss 0.573897 testacc 0.7875 
epoch= 44.0 trainloss= 0.732931 trainacc 0.875 testloss 0.575791 testacc 0.783653846154 
epoch= 45.0 trainloss= 0.714102 trainacc 0.875 testloss 0.57887 testacc 0.7825 
epoch= 46.0 trainloss= 0.454385 trainacc 0.95 testloss 0.57299 testacc 0.785961538462 
epoch= 47.0 trainloss= 0.804096 trainacc 0.875 testloss 0.565671 testacc 0.788461538462 
epoch= 48.0 trainloss= 0.649482 trainacc 0.775 testloss 0.589234 testacc 0.778076923077 
epoch= 49.0 trainloss= 0.587946 trainacc 0.85 testloss 0.567336 testacc 0.790576923077 
epoch= 50.0 trainloss= 0.639827 trainacc 0.85 testloss 0.562989 testacc 0.791923076923 
Final testloss 0.562989 testacc 0.791923076923 

---
0.789:
cat log/log-4train.py193.180-0-2017-08-10-17-17-01.txt
4train.py193.180
9.47.193.180
2017-08-10 17:17:01
1502399821.5

4train.py 
regularize_coef= 0.0006 
batch_size= 40 
stateStep= 1 
nUseChannels= 25 
secondPerFrame= 3.0 
width= 375 
height= 25 
convNum1= 64 
convNum2= 64 
convNum3= 64 
convNum4= 64 
input_proj_dim= 20 
CONV_KEEP_PROB= 0.5 
fully_out_dim= 90 
total_parameters= 148810 
init_lr= 0.01 
end_lr= 0.001 
nepoch= 50 
total_steps= 6500.0 
decay_steps= 32.5 
weight_decay_rate= 0.988553094657 
epoch= 1.0 trainloss= 1.75027 trainacc 0.225 testloss 49.7035 testacc 0.250192307692 
epoch= 2.0 trainloss= 1.84275 trainacc 0.175 testloss 6.67431 testacc 0.254038461538 
epoch= 3.0 trainloss= 1.636 trainacc 0.25 testloss 4.78117 testacc 0.25 
epoch= 4.0 trainloss= 1.49643 trainacc 0.45 testloss 1.7856 testacc 0.275384615385 
epoch= 5.0 trainloss= 1.50885 trainacc 0.375 testloss 1.35622 testacc 0.286346153846 
epoch= 6.0 trainloss= 1.39272 trainacc 0.375 testloss 1.26728 testacc 0.392115384615 
epoch= 7.0 trainloss= 1.36749 trainacc 0.475 testloss 1.19833 testacc 0.451538461538 
epoch= 8.0 trainloss= 1.3184 trainacc 0.525 testloss 1.2246 testacc 0.419615384615 
epoch= 9.0 trainloss= 1.33757 trainacc 0.525 testloss 1.0047 testacc 0.594807692308 
epoch= 10.0 trainloss= 1.19726 trainacc 0.625 testloss 0.912028 testacc 0.639615384615 
epoch= 11.0 trainloss= 1.00072 trainacc 0.75 testloss 0.905435 testacc 0.635 
epoch= 12.0 trainloss= 1.15036 trainacc 0.6 testloss 0.91077 testacc 0.625961538462 
epoch= 13.0 trainloss= 1.11203 trainacc 0.6 testloss 0.906146 testacc 0.638846153846 
epoch= 14.0 trainloss= 1.02875 trainacc 0.65 testloss 0.850061 testacc 0.661923076923 
epoch= 15.0 trainloss= 1.20615 trainacc 0.625 testloss 0.811366 testacc 0.675 
epoch= 16.0 trainloss= 0.911634 trainacc 0.75 testloss 0.825085 testacc 0.675576923077 
epoch= 17.0 trainloss= 1.05528 trainacc 0.75 testloss 0.754157 testacc 0.702884615385 
epoch= 18.0 trainloss= 1.07818 trainacc 0.7 testloss 0.699474 testacc 0.730384615385 
epoch= 19.0 trainloss= 0.828909 trainacc 0.825 testloss 0.693048 testacc 0.720769230769 
epoch= 20.0 trainloss= 1.18923 trainacc 0.675 testloss 0.667768 testacc 0.748269230769 
epoch= 21.0 trainloss= 0.839598 trainacc 0.825 testloss 0.68303 testacc 0.733846153846 
epoch= 22.0 trainloss= 1.10413 trainacc 0.675 testloss 0.659183 testacc 0.75 
epoch= 23.0 trainloss= 0.862755 trainacc 0.825 testloss 0.660691 testacc 0.742884615385 
epoch= 24.0 trainloss= 1.19196 trainacc 0.625 testloss 0.671527 testacc 0.738461538462 
epoch= 25.0 trainloss= 1.28506 trainacc 0.65 testloss 0.664737 testacc 0.743461538462 
epoch= 26.0 trainloss= 0.664186 trainacc 0.875 testloss 0.6405 testacc 0.759038461538 
epoch= 27.0 trainloss= 0.94255 trainacc 0.65 testloss 0.638567 testacc 0.749423076923 
epoch= 28.0 trainloss= 0.971323 trainacc 0.8 testloss 0.649502 testacc 0.754807692308 
epoch= 29.0 trainloss= 0.839704 trainacc 0.725 testloss 0.616456 testacc 0.778653846154 
epoch= 30.0 trainloss= 0.862637 trainacc 0.75 testloss 0.603017 testacc 0.779807692308 
epoch= 31.0 trainloss= 0.666749 trainacc 0.9 testloss 0.617569 testacc 0.768461538462 
epoch= 32.0 trainloss= 0.854823 trainacc 0.8 testloss 0.613728 testacc 0.775384615385 
epoch= 33.0 trainloss= 0.894368 trainacc 0.825 testloss 0.627511 testacc 0.765192307692 
epoch= 34.0 trainloss= 0.790519 trainacc 0.775 testloss 0.585213 testacc 0.785576923077 
epoch= 35.0 trainloss= 0.980877 trainacc 0.675 testloss 0.599308 testacc 0.780384615385 
epoch= 36.0 trainloss= 0.705183 trainacc 0.8 testloss 0.590616 testacc 0.786730769231 
epoch= 37.0 trainloss= 0.679667 trainacc 0.875 testloss 0.615466 testacc 0.769423076923 
epoch= 38.0 trainloss= 0.690863 trainacc 0.85 testloss 0.58097 testacc 0.786923076923 
epoch= 39.0 trainloss= 0.69463 trainacc 0.825 testloss 0.595464 testacc 0.778076923077 
epoch= 40.0 trainloss= 0.889827 trainacc 0.75 testloss 0.59144 testacc 0.778846153846 
epoch= 41.0 trainloss= 0.52696 trainacc 0.9 testloss 0.615355 testacc 0.771730769231 
epoch= 42.0 trainloss= 0.724152 trainacc 0.85 testloss 0.582521 testacc 0.784230769231 
epoch= 43.0 trainloss= 1.0756 trainacc 0.725 testloss 0.591247 testacc 0.779807692308 
epoch= 44.0 trainloss= 0.596689 trainacc 0.875 testloss 0.584664 testacc 0.780961538462 
epoch= 45.0 trainloss= 0.754506 trainacc 0.825 testloss 0.559397 testacc 0.796923076923 
epoch= 46.0 trainloss= 0.612336 trainacc 0.825 testloss 0.563541 testacc 0.790576923077 
epoch= 47.0 trainloss= 0.559755 trainacc 0.875 testloss 0.55324 testacc 0.798269230769 
epoch= 48.0 trainloss= 0.780063 trainacc 0.725 testloss 0.563947 testacc 0.788269230769 
epoch= 49.0 trainloss= 0.757731 trainacc 0.825 testloss 0.557124 testacc 0.791730769231 
epoch= 50.0 trainloss= 0.798948 trainacc 0.8 testloss 0.554231 testacc 0.789423076923 
Final testloss 0.554231 testacc 0.789423076923
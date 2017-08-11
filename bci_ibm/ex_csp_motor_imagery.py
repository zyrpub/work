from scipy import signal
from sklearn import svm, pipeline, base, metrics
import eegtools

import os, sys,subprocess
import subprocess
import random, time
import inspect, glob
import copy
mypydir =os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe())))
sys.path.append(mypydir)
sys.path.append(mypydir+"/mytools")
import collections
import math
import random
import zipfile
import numpy as np

from namehostip import get_my_ip
from hostip import ip2tarekc,tarekc2ip
from readconf import get_conf,get_conf_int,get_conf_float,get_list_startswith,get_dic_startswith
from logger import Logger
from util import read_lines_as_list,read_lines_as_dic
from inputClass import inputClass,FeedInput

iprint = 1

usr = os.path.expanduser("~")
if usr.endswith("shu"):
  WORKDIR = os.path.expanduser("~")
elif usr.endswith("srallap"):
  WORKDIR = os.path.expanduser("~")+"/eeg"

configfile = "conf.txt"
confs = "conf/conf-%s.txt"

ind2trainSubj = {
0: [0], #[0,1,2], # 0:.42,1:.34,2:.28
1: [1], #[0,1,2,3,4], # 0:.53,1:.72,2:.48,3:.51,4:.49
2: [0], #[0], # 0:.60,
3: [1], #[0,1,2], # 0:.29,1:.42,2:.32 
4: [0,2,3,4,6,7,8], #[0,1,2,3,4,5,6,7,8], # 0:.59,1:.42,2:.66,3:.57,4:.83,5:.47,6:.80,7:.77,8:.68
}
ind2testSubj = {
0: [0], #[0,1,2], # 0:.38,1:.34,2:.28
1: [1], #[0,1,2,3,4], # 0:.53,1:.72,2:.48,3:.51,4:.49
2: [0], #[0], # 0:.60,
3: [1], #[0,1,2], # 0:.29,1:.42,2:.32 
4: [0,2,3,4,6,7,8], #[0,1,2,3,4,5,6,7,8], # 0:.59,1:.45,2:.66,3:.57,4:.82,5:.47,6:.80,7:.77,8:.68
}
ind2dataName = {
0:'bci3d3a', 
1:'bci3d4a', 
2:'bci3d4c', 
3:'bci3d5', 
4:'bci4d2a', 
}
dataName2ind={}
for k,v in ind2dataName.items():
  dataName2ind[v]=k

My_IP = get_my_ip()
lg = Logger()
lg.lg_list([__file__])

from_conf = get_conf_int(configfile,"from_conf")
from_tasklist = get_conf_int(configfile,"from_tasklist")

use_cache = get_conf_int(configfile,"use_cache")
if use_cache:
  try:
    import memcache # https://www.sinacloud.com/doc/_static/memcache.html
  except:
    use_cache = 0
if use_cache:
  CACHE_SIZE = 100*1024*1024*1024 # 10G
  CACHE_TIME=0
  cache_place = get_conf(configfile,"cache_place")
  My_IP = get_my_ip()
  if iprint: print("cache loc "+cache_place)
  if cache_place=='local':
    cache_prefix = get_conf(configfile,"cache_prefix")
    mc = memcache.Client(['127.0.0.1:11211'], server_max_value_length=CACHE_SIZE)
  elif cache_place=='servers':
    cache_prefix = get_conf(configfile,"cache_prefix")
    servers= get_conf(configfile,"cache_servers").split(",")
    if iprint: print("cache servers",servers)
    if My_IP.startswith("172.22"):
      ips = [tarekc2ip[host]+":11211" for host in servers]
    else:
      ips = [host+":11211" for host in servers]
    mc = memcache.Client(ips, server_max_value_length=CACHE_SIZE)

  elif cache_place=="lookup":
    mcdir = WORKDIR+"/mc/"
    datasets = get_list_startswith(configfile,"datasets")
    dataindex = int(sys.argv[1])
    dataname = datasets[dataindex]
    conf = confs%dataname
    cache_prefix = get_conf(conf,"cache_prefix")
    
    flist = glob.glob(mcdir+"/cache_server*")
    servers = []
    for i in range(len(flist)):
      serverip = read_lines_as_list(flist[i])
      servers.append(serverip[0])
      serverip=serverip[0]

      tmp=memcache.Client([serverip+":11211"])
      ret = tmp.set("tmp", 1 , time=10)
      val = tmp.get("tmp")
      if not (ret and val):
        print("-- Cache server fail: "+flist[i],serverip)
        mc_ind = flist[i].split(".")[0].lstrip(mcdir+"/cache_server")
        print("-- Run this cmd at login node:")
        print("jbsub -interactive -queue x86_7d -mem 40g -mail sh "+WORKDIR+"/memcached/start.sh "+mc_ind)
        sys.exit(0)

    if iprint: print(__file__.split("/")[-1],"cache_prefix=",cache_prefix,dataname,servers)
    mc = memcache.Client([host+":11211" for host in servers], server_max_value_length=CACHE_SIZE)

  def mc_get(subj, key, pref=cache_prefix):
    return mc.get(pref+"s%d"%subj+key)
  def mc_set(subj, key, val, time=CACHE_TIME, pref=cache_prefix):
    return mc.set(pref+"s%d"%subj+key, val , time=time)

  ret = mc_set(0,"try_if_it_is_working", 1, time=10)
  val = mc_get(0,"try_if_it_is_working")
  if not (ret and val):
    print("memcache not working, exit")
    sys.exit(0)

MetaDir = get_conf(configfile,"metafolder")
DataDir = get_conf(configfile,"datafolder")
regularize_coef=get_conf_float(configfile,"regularize_coef")
convertToSampleRate = get_conf_int(configfile,"convertToSampleRate")
mc_train_da_str = get_conf(configfile,"mc_train_da_str")
mc_train_lb_str = get_conf(configfile,"mc_train_lb_str")
mc_test_da_str = get_conf(configfile,"mc_test_da_str")
mc_test_lb_str = get_conf(configfile,"mc_test_lb_str")

batch_size = get_conf_int(conf ,"batch_size")
scatter_mc = get_conf_int(conf,"scatter_mc")
numClasses = get_conf_int(conf,"numClasses")
stateStep = get_conf_int(conf,"stateStep")
nUseChannels = get_conf_int(conf,"nUseChannels")
secondPerFrame = get_conf_float(conf,"secondPerFrame")
height = nUseChannels
width = int(convertToSampleRate*secondPerFrame)

## -------------------------- data feed below -----------------------------
if use_cache:
  trainSubj = ind2trainSubj[ dataName2ind[dataname] ]
  testSubj = ind2testSubj[ dataName2ind[dataname] ]
  if iprint: print("trainSubj",trainSubj)
  if iprint: print("testSubj",testSubj)

  intrain_all_data1= []
  intrain_all_label= []
  intest_all_data1=[]
  intest_all_label=[]

  storelist = [intrain_all_data1,intrain_all_label,intest_all_data1,intest_all_label]
  storekey = [mc_train_da_str,mc_train_lb_str,mc_test_da_str,mc_test_lb_str]

  for s in trainSubj:
    mc_num_per_key=mc_get(s,"mc_num_per_key")
    if iprint: print(__file__.split("/")[-1],"mc_num_per_key:",mc_num_per_key)
    for i in [0,1]:
      da= storelist[i]
      sz= mc_get(s,storekey[i]+"-max")
      assert(sz)
      cnt=0
      while cnt<=sz:
        ret = mc_get(s,storekey[i]+"%d"%cnt)
        cnt+=mc_num_per_key
        if ret is None:
          print(__file__.split("/")[-1], "memcache fail ... exit")
          sys.exit(0)
        else:
          if cnt%1000==0:
            print(__file__.split("/")[-1], "memcache read %s"%(storekey[i]+"%d"%cnt))
            # print(ret[0].shape) #(1, 2, 750)
        for item in ret:
          da.append(item)

  for s in testSubj:
    mc_num_per_key=mc_get(s,"mc_num_per_key")
    if iprint: print(__file__.split("/")[-1],"mc_num_per_key:",mc_num_per_key)
    for i in [2,3]:
      da= storelist[i]
      sz= mc_get(s,storekey[i]+"-max")
      assert(sz)
      cnt=0
      while cnt<=sz:
        ret = mc_get(s,storekey[i]+"%d"%cnt)
        cnt+=mc_num_per_key
        if ret is None:
          print(__file__.split("/")[-1], "memcache fail ... exit")
          sys.exit(0)
        else:
          if cnt%1000==0:
            print(__file__.split("/")[-1], "memcache read %s"%(storekey[i]+"%d"%cnt))
            # print(ret[0].shape) #(1, 2, 750)
        for item in ret:
          da.append(item)

  scatter_ratio = height//scatter_mc
  print("scatter_mc",scatter_mc,scatter_ratio,height)

  intrain_all_data1=np.asarray(intrain_all_data1)
  intrain_all_label=np.asarray(intrain_all_label)
  intest_all_data1=np.asarray(intest_all_data1)
  intest_all_label=np.asarray(intest_all_label)

  if scatter_mc>0:
    tmp_shape = intrain_all_data1.shape # [None*?, step, scatter_mc, time]
    intrain_all_data1 = np.reshape(intrain_all_data1,(tmp_shape[0]//scatter_ratio, tmp_shape[1], height, tmp_shape[3]))
    tmp_shape = intest_all_data1.shape # [None*?, step, scatter_mc, time]
    intest_all_data1 = np.reshape(intest_all_data1,(tmp_shape[0]//scatter_ratio, tmp_shape[1], height, tmp_shape[3]))

intrain_all_data1 = np.squeeze(intrain_all_data1)
intest_all_data1 = np.squeeze(intest_all_data1)

intrain_all_label = np.argmax(intrain_all_label,axis=-1)
intest_all_label = np.argmax(intest_all_label,axis=-1)

print(__file__.split("/")[-1],"train data:")
print(__file__.split("/")[-1], intrain_all_data1.shape)
print(__file__.split("/")[-1], intrain_all_label.shape)
print(__file__.split("/")[-1], intrain_all_label[0:15])
print(__file__.split("/")[-1],"test data:")
print(__file__.split("/")[-1], intest_all_data1.shape)
print(__file__.split("/")[-1], intest_all_label.shape)
print(__file__.split("/")[-1], intest_all_label[0:15])
train_size = intrain_all_data1.shape[0]
test_size = intest_all_data1.shape[0]
assert(intrain_all_data1.shape[0]>0)
assert(intrain_all_data1.shape[0]==intrain_all_label.shape[0])
assert(intest_all_data1.shape[0]>0)
assert(intest_all_data1.shape[0]==intest_all_label.shape[0])


# Create sklearn-compatible feature extraction and classification pipeline:
class CSP(base.BaseEstimator, base.TransformerMixin):
  def fit(self, X, y):
    class_covs = []

    # calculate per-class covariance
    for ci in np.unique(y): 
      class_covs.append(np.cov(np.hstack(X[y==ci])))
    assert len(class_covs) == 2

    # calculate CSP spatial filters
    self.W = eegtools.spatfilt.csp(class_covs[0], class_covs[1], 6)
    return self


  def transform(self, X):
    # Note that the projection on the spatial filter expects zero-mean data.
    return np.asarray([np.dot(self.W, trial) for trial in X])


class ChanVar(base.BaseEstimator, base.TransformerMixin):
  def fit(self, X, y): return self
  def transform(self, X):
    return np.var(X, axis=2)  # X.shape = (trials, channels, time)


pipe = pipeline.Pipeline(
  [('csp', CSP()), ('chan_var', ChanVar()), ('svm', svm.SVC(kernel='linear'))])


# train model
pipe.fit(intrain_all_data1,intrain_all_label)

# make predictions on unseen test data
y_pred = pipe.predict(intest_all_data1)

# Show results. Competition results are available on
# http://www.bbci.de/competition/iii/results/index.html#berlin1
print metrics.classification_report(intest_all_label, y_pred)
print metrics.accuracy_score(intest_all_label, y_pred)

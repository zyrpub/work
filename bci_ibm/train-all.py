#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
print(tf.__version__)

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
np.random.seed(1)

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
from namehostip import get_my_ip
from hostip import ip2tarekc,tarekc2ip
from readconf import get_conf,get_conf_int,get_conf_float,get_list_startswith,get_dic_startswith
from logger import Logger
from util import read_lines_as_list,read_lines_as_dic
from inputClass import inputClass,FeedInput

layers = tf.contrib.layers 
activation_layer = tf.nn.elu

iprint = 1
usr = os.path.expanduser("~")
if usr.endswith("shu"):
	WORKDIR = os.path.expanduser("~")
elif usr.endswith("srallap"):
	WORKDIR = os.path.expanduser("~")+"/eeg"

ind2trainSubj = {
0: [0], #[0,1,2], # 0:.42,1:.34,2:.28
1: [0,1,2,3,4], #[0,1,2,3,4], # 0:.53,1:.72,2:.48,3:.51,4:.49
2: [0], #[0], # 0:.60,
3: [1], #[0,1,2], # 0:.29,1:.42,2:.32 
4: [0,1,2,3,4,5,6,7,8], #[0,1,2,3,4,5,6,7,8], # 0:.59,1:.42,2:.66,3:.57,4:.83,5:.47,6:.80,7:.77,8:.68
}
ind2testSubj = {
0: [0], #[0,1,2], # 0:.38,1:.34,2:.28
1: [0,1,2,3,4], #[0,1,2,3,4], # 0:.53,1:.72,2:.48,3:.51,4:.49
2: [0], #[0], # 0:.60,
3: [1], #[0,1,2], # 0:.29,1:.42,2:.32 
4: [0,1,2,3,4,5,6,7,8], #[0,1,2,3,4,5,6,7,8], # 0:.59,1:.45,2:.66,3:.57,4:.82,5:.47,6:.80,7:.77,8:.68
}
useDataSets = [ 1,2 ]
# useDataSets= [int(i) for i in sys.argv[1].split(",")]

numDataSets = len(useDataSets)
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

configfile = "conf.txt"
confs = "conf/conf-%s.txt"

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
		
		cache_prefix={}
		conff={}
		for ind in useDataSets:
			dataname = ind2dataName[ind]
			if iprint: print(__file__.split("/")[-1],dataname)
			conff[ind] = confs%dataname
			cache_prefix[ind] = get_conf(conff[ind],"cache_prefix")

		flist = glob.glob(mcdir+"/cache_server*")
		servers = []
		for i in range(len(flist)):
			serverip = read_lines_as_list(flist[i])
			serverip=serverip[0]
			servers.append(serverip)

			tmp=memcache.Client([serverip+":11211"])
			ret = tmp.set("tmp", 1 , time=10)
			val = tmp.get("tmp")
			if not (ret and val):
				print("-- Cache server fail: "+flist[i],serverip)
				mc_ind = flist[i].split(".")[0].lstrip(mcdir+"/cache_server")
				print("-- Run this cmd at login node:")
				print("jbsub -interactive -queue x86_7d -mem 40g sh "+WORKDIR+"/memcached/start.sh "+mc_ind)
				sys.exit(0)

		if iprint: print('servers',servers)
		mc = memcache.Client([host+":11211" for host in servers], server_max_value_length=CACHE_SIZE)
		if iprint: print(__file__.split("/")[-1],"cache_prefix=",cache_prefix)

	def mc_get(ind,subj, key, pref=cache_prefix):
		return mc.get(pref[ind]+"s%d"%subj+key)
	def mc_set(ind,subj, key, val, time=CACHE_TIME, pref=cache_prefix):
		return mc.set(pref[ind]+"s%d"%subj+key, val , time=time)

	for ind in useDataSets:
		ret = mc_set(ind,0,"try_if_it_is_working", 1, time=10)
		val = mc_get(ind,0,"try_if_it_is_working")
		if not (ret and val):
			print("memcache not working, exit")
			sys.exit(0)

MetaDir = get_conf(configfile,"metafolder")
DataDir = get_conf(configfile,"datafolder")
convertToSampleRate = get_conf_int(configfile,"convertToSampleRate")
mc_train_da_str = get_conf(configfile,"mc_train_da_str")
mc_train_lb_str = get_conf(configfile,"mc_train_lb_str")
mc_test_da_str = get_conf(configfile,"mc_test_da_str")
mc_test_lb_str = get_conf(configfile,"mc_test_lb_str")

batch_size = get_conf_int(configfile,"batch_size")
regularize_coef=get_conf_float(configfile,"regularize_coef")

scatter_mc={}
numClasses={}
nUseChannels={}
for ind in useDataSets:
	conf = conff[ind]
	scatter_mc[ind] = get_conf_int(conf,"scatter_mc")
	numClasses[ind] = get_conf_int(conf,"numClasses")
	stateStep = get_conf_int(conf,"stateStep")
	nUseChannels[ind] = get_conf_int(conf,"nUseChannels")
	secondPerFrame = get_conf_float(conf,"secondPerFrame")
	width = int(convertToSampleRate*secondPerFrame)

lg.lg_list(["regularize_coef=",regularize_coef])
lg.lg_list(["batch_size=",batch_size])
lg.lg_list(["stateStep=",stateStep])
lg.lg_list(["secondPerFrame=",secondPerFrame])
lg.lg_list(["width=",width])
lg.lg_list(["nUseChannels=",nUseChannels])


SCOPE_CONV = "conv%d"
SCOPE_NORM = "norm%d"
SCOPE_DROP = "drop%d"
scope_cnt = 0
def scnt(inc=0, reset=False):
	global scope_cnt
	if reset:
		scope_cnt=0
	scope_cnt+=inc
	return scope_cnt

def batch_norm_layer(inputs, phase_train, scope=None):
	if phase_train:
		return layers.batch_norm(inputs, is_training=True, scale=True, 
			updates_collections=None, scope=scope)
	else:
		return layers.batch_norm(inputs, is_training=False, scale=True,
			updates_collections=None, scope=scope, reuse = True)


CONV_KEEP_PROB = 0.5

tmpnum=30
convNum1 = tmpnum
convNum2 = tmpnum
convNum3 = tmpnum
convNum4 = tmpnum
convNum5 = tmpnum
convNum6 = tmpnum
# convNum1 = 64
# convNum2 = 48
# convNum3 = 48
# convNum4 = 48
# convNum5 = 64
# convNum6 = 64
input_proj_dim = 25
fully_out_dim = 60
lg.lg_list(["convNum1=",convNum1])
lg.lg_list(["convNum2=",convNum2])
lg.lg_list(["convNum3=",convNum3])
lg.lg_list(["convNum4=",convNum4])
lg.lg_list(["input_proj_dim=",input_proj_dim])
lg.lg_list(["CONV_KEEP_PROB=",CONV_KEEP_PROB])
lg.lg_list(["fully_out_dim=",fully_out_dim])


data1={}
labels={}
for ind in useDataSets:
	data1[ind] = tf.placeholder(tf.float32, shape=[batch_size,stateStep,nUseChannels[ind],width])
	labels[ind] = tf.placeholder(tf.float32, shape=[batch_size,numClasses[ind]])
global_step = tf.Variable(0, trainable=False)

is_training = True

from cnn2d import cnn2d,cnn2d_pool,cnn_pool_split
from cnn3d import cnn3d,cnn3d_dilated,cnn3d_pool_split,cnn3d_pool
param_dict={}
param_dict["convNum1"]=convNum1
param_dict["convNum2"]=convNum2
param_dict["convNum3"]=convNum3
param_dict["convNum4"]=convNum4
param_dict["convNum5"]=convNum5
param_dict["convNum6"]=convNum6
param_dict["CONV_KEEP_PROB"]=CONV_KEEP_PROB
param_dict["OUT_DIM"]=fully_out_dim
cnn_param_dict = param_dict

inputind={}
for ind in useDataSets:
	inputs = tf.transpose(data1[ind] , perm=[0,1,3,2]) # [None,1,width,nUseChannels[ind]]
	with tf.variable_scope('input_layer',reuse=False):
		inputs = layers.fully_connected(inputs, input_proj_dim, activation_fn=activation_layer, scope='input%d'%ind)
		inputs_shape = inputs.get_shape().as_list()
		print('input transpose fully_connected',inputs_shape)
	inputind[ind] = tf.transpose(inputs, perm=[0,1,3,2]) # [None,1,nUseChannels[ind],width]

inputs = tf.concat(inputind.values(),axis=1) # [None,ind,nUseChannels[ind],width]
cnn_out = cnn3d_pool(inputs, cnn_param_dict, is_training) #(None, ind, 100)
# sys.exit(0)

last_layer = cnn_out
last_layer_dim=last_layer.get_shape().as_list()[-1]
print("last_layer.get_shape()",last_layer.get_shape().as_list())
last_layer = tf.split(last_layer,numDataSets,axis=1)
for i in range(len(last_layer)):
	last_layer[i]=tf.squeeze(last_layer[i])

out_W={}
logits={}
predict={}
loss_ent=0
with tf.variable_scope('last_layer'):
	cnt=0
	for ind in useDataSets:
		out_W[ind] = tf.get_variable("out_W%d"%ind, shape=[last_layer_dim, numClasses[ind]], initializer=tf.contrib.layers.variance_scaling_initializer())
		logits[ind] = tf.matmul(last_layer[cnt], out_W[ind])
		predict[ind] = tf.argmax(logits[ind], axis=1)
		loss_ent += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits[ind],labels=labels[ind]))
		cnt+=1


# ------------------- test data ------------------dropout false ---------
data1_t={}
for ind in useDataSets:
	data1_t[ind] = tf.placeholder(tf.float32, shape=[batch_size,stateStep,nUseChannels[ind],width])

is_training = False

inputind={}
for ind in useDataSets:
	inputs = tf.transpose(data1_t[ind] , perm=[0,1,3,2]) # [None,1,width,nUseChannels[ind]]
	with tf.variable_scope('input_layer',reuse=True):
		inputs = layers.fully_connected(inputs, input_proj_dim, activation_fn=activation_layer, scope='input%d'%ind)
		inputs_shape = inputs.get_shape().as_list()
		print('input transpose fully_connected',inputs_shape)
	inputind[ind] = tf.transpose(inputs, perm=[0,1,3,2]) # [None,1,nUseChannels[ind],width]

inputs = tf.concat(inputind.values(),axis=1) # [None,ind,nUseChannels[ind],width]
cnn_out = cnn3d_pool(inputs, cnn_param_dict, is_training) #(None, ens, 100)

last_layer = cnn_out
last_layer = tf.split(last_layer,len(useDataSets),axis=1)
for i in range(len(last_layer)):
	last_layer[i]=tf.squeeze(last_layer[i])

predict_t={}
loss_ent_t=0
cnt=0
for ind in useDataSets:
	logits[ind] = tf.matmul(last_layer[cnt], out_W[ind])
	predict_t[ind] = tf.argmax(logits[ind], axis=1)
	loss_ent_t+= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits[ind],labels=labels[ind]))
	cnt+=1


# ------------------- summary ----------------------
t_vars = tf.trainable_variables()
regularizers = 0.0
for var in t_vars:
	regularizers += tf.nn.l2_loss(var)
loss = loss_ent+ regularize_coef * regularizers


total_parameters = 0
for variable in tf.trainable_variables():
	# shape is an array of tf.Dimension
	shape = variable.get_shape().as_list()
	variable_parametes = 1
	for dim in shape:
		variable_parametes *= dim
	print(variable.name, shape, variable_parametes)
	total_parameters += variable_parametes
print("total_parameters",total_parameters)
lg.lg_list(["total_parameters=",total_parameters])


## -------------------------- data feed below -----------------------------
trainFeed={}
testFeed={}
num_batches=0
max_feed_ind=0 # who determine epoch num?
test_size ={}

for ind in useDataSets:
	print(ind, ind2dataName[ind])
	trainSubj = ind2trainSubj[ ind ]
	testSubj = ind2testSubj[ ind ]
	if iprint: print("trainSubj",trainSubj)
	if iprint: print("testSubj",testSubj)

	intrain_all_data1=[]
	intrain_all_label=[]
	intest_all_data1=[]
	intest_all_label=[]

	storelist = [intrain_all_data1,intrain_all_label,intest_all_data1,intest_all_label]
	storekey = [mc_train_da_str,mc_train_lb_str,mc_test_da_str,mc_test_lb_str]
	for s in trainSubj:
		mc_num_per_key=mc_get(ind,s,"mc_num_per_key")
		if iprint: print(__file__.split("/")[-1],"mc_num_per_key:",mc_num_per_key)

		for i in [0,1]:
			da= storelist[i]
			sz= mc_get(ind,s,storekey[i]+"-max")
			assert(sz)
			cnt=0
			while cnt<=sz:
				ret = mc_get(ind,s,storekey[i]+"%d"%cnt)
				cnt+=mc_num_per_key
				if ret is None:
					print(__file__.split("/")[-1], "memcache fail ... exit")
					sys.exit(0)
				else:
					if cnt%2000==0:
						print(__file__.split("/")[-1], "memcache read %s"%(storekey[i]+"%d"%cnt))
				for item in ret:
					da.append(item)
	for s in testSubj:
		mc_num_per_key=mc_get(ind,s,"mc_num_per_key")
		if iprint: print(__file__.split("/")[-1],"mc_num_per_key:",mc_num_per_key)

		for i in [2,3]:
			da= storelist[i]
			sz= mc_get(ind,s,storekey[i]+"-max")
			assert(sz)
			cnt=0
			while cnt<=sz:
				ret = mc_get(ind,s,storekey[i]+"%d"%cnt)
				cnt+=mc_num_per_key
				if ret is None:
					print(__file__.split("/")[-1], "memcache fail ... exit")
					sys.exit(0)
				else:
					if cnt%2000==0:
						print(__file__.split("/")[-1], "memcache read %s"%(storekey[i]+"%d"%cnt))
				for item in ret:
					da.append(item)

	
	scatter_ratio = nUseChannels[ind]//scatter_mc[ind]
	print("scatter_mc",scatter_mc,scatter_ratio,nUseChannels[ind])

	# need sp size % batch == 0:
	size = len(intrain_all_label)
	size_round = size//batch_size * batch_size
	if size>size_round:
		size_add = batch_size - size + size_round
	else:
		size_add = 0
	print("size",size,"train size_add",size_add)

	for i in range(size_add):
		n = np.random.randint(0,high=size)
		if scatter_mc[ind]>0:
			for j in range(n*scatter_ratio, (n+1)*scatter_ratio):
				intrain_all_data1.append(copy.deepcopy(intrain_all_data1[j]))
		else:
			intrain_all_data1.append(copy.deepcopy(intrain_all_data1[n]))
		intrain_all_label.append(copy.deepcopy(intrain_all_label[n]))

	size = len(intest_all_label)
	size_round = size//batch_size * batch_size
	if size>size_round:
		size_add = batch_size - size + size_round
	else:
		size_add = 0
	print("size",size,"test size_add",size_add)

	for i in range(size_add):
		n = np.random.randint(0,high=size)
		if scatter_mc[ind]>0:
			for j in range(n*scatter_ratio, (n+1)*scatter_ratio):
				intest_all_data1.append(copy.deepcopy(intest_all_data1[j]))
		else:
			intest_all_data1.append(copy.deepcopy(intest_all_data1[n]))
		intest_all_label.append(copy.deepcopy(intest_all_label[n]))
		
	intrain_all_data1=np.asarray(intrain_all_data1)
	intrain_all_label=np.asarray(intrain_all_label)
	intest_all_data1=np.asarray(intest_all_data1)
	intest_all_label=np.asarray(intest_all_label)

	if scatter_mc[ind]>0:
		tmp_shape = intrain_all_data1.shape # [None*?, step, scatter_mc, time]
		intrain_all_data1 = np.reshape(intrain_all_data1,(tmp_shape[0]//scatter_ratio, tmp_shape[1], nUseChannels[ind], tmp_shape[3]))
		tmp_shape = intest_all_data1.shape # [None*?, step, scatter_mc, time]
		intest_all_data1 = np.reshape(intest_all_data1,(tmp_shape[0]//scatter_ratio, tmp_shape[1], nUseChannels[ind], tmp_shape[3]))

	print(__file__.split("/")[-1],"train data:",[ind])
	print(__file__.split("/")[-1], intrain_all_data1.shape)
	print(__file__.split("/")[-1], intrain_all_label.shape)
	print(__file__.split("/")[-1], intrain_all_label[0:15])
	print(__file__.split("/")[-1],"test data:",[ind])
	print(__file__.split("/")[-1], intest_all_data1.shape)
	print(__file__.split("/")[-1], intest_all_label.shape)
	print(__file__.split("/")[-1], intest_all_label[0:15])
	test_size[ind] = intest_all_data1.shape[0]
	assert(intrain_all_data1.shape[0]>0)
	assert(intrain_all_data1.shape[0]==intrain_all_label.shape[0])
	assert(intest_all_data1.shape[0]>0)
	assert(intest_all_data1.shape[0]==intest_all_label.shape[0])

	trainFeed[ind] = FeedInput([intrain_all_data1, intrain_all_label],batch_size)
	trainFeed[ind].shuffle_all()
	testFeed[ind] = FeedInput([intest_all_data1, intest_all_label],batch_size)
	testFeed[ind].shuffle_all()
	tmp = trainFeed[ind].get_num_batches()
	if tmp>num_batches:
		num_batches=tmp
		max_feed_ind = ind
		print("num_batches per epoch",num_batches)

if iprint: print("test_size",test_size)
test_size = max(test_size.values())
if iprint: print("test_size",test_size)

## -------------------------- data feed finish -----------------------------


init_lr = get_conf_float(configfile,"init_lr")
end_lr = get_conf_float(configfile,"end_lr")
nepoch = get_conf_int(configfile,"nepoch") 
total_steps = nepoch*num_batches
decay_steps = num_batches/4.0
weight_decay_rate = (end_lr/init_lr)**(decay_steps/total_steps)
lg.lg_list(["init_lr=",init_lr])
lg.lg_list(["end_lr=",end_lr])
lg.lg_list(["nepoch=",nepoch])
lg.lg_list(["total_steps=",total_steps])
lg.lg_list(["decay_steps=",decay_steps])
lg.lg_list(["weight_decay_rate=",weight_decay_rate])
lg.flush()
print("nepoch",nepoch,"total_steps",total_steps,"decay_steps",decay_steps,"weight_decay_rate",weight_decay_rate)

lr = tf.train.exponential_decay(init_lr, global_step, decay_steps, weight_decay_rate, staircase=True)

optimizer = tf.train.RMSPropOptimizer(lr)
gvs = optimizer.compute_gradients(loss, var_list=t_vars)
capped_gvs = [(tf.clip_by_norm(grad, 1.0), var) for grad, var in gvs]
discOptimizer = optimizer.apply_gradients(capped_gvs, global_step=global_step) # inc global_step 
saver = tf.train.Saver() # must be in graph 

iteration = 0

with tf.Session(config=tf.ConfigProto(log_device_placement=False,allow_soft_placement=True)) as sess:
	print("graph_def",sess.graph_def.ByteSize()/1024.0,"KB")

	tf.global_variables_initializer().run()
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	while trainFeed[max_feed_ind].get_epoch()<nepoch:
		iteration+=1
		feed_dict = {}
		for ind in useDataSets:
			batch_inputs, batch_labels = trainFeed[ind].get_batch()
			feed_dict[data1[ind]]= batch_inputs 
			feed_dict[labels[ind]]= batch_labels

		runlist = []
		runlist.extend(labels.values())
		runlist.extend(predict.values())
		runlist.append(discOptimizer)
		runlist.append(loss)
		outlist = sess.run(runlist, feed_dict=feed_dict)

		if iteration % 10 == 0:
			lr_val = sess.run([lr])
			_accuracy=[]
			for i in range(numDataSets):
				_label = np.argmax(outlist[i], axis=1)
				_accuracy.append( np.mean(_label == outlist[i+numDataSets]) )
				lossV = outlist[-1]
				print('train loss %d'%numClasses[useDataSets[i]], lossV,'train accuracy', _accuracy[-1])
			print("epoch",trainFeed[max_feed_ind].get_epoch(),"batch",iteration,"lr %.4f"%lr_val[0])
			trainloss = lossV
			trainacc = (_accuracy)


		if trainFeed[max_feed_ind].get_epoch()>1 and iteration%200==0:
			dev_accuracy=[]
			dev_cross_entropy = []
			for ind in useDataSets:
				dev_accuracy.append([]) 
				dev_cross_entropy.append([]) 
			for eval_idx in xrange(int(test_size/batch_size)):
				feed_dict = {}
				for ind in useDataSets:
					batch_inputs, batch_labels = testFeed[ind].get_batch()
					feed_dict[data1_t[ind]]= batch_inputs 
					feed_dict[labels[ind]]= batch_labels

				runlist = []
				runlist.extend(labels.values())
				runlist.extend(predict_t.values())
				runlist.append(loss_ent_t)
				outlist = sess.run(runlist, feed_dict=feed_dict)
				for i in range(numDataSets):
					_label = np.argmax(outlist[i], axis=1)
					_accuracy = np.mean(_label == outlist[i+numDataSets])
					dev_accuracy[i].append(_accuracy)
					dev_cross_entropy[i].append(outlist[-1])
			for ind in useDataSets:
				print(ind2dataName[ind],numClasses[ind],'----test l=',np.mean(dev_cross_entropy),"a=",np.mean(dev_accuracy,axis=-1))
				testloss = np.mean(dev_cross_entropy)
				testacc = np.mean(dev_accuracy,axis=-1)
			lg.lg_list([trainFeed[max_feed_ind].get_epoch(),trainloss,trainacc,testloss,testacc])
			lg.flush()
			if iprint: print(My_IP)


	dev_accuracy = []
	dev_cross_entropy = []
	for ind in useDataSets:
		dev_accuracy.append([]) 
		dev_cross_entropy.append([]) 
	for eval_idx in xrange(int(test_size/batch_size)):
		feed_dict = {}
		for ind in useDataSets:
			batch_inputs, batch_labels = testFeed[ind].get_batch()
			feed_dict[data1_t[ind]]= batch_inputs 
			feed_dict[labels[ind]]= batch_labels

		runlist = []
		runlist.extend(labels.values())
		runlist.extend(predict_t.values())
		runlist.append(loss_ent_t)
		outlist = sess.run(runlist, feed_dict=feed_dict)
		for i in range(numDataSets):
			_label = np.argmax(outlist[i], axis=1)
			_accuracy = np.mean(_label == outlist[i+numDataSets])
			dev_accuracy[i].append(_accuracy)
			dev_cross_entropy[i].append(outlist[-1])
	for ind in useDataSets:
		print(ind2dataName[ind],numClasses[ind],'---test l=',np.mean(dev_cross_entropy),"a=",np.mean(dev_accuracy,axis=-1))
		testloss = np.mean(dev_cross_entropy)
		testacc = np.mean(dev_accuracy,axis=-1)
	lg.lg_list(["Final testloss",testloss,"testacc",testacc])
	lg.flush()


	save_path = saver.save(sess, "./model.ckpt")
	print("Saved: %s" % (save_path), My_IP)



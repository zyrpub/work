#!/usr/bin/env python

import os, sys
import subprocess
import random, time
import inspect, glob
import collections
import math
import datetime
from shutil import copy2, move as movefile
import getpass
mypydir =os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe())))
sys.path.append(mypydir)
sys.path.append(mypydir+"/mytools")
from namehostip import get_my_ip
from hostip import ip2tarekc,tarekc2ip
from readconf import get_conf,get_conf_int,get_conf_float,get_list_startswith,get_dic_startswith
from util import read_lines_as_list,read_lines_as_dic
from inputClass import inputClass

iprint = 1

HomeDir = os.path.expanduser("~")
User = getpass.getuser()

if HomeDir.endswith("shu"):
	WORKDIR = os.path.expanduser("~")
elif HomeDir.endswith("srallap"):
	WORKDIR = os.path.expanduser("~")+"/eeg"

configfile = "conf.txt"

import memcache # https://www.sinacloud.com/doc/_static/memcache.html
CACHE_TIME = 0 # or unix 1497190272, if larger than 30 days. 
CACHE_SIZE = 100*1024*1024*1024 # 100G, no use, use etc/config 
use_cache = get_conf_int(configfile,"use_cache")
if not use_cache:
	print(__file__.split("/")[-1], "did not plan to use cache, exit, edit conf")
	sys.exit(0)

cache_place = get_conf(configfile,"cache_place")

my_ip = get_my_ip()
print("my_ip",my_ip)
	
if cache_place=='local':
	mc = memcache.Client(['127.0.0.1:11211'], server_max_value_length=CACHE_SIZE)

elif cache_place=='servers':
	servers= get_conf(configfile,"cache_servers").split(",")
	if iprint: print("cache servers",servers)
	if my_ip.startswith("172.22"):
		ips = [tarekc2ip[host]+":11211" for host in servers]
	else:
		ips = [host+":11211" for host in servers]
	mc = memcache.Client(ips, server_max_value_length=CACHE_SIZE)

elif cache_place=="lookup":
	mcdir = WORKDIR+"/mc/"
	try:
		dataindex = int(sys.argv[1])
	except:
		print('cache_place=="lookup"',"one data set per job, [0,4], default 4 ")
		dataindex=4

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

	datasets = get_list_startswith(configfile,"datasets")
	dataname = datasets[dataindex]
	print(__file__.split("/")[-1],dataindex,datasets,dataname)
	mc = memcache.Client([tmp+":11211" for tmp in servers], server_max_value_length=CACHE_SIZE)

	conf = "conf/conf-"+dataname+".txt"
	cache_prefix = get_conf(conf,"cache_prefix")
	if iprint: print("cache_prefix=",cache_prefix,servers)



def mc_get(ind, key, pref=cache_prefix): # ind subj 
	return mc.get(pref+"s%d"%ind+key)
def mc_set(ind, key, val, time=CACHE_TIME, pref=cache_prefix):
	return mc.set(pref+"s%d"%ind+key, val , time=time)

if iprint: print("caching to "+cache_place)
ret = mc_set(0,"try_if_it_is_working", 1, time=10)
val = mc_get(0,"try_if_it_is_working")
if not (ret and val):
	print(__file__.split("/")[-1], "memcache not working, exit")
	sys.exit(0)


stateStep = get_conf_int(conf,"stateStep")
nUseChannels=get_conf_int(conf,"nUseChannels")
height=nUseChannels
mc_num_per_key=get_conf_int(conf,"mc_num_per_key")

mc_train_da_str = get_conf(configfile,"mc_train_da_str")
mc_train_lb_str = get_conf(configfile,"mc_train_lb_str")
mc_test_da_str = get_conf(configfile,"mc_test_da_str")
mc_test_lb_str = get_conf(configfile,"mc_test_lb_str")

try:
	intrain_all_data1= mc_get(0,mc_train_da_str+"0")
	intrain_all_label= mc_get(0,mc_train_lb_str+"0")
	if intrain_all_data1.shape[0]>0 and intrain_all_data1.shape[0]==intrain_all_label.shape[0]:
		print(__file__.split("/")[-1], "data already cached, exit")
		sys.exit(0)
except:
	print(__file__.split("/")[-1], "caching...")

label_dict = {
"bci3d3a":{
1:[1.0,0.0,0.0,0.0],
2:[0.0,1.0,0.0,0.0],
3:[0.0,0.0,1.0,0.0],
4:[0.0,0.0,0.0,1.0],
},
"bci3d4a":{
1:[1.0,0.0],
2:[0.0,1.0],
},
"bci3d4c":{
1:[1.0,0.0],
2:[0.0,1.0],
},
"bci3d5":{
2:[1.0,0.0,0.0],
3:[0.0,1.0,0.0],
7:[0.0,0.0,1.0],
},
"bci4d2a":{
1:[1.0,0.0,0.0,0.0],
2:[0.0,1.0,0.0,0.0],
3:[0.0,0.0,1.0,0.0],
4:[0.0,0.0,0.0,1.0],
},
}
params={}

dataSetsDir = get_conf(configfile,"dataSetsDir"+User) #dataSetsDir = WORKDIR+/bci/
params["dataProcDir"] = get_conf(configfile,"dataProcDir"+User)
params["MetaDir"] = get_conf(configfile,"metafolder")
params["DataDir"] = get_conf(configfile,"datafolder")
params["dataSetsDir"] = dataSetsDir
params["convertToSampleRate"] = get_conf_int(configfile,"convertToSampleRate")

trainlist = get_conf(conf, "trainfiles").split(",")
testlist = get_conf(conf, "testfiles").split(",")
params["secondPerFrame"] = get_conf_float(conf,"secondPerFrame")
params["sampleRate"] = get_conf_float(conf,"sampleRate")
params["label2softmax"] = label_dict[dataname]
params["numRawChannels"] = get_conf_int(conf, "numRawChannels")
params["shiftRatio"] = get_conf_float(conf, "shiftRatio")
params["augmentTimes"] = get_conf_int(conf,"augmentTimes")
params["init_window_size"] = min(375,get_conf_int(conf,"init_window_size"))
params["dataname"]=dataname
params["stateStep"]=stateStep
params["scatter_mc"]=get_conf_int(conf,"scatter_mc")
num_subj = get_conf_int(conf,"num_subj")

for s in range(num_subj):
	skey = "s%d"%s
	trainlist = get_list_startswith(conf,skey,",")
	testlist = get_list_startswith(conf,skey+"-t",",")
	if iprint: print(s,trainlist,testlist)

	intrain = inputClass(trainlist,params)
	tmpd1,tmplb = intrain.get_a_sample(stateStep)
	success=0
	while success==0:
		ret = mc_set(s,"tmpd1", tmpd1[0:mc_num_per_key] , time=20)
		if not (ret):
			mc_num_per_key-=1
			print(__file__.split("/")[-1], "decrease mc_num_per_key to %d "%mc_num_per_key)
			if mc_num_per_key<=0:
				print("memcache fails no matter how small entry is")
				sys.exit(0)
		else:
			success=1
			mc_set(s,"mc_num_per_key", mc_num_per_key, time=CACHE_TIME)
			print(__file__.split("/")[-1], "good mc_num_per_key = %d"%mc_get(s,"mc_num_per_key"))


	intrain.read_augmented_data_in_steps(stateStep)
	# if params["scatter_mc"]<=0:
	# 	intrain.shuffle_all()

	intest = inputClass(testlist,params)
	intest.read_augmented_data_in_steps(stateStep)
	# if params["scatter_mc"]<=0:
	# 	intest.shuffle_all()

	print(__file__.split("/")[-1], "train data:")
	print(__file__.split("/")[-1], intrain.all_data1.shape)
	print(__file__.split("/")[-1], intrain.all_label.shape)
	print(__file__.split("/")[-1], "test data:")
	print(__file__.split("/")[-1], intest.all_data1.shape)
	print(__file__.split("/")[-1], intest.all_label.shape)

	storelist = [intrain.all_data1,intrain.all_label,intest.all_data1,intest.all_label]
	storekey = [mc_train_da_str,mc_train_lb_str,mc_test_da_str,mc_test_lb_str]


	for i in range(len(storelist)):
		da= storelist[i]
		sz= da.shape[0]
		cnt=0
		while cnt+mc_num_per_key<=sz:
			ret = mc_set(s,storekey[i]+"%d"%cnt, da[cnt:cnt+mc_num_per_key] , time=CACHE_TIME)
			if not (ret):
				print(__file__.split("/")[-1], "memcache not storing %d ... exit"%cnt)
				sys.exit(0)

			cnt+=mc_num_per_key

		ret = mc_set(s,storekey[i]+"%d"%cnt, da[cnt:sz] , time=CACHE_TIME)
		mc_set(s,storekey[i]+"-max", cnt , time=CACHE_TIME)

		if not (ret):
			print(__file__.split("/")[-1], "memcache not storing last %d ... exit"%cnt)
			sys.exit(0)

print(__file__.split("/")[-1], "memcache store success")

if my_ip.startswith("9.47."):
	fd = open(WORKDIR+"/log_cache.txt","a")
	fd.write(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+" "+__file__.split("/")[-1]+" cache done!\n")
	fd.close()

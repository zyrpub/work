#!/usr/bin/env python

import os, sys
import subprocess
import random, time
import inspect

rootdir = '/u/yiran/bci/'

folders = [ 'bci3d3a','bci3d4a','bci3d4c','bci3d5','bci4d2a' ]

subf = ['/meta','/meta125','/data','/data125']

for folder in folders:
	for sub in subf:
		print("rm -rf "+rootdir+folder+sub)
		subprocess.call("rm -rf "+rootdir+folder+sub, shell=True)

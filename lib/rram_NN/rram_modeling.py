# file: train.py
# Author	: Abinash Mohanty
# Date		: 05/14/2017
# Project	: RRAM training NN

import numpy as np
from rram_NN.config import cfg
import os
import sys
import random
import math
import cPickle

def largest_indices(ary, n):
	""" 
	Helper function.
	Returns the n largest indices from a numpy array.
	"""
	flat = ary.flatten()
	indices = np.argpartition(flat, -n)[-n:]
	indices = indices[np.argsort(-flat[indices])]
	return np.unravel_index(indices, ary.shape)


def readVerifyTopN(rramTensors, topN, stddev, netName, datasetName):
	"""
	args:
		rramTensors: variables on which the read and verify operation is needed to be done.
		topN: Percentage of the top parameters on which read and verify needs to be done.
		stddev: stddev of variation in the parameters.
	"""
	if cfg.DEBUG_ALL:
		print '[',os.path.basename(sys.argv[0]),'] Reading and verifying top ', topN,'% of parameters'
	print 'readVerifyTopN - rram modeling| stddev', str(stddev) 	
	path = os.path.join('output',netName, 'variation', netName+'_'+datasetName+'_'+str(stddev)+'.pkl')
	print 'Loading variation models from ... ', path
	with open(path, 'rb') as fid:
		allParams = cPickle.load(fid)	
	for v in rramTensors:
		actualValue = v.eval()
		sh = v.get_shape().as_list()
		param = allParams[v.name]
		dim = len(sh)
		if dim == 1:
			n = int(math.ceil(sh[0]*topN/100.0))
		elif dim == 2:
			n = int(math.ceil(sh[0]*sh[1]*topN/100.0))
		elif dim == 4:
			n = int(math.ceil(sh[0]*sh[1]*sh[2]*sh[3]*topN/100.0))
		idx = largest_indices(np.absolute(actualValue), n)
		param[idx] = actualValue[idx]
		v.load(param)	


def readVariation(rramTensors, stddevVar=0.1):
	"""
	This function adds read variations to trained models.
	New params is randomly sampled from a normal distribution centered
	at the original value with standard deviation of percentVar
	"""
	if cfg.DEBUG_ALL:
		print '[',os.path.basename(sys.argv[0]),'] Adding Read Variations.'
	if stddevVar != 0.0:
		allParameters = [v.eval() for v in rramTensors]
		allShapes = [v.get_shape().as_list() for v in rramTensors]
		for i in range(len(allParameters)):
			if ('beta' not in rramTensors[i].name) or ('gamma' not in rramTensors[i].name):
				param = allParameters[i]*np.exp(np.random.normal(0, stddevVar, allShapes[i]))
				signMat = np.ones(param.shape, dtype=np.float32)
				signMat[np.where(param < 0.0)] = -1.0
				param = np.absolute(param)
				param[np.where(param > cfg.RRAM.SA0_VAL)] = cfg.RRAM.SA0_VAL
				param[np.where(param < cfg.RRAM.SA1_VAL)] = cfg.RRAM.SA1_VAL
				param = param*signMat
				rramTensors[i].load(param)		
			else:
				print 'Not adding read variation for ', rramTensors[i].name

def addDeviceVariation(rramTensors, stddevVar=0.1):
	"""
	This function adds write variations to trained models.
	New params is randomly sampled from a log normal distribution centered
	at the original value with standard deviation of stddevVar.
	W' = W.exp(N(0, stddev)), where N is the normal distribution with 0 mean and stddev standard deviation
	After adding the variations, the values are limitied between -1 and 1. 
	"""
	if cfg.DEBUG_ALL:
		print '[',os.path.basename(sys.argv[0]),'] Adding Device Level Variations.'
	if stddevVar != 0.0:
		allParameters = [v.eval() for v in rramTensors]
		allShapes = [v.get_shape().as_list() for v in rramTensors]
		for i in range(len(allParameters)):
			if ('beta' not in rramTensors[i].name) or ('gamma' not in rramTensors[i].name):
				param = allParameters[i]*np.exp(np.random.normal(0, stddevVar, allShapes[i]))
				signMat = np.ones(param.shape, dtype=np.float32)
				signMat[np.where(param < 0.0)] = -1.0
				param = np.absolute(param)
				param[np.where(param > cfg.RRAM.SA0_VAL)] = cfg.RRAM.SA0_VAL
				param[np.where(param < cfg.RRAM.SA1_VAL)] = cfg.RRAM.SA1_VAL
				param = param*signMat
				rramTensors[i].load(param)		
			else:
				print 'Not adding write variation for ', rramTensors[i].name

def addIRDrop(rramTensors):
	"""
	This function adds the defects due to IR Drop in the crossbar. 
	TODO
	"""	
	print '[',os.path.basename(sys.argv[0]),'] Adding IR Drop effects into crossbar. NOT YET IMPLEMENTED.'
		
def addSA1(rramTensors, percentSA1):
	"""
	This function adds the SAF low defects into the crossbar.		
	"""
	if cfg.DEBUG_ALL:
		print '[',os.path.basename(sys.argv[0]),'] Adding SA1 defects.'
	allParameters = [v.eval() for v in rramTensors]
	shapes = [v.get_shape().as_list() for v in rramTensors]
	minValues = []
	for i in range(len(allParameters)):
		minValues.append(np.amin(allParameters[i]))	
	lowVal = cfg.RRAM.SA1_VAL
	for i in range(len(allParameters)):
		if ('beta' not in rramTensors[i].name) or ('gamma' not in rramTensors[i].name):
			param = allParameters[i]
			dims = len(shapes)
			if dims == 1:
				num = int(math.ceil(shapes[i][0]*percentSA1/100.0))
				x = np.arange(shapes[i][0])
				random.shuffle(x)
				for j in range(num):
					param[x[j]] = lowVal if param[x[j]] > 0 else -1.0*lowVal
			elif dims == 2:
				num = int(math.ceil(shapes[i][0]*shapes[i][1]*percentSA1/100.0))
				x = np.arange(shapes[i][0])
				y = np.arange(shapes[i][1])
				random.shuffle(x)
				random.shuffle(y)
				for j in range(num):
					param[x[j], y[j]] = lowVal	if param[x[j], y[j]] > 0 else -1.0*lowVal		
			elif dims == 4:	
				num = int(math.ceil(shapes[i][0]*shapes[i][1]*shapes[i][2]*shapes[i][3]*percentSA1/100.0))
				x = np.arange(shapes[i][0])
				y = np.arange(shapes[i][1])
				z = np.arange(shapes[i][2])
				k = np.arange(shapes[i][3])
				random.shuffle(x)
				random.shuffle(y)
				random.shuffle(z)
				random.shuffle(k)
				for j in range(num):
					param[x[j], y[j], z[j], k[j]] = lowVal if param[x[j], y[j], z[j], k[j]] > 0 else -1.0*lowVal
			rramTensors[i].load(param)
		else:
			print 'not adding SA1 for ', rramTensors[i].name
		
def addSA0(rramTensors, percentSA0):
	"""
	This function adds the SAF high defects into the crossbar.		
	"""
	if cfg.DEBUG_ALL:
		print '[',os.path.basename(sys.argv[0]),'] Adding SA0 defects.'
	allParameters = [v.eval() for v in rramTensors]
	shapes = [v.get_shape().as_list() for v in rramTensors]
	maxValues = []
	for i in range(len(allParameters)):
		maxValues.append(np.amax(allParameters[i]))	
	highVal = cfg.RRAM.SA0_VAL
	for i in range(len(allParameters)):
		if ('beta' not in rramTensors[i].name) or ('gamma' not in rramTensors[i].name):
			param = allParameters[i]
			dims = len(shapes)
			if dims == 1:
				num = int(math.ceil(shapes[i][0]*percentSA0/100.0))
				x = np.arange(shapes[i][0])
				random.shuffle(x)
				for j in range(num):
					param[x[j]] = highVal
			elif dims == 2:
				num = int(math.ceil(shapes[i][0]*shapes[i][1]*percentSA0/100.0))
				x = np.arange(shapes[i][0])
				y = np.arange(shapes[i][1])
				random.shuffle(x)
				random.shuffle(y)
				for j in range(num):
					param[x[j], y[j]] = highVal			
			elif dims == 4:	
				num = int(math.ceil(shapes[i][0]*shapes[i][1]*shapes[i][2]*shapes[i][3]*percentSA0/100.0))
				x = np.arange(shapes[i][0])
				y = np.arange(shapes[i][1])
				z = np.arange(shapes[i][2])
				k = np.arange(shapes[i][3])
				random.shuffle(x)
				random.shuffle(y)
				random.shuffle(z)
				random.shuffle(k)
				for j in range(num):
					param[x[j], y[j], z[j], k[j]] = highVal	
			rramTensors[i].load(param)	
		else:
			print 'not adding SA0 for ', rramTensors[i].name
		
def addSAFs(rramTensors, percentSA0, percentSA1):
	"""		
	This function adds the stuck at faults. 
	Internally calls addSAF1 (stuck at fault high) and addSAF0 (stuck at fault low).		
	"""	
	addSA0(rramTensors, percentSA0)
	addSA1(rramTensors, percentSA1)	
		

def quantize(rramTensors, levels=32):
	if cfg.DEBUG_ALL:
		print '[',os.path.basename(sys.argv[0]),'] Truncating parameters to emulate crossbar with level : ', levels
	allParameters = [v.eval() for v in rramTensors]
	allShapes = [v.get_shape().as_list() for v in rramTensors]
	for i in range(len(allParameters)):
		param = allParameters[i]
		signMat = np.ones(param.shape, dtype=np.float32)
		signMat[np.where(param < 0.0)] = -1.0
		param = np.absolute(param)
		param[np.where(param < cfg.RRAM.SA1_VAL)] = 0.0
		param = (cfg.RRAM.SA0_VAL-cfg.RRAM.SA1_VAL)*np.ceil(param*levels)/levels + cfg.RRAM.SA1_VAL
		param[np.where(param > cfg.RRAM.SA0_VAL)] = cfg.RRAM.SA0_VAL
		param[np.where(param < cfg.RRAM.SA1_VAL)] = cfg.RRAM.SA1_VAL
		param = param*signMat
		rramTensors[i].load(param)		
		
# def quantize_v1(rramTensors, levels=32):
	# if cfg.DEBUG_ALL:
		# print '[',os.path.basename(sys.argv[0]),'] Truncating parameters to emulate crossbar with level : ', levels
	# allParameters = [v.eval() for v in rramTensors]
	# allShapes = [v.get_shape().as_list() for v in rramTensors]
	# for i in range(len(allParameters)):
		# param = allParameters[i]
		# signMat = np.ones(param.shape, dtype=np.float32)
		# signMat[np.where(param < 0.0)] = -1.0
		# param = np.absolute(param)
		# param = np.ceil(param*levels)/levels;
		# param[np.where(param > cfg.RRAM.SA0_VAL)] = cfg.RRAM.SA0_VAL
		# param[np.where(param < cfg.RRAM.SA1_VAL)] = cfg.RRAM.SA1_VAL
		# param = param*signMat
		# rramTensors[i].load(param)	
		
# def quantize_v0(rramTensors, levels=32):
	# if cfg.DEBUG_ALL:
		# print '[',os.path.basename(sys.argv[0]),'] Truncating parameters to emulate crossbar with levels = ', levels
	# numLevels = levels
	# allParameters = [v.eval() for v in rramTensors]
	# for i in range(len(allParameters)):
		# param = np.floor(allParameters[i]*numLevels)/numLevels;
		# rramTensors[i].load(param)	
	
def addDefects(rramTensors, stddevVar=0.5, levels=32, percentSA0=cfg.RRAM.SA0, percentSA1=cfg.RRAM.SA1):
	quantize(rramTensors, levels)
	addIRDrop(rramTensors)
	addDeviceVariation(rramTensors, stddevVar)
	addSAFs(rramTensors, percentSA0, percentSA1)
	
	

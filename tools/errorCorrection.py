# file: errorCorrection.py
# Author	: Abinash Mohanty
# Date		: 06/01/2017
# Project	: RRAM training NN

import _init_paths
#from tensorflow.examples.tutorials.mnist import input_data
from networks.factory import get_network
from datasets.factory import get_dataset
from rram_NN.train import train_net, verifyTopN, train_net_v1
from rram_NN.test import eval_net
import numpy as np
from rram_NN.config import get_output_dir
import os
import sys
import matplotlib.pyplot as plt
import cPickle
import argparse
import tensorflow as tf

iters = {}
iters['mnist'] = 40000
iters['mlp_1'] = 50000
iters['mlp_2'] = 50000
#iters['cifar10'] = 150000
iters['cifar10'] = 40000

writeToPickle = True	

def createModelWithVariation(percentRetrainable, stddevVar, network_name, dataset_name, num_levels=32):
	tf.reset_default_graph()
	dataset = get_dataset(dataset_name)	
	network = get_network(network_name)
	max_iters = iters[dataset_name]
	output_dir = get_output_dir()
	acc = train_net(network, dataset, output_dir, max_iters, stddevVar, percentRetrainable, dataset_name, num_levels)
	return acc
	
def quantization_test(network_name, dataset_name):	
	percentRetrainable = 5.0
	accuracy = []
	stddevVar = 0.5
	accuracy = []
	levels = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
	numExp = 1
	path = 'experiments/out/quantization/'+network_name+'_'+dataset_name+'_qsf.csv'
	with open(path,'wb+') as fid:
		for num_levels in levels:	
			for i in range(numExp):
				acc = createModelWithVariation(percentRetrainable, stddevVar, network_name, dataset_name, num_levels)	
				fid.write(str(acc))
				if(i != numExp-1):
					fid.write(',')
				else:
					fid.write('\n')
	"""	
	for num_levels in levels:	
		acc = 0
		for _ in range(numExp):
			acc += createModelWithVariation(percentRetrainable, stddevVar, network_name, dataset_name, num_levels)
		acc = acc/float(numExp)	
		print ' ------------------------------- ', acc
		accuracy.append(acc)
	plt.plot(levels, accuracy, 'xb-')
	plt.ylabel('Basline model accuracy')
	plt.xlabel('Number of levels in RRAM device.')
	plt.show()	
	"""	
		
def accuracyVsPercentVar_Baseline(network_name, dataset_name):
	"""
	Experiment to determine the maximum variation that can be rectified by re-training
	This method assumes that only 5% of the parameters are re-trained. 
	percentVar used here is the standard deviation for the distribution centered around the 
	baseline parameter value.
	"""	
	percentRetrainable = 5.0
	accuracy = []
	accuracy_1 = []
	stddevVar = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, \
					1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
	num_levels = 32			
	numExp = 1
	"""
	path = 'experiments/out/accuracyVsPercentVar_Baseline/'+network_name+'_'+dataset_name+'_'+str(num_levels)+'_vqs.csv'
	with open(path, 'wb+') as fid:
		for stddev in stddevVar:
			for i in range(numExp):
				acc =  createModelWithVariation(percentRetrainable, stddev, network_name, dataset_name, num_levels)
				fid.write(str(acc))
				if(i != numExp-1):
					fid.write(',')
				else:
					fid.write('\n')
	"""
					
	for stddev in stddevVar:
		acc = 0.0
		print '[',os.path.basename(sys.argv[0]),'] standard deviation of variations in parameters = ', stddev
		for i in range(numExp):	
			acc += createModelWithVariation(percentRetrainable, stddev, network_name, dataset_name, num_levels)
		accuracy.append(acc/numExp)	

	plt.plot(stddevVar, accuracy, 'xb-')
	plt.ylabel('Basline model accuracy')
	plt.xlabel('standard deviation of variations in parameters')
	plt.show()		

def accuracyVsRetrainable(network_name, dataset_name):
	""" 
	Experiment for determining the minimum ratio of re-trainable
	"""
	percentRetrainable = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
	#percentRetrainable = [5.0]
	accuracy = []
	stddevVar = 0.8
	num_levels = 32
	path = 'experiments/out/accuracyVsRetrainable/'+network_name+'_'+dataset_name+'_1.csv'
	numExp = 1
	with open(path, 'wb+') as fid:
		for per in percentRetrainable:
			print '[',os.path.basename(sys.argv[0]),'] % of retrainable parameters = ', per
			acc = 0.0
			acc1 = 0.0
			fid.write(str(per)+',')
			for i in range(numExp):	
				acc1 = createModelWithVariation(per, stddevVar, network_name, dataset_name, num_levels)
				acc += acc1
				fid.write(str(acc1))
				if(i != numExp-1):
					fid.write(',')
				else:
					fid.write('\n') 
			acc = acc/numExp	
			accuracy.append(acc)

	if writeToPickle:
		result = []
		metaData = 'accuracy, percentRetrainable, stddevVar,'+network_name+','+dataset_name+', num_levels'
		result.append(metaData)
		result.append(accuracy)
		result.append(percentRetrainable)
		result.append(stddevVar)
		result.append(num_levels)
		path = 'experiments/out/accuracyVsRetrainable/'+network_name+'_'+dataset_name+'_'+str(num_levels)+'_1.pkl'
		with open(path,'wb') as fid:
			cPickle.dump(result, fid, cPickle.HIGHEST_PROTOCOL)			
	plt.plot(percentRetrainable, accuracy, 'xb-')
	plt.ylabel('Accuracy')
	plt.xlabel('% of parameters re-trained')
	plt.show()		

def accuracyVsIterations(network_name, dataset_name):
	percentRetrainable = [1.0, 2.0, 4.0, 5.0, 6.0, 7.0 ] 
					#8.0, 9.0, 1.0, 11.0, 12.0, 13.0, 14.0, 15.0]
	#per = 10.0 
	stddevVar = 0.8
	num_levels = 32
	tf.reset_default_graph()
	dataset = get_dataset(dataset_name)	
	network = get_network(network_name)
	max_iters = iters[dataset_name]
	output_dir = get_output_dir()
	path = 'experiments/out/accuracyVsIterations/'+network_name+'_'+dataset_name
	acc, iterations = train_net_v1(network, dataset, output_dir, max_iters, stddevVar, per, dataset_name, num_levels)
	"""
	path1 = path + str(int(per*10))+'.csv'
	with open(path1, 'wb+') as fid:
		for i in range(len(acc)):
			fid.write(str(iterations[i])+','+str(acc[i])+'\n')
	"""
	for per in percentRetrainable:
		path1 = path + str(int(per*10))+'.csv'
		acc, iterations = train_net_v1(network, dataset, output_dir, max_iters, stddevVar, per, dataset_name, num_levels)
		with open(path1, 'wb+') as fid:
			for i in range(len(acc)):
				fid.write(str(iterations[i])+','+str(acc[i])+'\n')
	"""
	plt.plot(iterations, acc, 'xb-')
	plt.ylabel('Accuracy')
	plt.xlabel('Number of batches in training')
	plt.show()	
	"""
	
def readVerifyTopN(network_name, dataset_name):
	stddevVar = 0.8
	num_levels = 32
	tf.reset_default_graph()
	dataset = get_dataset(dataset_name)	
	network = get_network(network_name)
	output_dir = get_output_dir()
	topNs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, \
			 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, \
			 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, \
			 31.0, 32.0, 33.0, 33.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, \
			 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, \
			 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, \
			 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, \
			 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, \
			 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, \
			 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0 ]
	accuracy = []
	p = 'experiments/out/readVerifyTopN/'+network_name+'_'+dataset_name+'.csv'
	with open(p, 'wb+') as fid:
		for topN in topNs:
			print 'Doing read and verify for top {} of parameters'.format(topN)
			print 'readVerifyTopN | stddev', str(stddevVar) 
			acc = verifyTopN(network, dataset, output_dir, stddevVar, dataset_name, num_levels, topN)
			fid.write(str(acc)+'\n')
			accuracy.append(acc)

	if writeToPickle:
		result = []
		metaData = 'accuracy, read Verified %, stddevVar,'+network_name+','+dataset_name+', num_levels'
		result.append(metaData)
		result.append(accuracy)
		result.append(topNs)
		result.append(stddevVar)
		result.append(num_levels)
		path = 'experiments/out/readVerifyTopN/'+network_name+'_'+dataset_name+'_'+str(num_levels)+'_.pkl'
		with open(path,'wb') as fid:
			cPickle.dump(result, fid, cPickle.HIGHEST_PROTOCOL)		

	plt.plot(topNs, accuracy, 'xb-')
	plt.ylabel('Accuracy')
	plt.xlabel('% of parameters read_verified')
	plt.show()	

def getAllParameters(netName, datasetName):
	
	path_baseline = 'output/'+netName+'/baseline/'+netName+'_'+datasetName+'.pkl'
	path_variation = 'output/'+netName+'/variation/'+netName+'_'+datasetName+'_0.5.pkl'
	
	with open(path_baseline, 'r') as fid:
		param_baseline = cPickle.load(fid)
		
	with open(path_variation, 'r') as fid:
		param_variation = cPickle.load(fid)
		
	outputPath = 'experiments/out/weightDistribution/'
	
	keys = param_baseline.keys()
	for key in keys:
		param_baseline[key] = param_baseline[key].flatten()
		param_variation[key] = param_variation[key].flatten()
		opPath = outputPath+'baseline_'+netName+key.split('/')[0]+'_'+key.split('/')[1].split(':')[0]+'.csv'
		opPath1 = outputPath+'variation_'+netName+key.split('/')[0]+'_'+key.split('/')[1].split(':')[0]+'.csv'
		#"""
		with open(opPath, 'wb+') as fid:
			print key, ' -- ' , param_baseline[key].shape
			for i in range(len(param_baseline[key])):
				fid.write(str(param_baseline[key][i])+'\n')
		#"""
		"""		
		with open(opPath1, 'wb+') as fid:
			print key, ' -- ' , param_variation[key].shape
			for i in range(len(param_variation[key])):
				fid.write(str(param_variation[key][i])+'\n')		
		"""	
	
if __name__ == '__main__':

	netName = 'cifarCNN'
	datasetName = 'cifar10'
	#createModelWithVariation(5.0, 0.5, netName, datasetName, 32)
	#quantization_test(netName, datasetName)
	#accuracyVsPercentVar_Baseline(netName, datasetName)
	#accuracyVsRetrainable(netName, datasetName)
	accuracyVsIterations(netName, datasetName)
	#readVerifyTopN(netName, datasetName)
	#getAllParameters(netName, datasetName)

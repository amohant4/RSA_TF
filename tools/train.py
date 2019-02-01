# file: train.py
# Author	: Abinash Mohanty
# Date		: 05/05/2017
# Project	: RRAM training NN

import _init_paths
from networks.factory import get_network
from datasets.factory import get_dataset
from rram_NN.train import train_net
import numpy as np
from rram_NN.config import cfg, get_output_dir
import os
import sys
import matplotlib.pyplot as plt
import cPickle
import argparse
import tensorflow as tf

def parse_args():
	"""
	parse the inputs.
	--dataset = data directory containing MNIST
	"""
	parser = argparse.ArgumentParser(description="RRAM Train")
	parser.add_argument('--dataset', dest='dataset', help='dataset to use',
						default='mnist', type=str)
	parser.add_argument('--iters', dest='max_iters',
						help='number of iterations to train',
						default=50000, type=int)
	parser.add_argument('--network', dest='network',
						help='name of the network',
						default='leNet', type=str)
	parser.add_argument('--write_pickle', dest='writePickle',
						help='write to pickle',
						default=0, type=int)
	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(1)
	args = parser.parse_args()
	return args

def crossBarTests(dataset_name, max_iters, network_name, output_dir, stddev=0.1, percentRetrain=5.0):
    tf.reset_default_graph()
    network = get_network(network_name)
    dataset = get_dataset(dataset_name)
    acc = train_net(network,dataset, output_dir, max_iters, stddevVar, percentRetrainable, dataset_name)
    return acc

if __name__ == '__main__':
	args = parse_args()
	dataset	=   args.dataset
	max_iters	=   args.max_iters
	network 	=   args.network
	output_dir	=   get_output_dir()
	print 'Use network `{:s}` in training'.format(args.network)
	print 'Use dataset `{:s}`'.format(args.dataset)
	stddevVar = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,0, 1.1, 1.2, 1.3, 1.4, 1.5]
	percentRetrainable = [5.0]
	accuracy = []
	for stddev in stddevVar:
    		print 'Experimenting for stddev = ', stddev
		acc = 0.0
		for j in range(10):
			print 'iter = ', j 
			acc += crossBarTests(dataset, max_iters, network, output_dir, stddev, percentRetrainable) 
		acc = acc/10.0		
		accuracy.append(acc)

	if args.writePickle == 1:
		result=[]
		result.append('accuracy,stddevVar,mnist,lenet')
		result.append(accuracy)
		result.append(stddevVar)
		with open('experiments/out/accuracyVspercentvar/accuracyVspercentvar_mnist_lenet_withSA.pkl','wb') as fid:
			cPickle.dump(result, fid, cPickle.HIGHEST_PROTOCOL)
	
	plt.plot(stddevVar, accuracy, 'g^-')
	plt.ylabel('Retrained Network accuracy')
	plt.xlabel('Standard deviation of variation')
	plt.show()

# file: test.py
# Author	: Abinash Mohanty
# Date		: 05/05/2017
# Project	: RRAM training NN

import numpy as np
import cPickle

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


network_name = 'cifarCNN'
dataset_name = 'cifar10'
num_levels = 32

path = 'experiments/out/accuracyVsPercentVar_Baseline/'+network_name+'_'+dataset_name+'_'+str(num_levels)+'_.pkl'

with open(path, 'r') as fid:
	result = cPickle.load(fid)
	for i in range(len(result[1])):
		for j in range(len(result[1][0])):
			fid.write(str(result[1][i][j]))
			fid.write(',')	
		fid.write('\n')	






# file: cifar10_train.py
# Author	: Abinash Mohanty
# Date		: 07/07/2017
# Project	: RRAM training NN

import numpy as np
import os.path as osp
from datasets.cifar10_base import cifar10_base
import tensorflow as tf

class cifar10_train(cifar10_base):
	def __init__(self):
		self._num_classes = 10
		self._height = 32
		self._width = 32
		self._num_channels = 3
		self._dataset_path = self._get_default_path()
		self._fileNames = ['data_batch_1', 'data_batch_2', \
						   'data_batch_3', 'data_batch_4', 'data_batch_5']
		self._classes = self._load_names()
		self._datasetSize= 50000
		self._currIdx = 0	
		self._images, self._labels = self._loadData()
		self._shuffle_inds()

	def _shuffle_inds(self):
		"""
		Randomly shuffle the training set.
		"""
		self._currIdx = 0
		self._perm_indx = np.random.permutation(np.arange(self._datasetSize))
	
	def _get_next_batch_inds(self, batchSize):
		"""
		Return indices of num randomly selected images.
		"""
		if (self._currIdx + batchSize >= self._datasetSize):
			self._shuffle_inds()
		inds = self._perm_indx[self._currIdx:self._currIdx+batchSize]
		self._currIdx += batchSize
		return inds	

	def next_batch(self, batchSize):
		"""
		Return num randomly selected train images.
		"""
		images = np.zeros(shape=[batchSize, self._height, self._width, self._num_channels], dtype=np.float32)
		labels = np.zeros(shape=[batchSize, self._num_classes], dtype=int)
		inds = self._get_next_batch_inds(batchSize)
		j = 0	
		for i in inds:
			labels[j,:] = self._labels[i,:] 
			images[j,:] = self._images[i,:]
			j += 1
		return [images, labels]

		
















	

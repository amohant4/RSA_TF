# file: cifar10_base.py
# Author	: Abinash Mohanty
# Date		: 07/07/2017
# Project	: RRAM training NN

import os
import os.path as osp
import cPickle
import numpy as np
from rram_NN.config import cfg
import tensorflow as tf
import matplotlib.pyplot as plt

class cifar10_base():
	def __init__(self):
		self._num_classes = None
		self._classes = None
		self._dataset_path = self._get_default_path()
		assert osp.exists(self._dataset_path), \
			'Cifar10 dataset path dosenot exist: {}'.format(self._dataset_path)
								
	def _get_default_path(self):
		"""
		Return the default path where cifar10 data is expected
		"""
		return osp.join(cfg.DATA_DIR, 'cifar-10-batches-py')

	def loadCIFAR(self, path):
		with open(path, 'rb') as fid:
			batch = cPickle.load(fid)
		return batch

	def _load_names(self):
		raw = self.loadCIFAR(osp.join(self._dataset_path, 'batches.meta'))['label_names']
		names = [x.decode('utf-8') for x in raw]
		return names

	def _one_hot_encoded(self, class_numbers, num_classes=None):
		if num_classes is None:
			num_classes = np.max(class_numbers)+1
		return np.eye(num_classes, dtype=float)[class_numbers]

	def _loadData(self):
		images = np.zeros(shape=[self._datasetSize, self._height, self._width, self._num_channels], dtype=np.float32)
		labels = np.zeros(shape=[self._datasetSize], dtype=int)
		begin = 0
		for files in self._fileNames:
			path = osp.join(self._dataset_path, files)
			data = self.loadCIFAR(path)
			raw_labels = np.array(data['labels'])
			raw_images = data['data']
			#raw_images_float = np.array(raw_images, dtype=np.float32) / 255.0
			#images_batch = raw_images_float.reshape([-1, self._num_channels, self._height, self._width ])
			raw_images_float = np.array(raw_images, dtype=np.float32)	# Added
			images_batch = raw_images_float.reshape([-1, self._num_channels, self._height, self._width ])
			images_batch = images_batch.transpose([0,2,3,1])
			end = begin + len(images_batch)
			images[begin:end, :] = images_batch
			labels[begin:end] = raw_labels
			begin = end
		return images, self._one_hot_encoded(labels, self._num_classes)
				

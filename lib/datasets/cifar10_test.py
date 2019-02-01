# file: cifar10_test.py
# Author	: Abinash Mohanty
# Date		: 07/07/2017
# Project	: RRAM training NN

import numpy as np
import os.path as osp
from datasets.cifar10_base import cifar10_base
import tensorflow as tf
import matplotlib.pyplot as plt

class cifar10_test(cifar10_base):
	def __init__(self):
		self._num_classes = 10
		self._height = 32
		self._width = 32
		self._num_channels = 3		
		self._dataset_path = self._get_default_path()
		self._fileNames = ['test_batch']
		self._classes = self._load_names()
		self._datasetSize = 10000
		self.images, self.labels = self._load_testData()

	def _load_testData(self):
		images, labels = self._loadData()
		return images, labels
		
		

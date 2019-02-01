# file: cifar10.py
# Author	: Abinash Mohanty
# Date		: 07/07/2017
# Project	: RRAM training NN

from datasets.cifar10_test import cifar10_test
from datasets.cifar10_train import cifar10_train

class cifar10():
	def __init__(self):
		self.name='cifar10'
		self.train = cifar10_train()
		self.test = cifar10_test()

	


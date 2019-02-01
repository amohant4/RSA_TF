# --------------------------------------------------------
# Cough Detection NN
# Arizona State Univerity
# Written by Abinash Mohanty
# --------------------------------------------------------

"""Factory method for easily getting database by name."""
import numpy as np
from datasets.cifar10 import cifar10
from tensorflow.examples.tutorials.mnist import input_data
__sets = {}
__sets['cifar10'] = (lambda : cifar10())
__sets['mnist'] = (lambda : input_data.read_data_sets('data/MNIST', one_hot=True, reshape=False))

def get_dataset(name):
	""" 
	Get an cough dataset by name. 
	"""
	if not __sets.has_key(name):
		raise KeyError('Unknown dataset: {}'.format(name))
	return __sets[name]()

def list_datasets():
	"""
	List all registered datasets
	"""
	return __sets.keys()




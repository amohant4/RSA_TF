# file: factory.py
# Author	: Abinash Mohanty
# Date		: 05/08/2017
# Project	: RRAM training NN

""" Factory method to get the approriate network. """

from networks.leNet import leNet
from networks.cifarCNN import cifarCNN
from networks.cifarCNN_sram import cifarCNN_sram
from networks.leNet_sram import leNet_sram
from networks.mlp_2 import mlp_2
from networks.mlp_1 import mlp_1

__sets = {}
__sets['leNet']=(lambda : leNet())
__sets['leNet_sram']=(lambda : leNet_sram())
__sets['cifarCNN']=(lambda : cifarCNN())
__sets['cifarCNN_sram']=(lambda : cifarCNN_sram())
__sets['mlp_1']=(lambda : mlp_1())
__sets['mlp_2']=(lambda : mlp_2())

def get_network(name):
	"""
	Get a network by name.
	"""
	if not __sets.has_key(name):
		raise KeyError('Unknown network architecture: {}'.format(name))
	return __sets[name]()

def list_networks():
	"""
	List all implemented networks
	"""
	return __sets.keys()


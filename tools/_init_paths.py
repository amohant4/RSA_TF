# Initialize the paths to library. Adds the lib folder to the python path
# Author	: Abinash Mohanty
# Date 		: 05/08/2017
# Project	: RRAM Training for NN

import os.path as osp
import sys

def add_path(path):
	"""
	This function adds path to python path. 
	"""
	if path not in sys.path:
		sys.path.insert(0,path)

this_dir = osp.dirname(__file__)
lib_path = osp.join(this_dir, '..','lib')
add_path(lib_path)


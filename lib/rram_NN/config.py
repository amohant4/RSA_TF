# file: config.py
# Author	: Abinash Mohanty
# Date		: 05/05/2017
# Project	: RRAM training NN

"""
Configuration file for RRAM training. 
"""

from easydict import EasyDict as edict
import os
import os.path as osp

__C = edict()
__C.TRAIN = edict()
__C.TEST = edict()
__C.RRAM = edict()

cfg = __C

__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
__C.DATA_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..', 'data'))
__C.RNG_SEED = 3

# DEBUG CONFIGURATIONS
__C.DEBUG_LEVEL_1 = 0
__C.DEBUG_TRAINING = 1
__C.DEBUG_ALL = 0
__C.WRITE_TO_SUMMARY = 0

# TRAINING CONFIGURATIONS
__C.TRAIN.BASELINE_ITERS = 150000
__C.TRAIN.SNAPSHOT_ITERS = 5000
__C.TRAIN.LEARNING_RATE = 0.01 
#__C.TRAIN.LEARNING_RATE = 0.1 
__C.TRAIN.NUM_EPOCHS_PER_DECAY = 150
__C.TRAIN.TRAIN_BATCH_SIZE = 128
#__C.TRAIN.DECAY_STEPS = int(20000) 
#__C.TRAIN.DECAY_STEPS = int(50000) 
__C.TRAIN.DECAY_STEPS = int(25000) 
__C.TRAIN.DECAY_RATE = 0.1
__C.TRAIN.WEIGHT_DECAY = 0.0005 
#__C.TRAIN.WEIGHT_DECAY = 0.0 

# RAM MODELING CONFIGURATIONS
__C.RRAM.NUM_RRAM_LEVELS = 64.0
__C.RRAM.RON_OVER_ROFF = 100
__C.RRAM.SA0 = 1.75
__C.RRAM.SA1 = 9.04
__C.RRAM.SA0_VAL = 1.0
__C.RRAM.SA1_VAL = __C.RRAM.SA0_VAL / __C.RRAM.RON_OVER_ROFF
__C.RRAM.PERCENTAGE_READ_VARIATION = 30.0
__C.RRAM.PERCENTAGE_WRITE_VARIATION = 10.0

def get_output_dir():
	"""
	Return the directory where the experimental artifacts are placed.
	If the directory does not exit, it is created.
	"""
	outdir=osp.abspath(osp.join(__C.ROOT_DIR, 'output'))
	return outdir

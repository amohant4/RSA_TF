# file: leNet_sram.py
# Author	: Abinash Mohanty
# Date		: 06/22/2017
# Project	: RRAM training NN

# This class implements leNet. 
# It is a convolution neural network with 2 conv and 3 fc layers. 
# For more details:

import tensorflow as tf
from rram_NN.config import cfg
from networks.network import Network

n_classes = 10

class leNet_sram(Network):
	def __init__(self, trainable=True):
		self.inputs = []
		self.x = tf.placeholder(tf.float32, shape=[None, 784]) 
		self.y_ = tf.placeholder(tf.float32, shape=[None, 10])
		self.keep_prob = tf.placeholder(tf.float32)
		self.layers = dict({'x': self.x, 'y_': self.y_, 'keep_prob':self.keep_prob})
		self.trainable = trainable
		self.name = "leNet"
		self.setup()
		self._accuracy = None
		self._global_step = None
		self._lr = None
		self._gradients = None
		self._optimizer = None

	def setup(self):
		(self.feed('x')
			 .reshape_leNet(name='reshape_to_image')
			 .pad_leNet(name='pad_2_leNet_compatible')
			 .conv(5, 5, 6, 1 ,1, name='conv_1', relu=True, trainable=False))
		(self.feed('pad_2_leNet_compatible')
			 .conv(5, 5, 6, 1 ,1, name='conv_1_sram', relu=True, trainable=True))
		(self.feed('conv_1','conv_1_sram')
			 .sum_fc_ops(name='conv1_op')
			 .max_pool(2, 2, 2 ,2, name='max_pool_1')
			 .conv(5, 5, 16, 1, 1, name='conv_2', relu=True, trainable=False))
		(self.feed('max_pool_1')
			 .conv(5, 5, 16, 1, 1, name='conv_2_sram', relu=True, trainable=True))
		(self.feed('conv_2', 'conv_2_sram')
			 .sum_fc_ops(name='conv2_op')
			 .max_pool(2, 2, 2, 2, name='max_pool_2') 
			 .fc(120, name='fc_1', relu=True, trainable=False))
		(self.feed('max_pool_2')
			 .fc(120, name='fc_1_sram', relu=True, trainable=True))
		(self.feed('fc_1', 'fc_1_sram')
			 .sum_fc_ops(name='fc1_op')
			 .fc(84, name='fc_2', relu=True, trainable=False))
		(self.feed('fc1_op')
			 .fc(84, name='fc_2_sram', relu=True, trainable=True))
		(self.feed('fc_2','fc_2_sram')
			 .sum_fc_ops(name='fc2_op')
			 .fc(n_classes, name='class_pred', relu=False, trainable=False))
		(self.feed('fc2_op')
			 .fc(n_classes, name='fc_3_sram', relu=False, trainable=True))
		(self.feed('class_pred','fc_3_sram')
			 .sum_fc_ops(name='class_pred_sram')	
		)

	@property
	def accuracy(self):
		"""
		Computes accuracy of the network
		"""	
		if self._accuracy is None:
			y = self.get_output('class_pred_sram')
			y_ = self.get_output('y_')		
			correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
			self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
			if cfg.DEBUG_TRAINING or cfg.DEBUG_ALL:
				tf.summary.scalar('accuracy', self._accuracy)
		return self._accuracy	

	@property
	def global_step(self):
		"""
		Function to ensure that the global_step is not created
		many times during experiments.
		"""
		if self._global_step is None:
			 self._global_step = tf.Variable(0, trainable=False, name='global_step')
		return self._global_step

	@property
	def optimizer(self):
		"""
		Optimizer used to minimize error.	
		"""		
		if self._optimizer is None:
			lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE, self.global_step, 
						cfg.TRAIN.DECAY_STEPS, cfg.TRAIN.DECAY_RATE, staircase=True, name='lr')
			if cfg.DEBUG_TRAINING or cfg.DEBUG_ALL:			
				tf.summary.scalar('lr', lr)			
			self._optimizer = tf.train.GradientDescentOptimizer(lr)
		return self._optimizer

	@property
	def gradients(self):
		"""
		Computes gradients !
		"""
		if self._gradients is None:
			y = self.get_output('class_pred_sram')
			y_ = self.get_output('y_')
			cross_entropy = tf.reduce_mean(
				tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
			if cfg.DEBUG_TRAINING or cfg.DEBUG_ALL:	
				tf.summary.scalar('cross_entropy', cross_entropy)	
			vars = tf.trainable_variables()
			self._gradients = tf.gradients(cross_entropy, vars)			
		return self._gradients

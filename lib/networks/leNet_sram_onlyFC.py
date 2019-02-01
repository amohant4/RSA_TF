# file: factory.py
# Author	: Abinash Mohanty
# Date		: 06/21/2017
# Project	: RRAM training NN

import tensorflow as tf
from networks.network import Network

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
		self.pretrainedLayers = ['conv_1','conv_2','fc_1','fc_2']
		self._loss = None
		self._accuracy = None

	def setup(self):
		(self.feed('x')
			 .reshape_leNet(name='reshape_to_image')
			 .pad_leNet(name='pad_2_leNet_compatible')
			 .conv(5, 5, 6, 1 ,1, name='conv_1', relu=True, trainable=False)
			 .max_pool(2, 2, 2 ,2, name='max_pool_1')		
			 .conv(5, 5, 16, 1, 1, name='conv_2', relu=True, trainable=False)
			 .max_pool(2, 2, 2 ,2, name='max_pool_2')
			 .fc(120, name='fc_1', relu=True, trainable=False)
			 .fc(84, name='fc_2', relu=True, trainable=False))
		(self.feed('fc_1')
			 .fc(84, name='fc_2_sram', relu=True, trainable=True))
		(self.feed('fc_2','fc_2_sram')
			 .sum_fc_ops(name='fc2_op')
			 .fc(10, name='class_pred', relu=False, trainable=False))
		(self.feed('fc2_op')
			 .fc(10, name='fc_3_sram', relu=False, trainable=True))
		(self.feed('class_pred','fc_3_sram')
			 .sum_fc_ops(name='class_pred_sram'))

	@property
	def loss(self):
		"""	
		loss is the parameter passed to the optimizer over which the gradient is 
		computed. As loss of different networks can be different. loss is implemented
		in the networks  call. 
		The module using the network will have to call the loss property of this class as:
		loss = network.loss
		This also ensures that multiple graphs for loss is not created. 
		"""	
		if self._loss is None:
			y = self.get_output('class_pred_sram')
			y_ = self.get_output('y_')	
			cross_entropy = tf.reduce_mean(
				tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
			self._loss = cross_entropy
		return self._loss

	@property
	def accuracy(self):
		"""
		Computes accuracy of the network
		"""	
		if self._accuracy is None:
			y = self.get_output('class_pred_sram')
			y_ = self.get_output('y_')		
			correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
			acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
			self._accuracy = acc
		return self._accuracy	

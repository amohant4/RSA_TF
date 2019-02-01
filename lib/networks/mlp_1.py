# file: mlp_2_train.py
# Author	: Abinash Mohanty
# Date		: 05/08/2017
# Project	: RRAM training NN

import tensorflow as tf
from networks.network import Network
from rram_NN.config import cfg

n_classes = 10

class mlp_1(Network):
	def __init__(self, trainable=True):
		self.inputs = []
		self.x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1]) 
		self.y_ = tf.placeholder(tf.float32, shape=[None, 10])
		self.keep_prob = tf.placeholder(tf.float32)
		self.phase = tf.placeholder(tf.int32, name='phase')		# phase = 1 for train 0 for test
		self.layers = dict({'x': self.x, 'y_': self.y_, 'phase': self.phase,'keep_prob':self.keep_prob})
		self.trainable = trainable
		self.input_height = 20	
		self.input_width = 20	
		self.setup()
		self.name = "mlp_1"
		self._accuracy = None	
		self._global_step = None
		self._lr = None
		self._gradients = None	
		self._optimizer = None
		self._loss = None

	def setup(self):
		(self.feed('x')
			 .reshape_and_resize(self.input_height, self.input_width)	
			 .fc(100, name='fc_1', relu=True, trainable=True)
			 .fc(n_classes, name='class_pred', relu=False, trainable=True))

	@property
	def accuracy(self):
		"""
		Computes accuracy of the network
		"""	
		if self._accuracy is None:
			y = self.get_output('class_pred')
			y_ = self.get_output('y_')		
			correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
			self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
			if cfg.WRITE_TO_SUMMARY or cfg.DEBUG_ALL:
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
	def loss(self):
		if self._loss is None:
			y = self.get_output('class_pred')
			y_=self.get_output('y_')
			cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
				labels=y_, logits=y, name='cross_entropy'))
			self._loss = cross_entropy
			if cfg.TRAIN.WEIGHT_DECAY > 0:
				regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
				self._loss = tf.add_n(regularization_losses) + self._loss
			if cfg.WRITE_TO_SUMMARY or cfg.DEBUG_ALL:	
				tf.summary.scalar('total loss', self._loss)
		return self._loss		

	@property
	def optimizer(self):
		"""
		Optimizer used to minimize error.	
		"""		
		if self._optimizer is None:
			lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE, self.global_step, 
						cfg.TRAIN.DECAY_STEPS, cfg.TRAIN.DECAY_RATE, staircase=True, name='lr')
			if cfg.WRITE_TO_SUMMARY or cfg.DEBUG_ALL:
				tf.summary.scalar('lr', lr)			
			self._optimizer = tf.train.GradientDescentOptimizer(lr)
		return self._optimizer	

	@property
	def gradients(self):
		"""
		Computes gradients !
		"""
		if self._gradients is None:
			vars = tf.trainable_variables()
			self._gradients = tf.gradients(self.loss, vars)			
		return self._gradients

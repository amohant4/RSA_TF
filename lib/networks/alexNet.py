# file: alexNet.py
# Author	: Abinash Mohanty
# Date		: 06/15/2017
# Project	: RRAM training NN
# Network architecture for alexNet

import tensorflow as tf
from networks.network import Network

n_classes = 10

class alexNet(Network):
	def __init__(self, trainable=True):
		self.inputs = []
		self.x = tf.placeholder(tf.float32, shape=[None, 1024])
		self.y_ = tf.placeholder(tf.float32, shape=[None, 10])
		self.keep_prob = tf.placeholder(tf.float32)
		self.layers = dict({'x':self.x, 'y_':self.y_, 'keep_prob':self.keep_prob})
		self.trainable = trainable
		self.setup()
		self.name="alexNet"

	def setup(self):
		"""
		Create the neural network in tensorflow. 
		"""
		(self.feed('x')
			.conv(11,11,96,4,4,padding='VALID',name='conv1', ,trainable=self.trainable)
			.lrn(2, 2e-05,0.75,name='norm1')
			.max_pool(3,3,2,2,padding='VALID', name='pool1')
			.conv(5,5,256,1,1,group=2, name='conv2')
			.lrn(2,2e-05,0.75,name='norm2')
			.max_pool(3,3,2,2,padding='VALID', name='pool2')
			.conv(3,3,384,1,1,name='conv3')
			.conv(3,3,384,1,1,group=2, name='conv4')
			.conv(3,3,256,1,1,group=2, name='conv5')
			.fc(4096, name='fc6')
			.fc(4096, name='fc7')
			.fc(n_classes, relu=False, name='fc8')
			.softmax(name='prob'))

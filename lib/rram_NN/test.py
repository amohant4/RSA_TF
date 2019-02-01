# file: test.py
# Author	: Abinash Mohanty
# Date		: 05/11/2017
# Project	: RRAM training NN

import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from rram_NN.config import cfg
from rram_NN.rram_modeling import addDefects

def eval_net(network, dataset, weights):
	"""
	Evaluates the accuracy of a trained net.
	"""
	accuracy = network.accuracy
	saver = tf.train.Saver()
	sess = tf.InteractiveSession()
	saver.restore(sess, weights)
	netAccuracy = accuracy.eval(feed_dict={
									network.x: dataset.test.images,
									network.y_: dataset.test.labels,
									network.keep_prob : 1.0})

	print 'Model test accuracy : {}'.format(netAccuracy)
	return netAccuracy


# def test_net(network, dataset, weights):
	# """ 
	# tests a network. inputs:
	# network as tensorflow graph, dataset, pretrained weights
	# """
	# accuracy = network.accuracy
	# saver = tf.train.Saver()
	# sess = tf.InteractiveSession()
	# saver.restore(sess, weights)
	# print("Original Model test accuracy : %g"%accuracy.eval(feed_dict={
                                        # network.x: dataset.test.images,
                                        # network.y_: dataset.test.labels,
										# network.keep_prob : 1.0}))

	# allParameters = [v.eval() for v in tf.trainable_variables()]
	# allparameter_tensors = [v for v in tf.trainable_variables()]

	# netAccuracy = []
	# for exp in range(50):
		# percentVar = exp*5.0
		# truncatedParams = readVariation(allParameters, percentVar)
		# for i in range(len(allparameter_tensors)):
			# allparameter_tensors[i].load(truncatedParams[i], sess)	
		# netAccuracy.append(accuracy.eval(feed_dict={
											# network.x: dataset.test.images,
											# network.y_: dataset.test.labels,
											# network.keep_prob : 1.0}))
	# x = np.arange(0. , 50*5.0 , 5.0)
	# plt.plot(x , netAccuracy, 'xb-')
	# plt.ylabel('Accuracy')
	# plt.xlabel('Write Variation %')
	# plt.show()

#	netAccuracy = []
#	for exp in range(50):
#		percentVar = exp*5.0
#		netOut = []
#		for i in range(len(dataset.test.images)):
#			readVarParams = readVariation(truncatedParams, percentVar)	
#			for j in range(len(allparameter_tensors)):
#				allparameter_tensors[j].load(readVarParams[j], sess)
#			netOut.append(accuracy.eval(feed_dict={
#					network.x : [dataset.test.images[i,:,:,:]],
#					network.y_ : [dataset.test.labels[i,:]],
#					network.keep_prob : 1.0}))	
#		netAccuracy.append(sum(netOut)/len(netOut))
#		print 'Model accuracy with read variation {} : {}'.format(percentVar , sum(netOut)/len(netOut))	
#
#	x = np.arange(0. , 50*5.0 , 5.0)
#	plt.plot(x , netAccuracy, 'xb-')
#	plt.ylabel('accuracy')
#	plt.xlabel('% variation')
#	plt.show()


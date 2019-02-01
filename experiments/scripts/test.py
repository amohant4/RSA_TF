"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
import argparse
import sys
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
import random


def readVariation(allParameters, percentVar):
	"""
	This function adds read variations to trained models.
	"""
	modifiedParams = []
	for i in range(len(allParameters)):
		param = allParameters[i]
		for j in range(len(param)):
			param[j] = np.random.normal(param[j], percentVar*param[j]/100.0)	
		modifiedParams.append(param)
	return modifiedParams

def test_readVar():
	allParameters = []
	allParameters.append([1., 20. , 40., 50.])
	for m in range(len(allParameters)):
		print allParameters[m]
	print '~~~~~~~'
	newParameters = readVariation(allParameters, 10.0)
	for n in range(len(newParameters)):
		print newParameters[n]

def test():
	mnist = input_data.read_data_sets("../../data/MNIST", one_hot=True, reshape=False)
	sess = tf.InteractiveSession()
	batch_x, batch_y = mnist.train.next_batch(10)
	x = tf.placeholder(tf.float32, [None, 28, 28, 1])

	#images = tf.convert_to_tensor(x, name="images")
	images_ = tf.image.resize_images(x, [20 , 20])
	
	sess = tf.InteractiveSession()
	im = images_.eval(feed_dict={x : batch_x})

if __name__ == '__main__':
	test_readVar()
	#x = [1.  ,1. ,1. ,1. ,1.]
 	#np.random.seed(10)	
	#for i in range(len(x)):
	#	print np.random.normal(x[i], 0.1*x[i])



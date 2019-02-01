# file: train.py
# Author	: Abinash Mohanty
# Date		: 05/10/2017
# Project	: RRAM training NN

import tensorflow as tf
import os 
from rram_NN.config import cfg
from rram_NN.rram_modeling import truncate, writeVariation, readVariation
import numpy as np

class SolverWrapper(object):
	def __init__(self, sess, saver, dataset, network, output_dir):
		"""
		SolverWrapper constructor. Inputs are: 
			tensorflow session, tensorflow saver, dataset, network, output directory.
		"""
		self.net = network
		self.dataset = dataset
		self.output_dir = output_dir
		self.saver = saver
		self.pretrained_model = os.path.join(self.output_dir, self.net.name, 'baseline', \
					self.net.name + '_iter_{:d}'.format(cfg.BASELINE_ITERS) + '.ckpt')

	def _snapshot(self, sess, iter, mode=1):
		"""
		Writes snapshot of the network to file in output directory.
		inputs: tensorflow session, current iteration number, mode.
		mode :	0 = baseline model
				1 = with variation model
				2 = retrained model to rectify variation	
		"""
		if not os.path.exists(self.output_dir):
			os.makedirs(self.output_dir)
		if not os.path.exists(os.path.join(self.output_dir, self.net.name)):
			os.makedirs(os.path.join(self.output_dir, self.net.name))
		if not os.path.exists(os.path.join(self.output_dir, self.net.name, 'baseline')):
			os.makedirs(os.path.join(self.output_dir, self.net.name, 'baseline'))
		if not os.path.exists(os.path.join(self.output_dir, self.net.name, 'variation')):
			os.makedirs(os.path.join(self.output_dir, self.net.name, 'variation'))

		prefix = ''
		if mode == 2:
			prefix='retrained_'
		elif mode == 1:
			prefix='withVariation_'

		filename = prefix + self.net.name + '_iter_{:d}'.format(iter+1) + '.ckpt'
		if mode == 0:
			filename = os.path.join(self.output_dir, self.net.name, 'baseline', filename)
		elif mode == 1:
			filename = os.path.join(self.output_dir, self.net.name,'variation', filename)
		elif mode == 2:
			filename = os.path.join(self.output_dir, self.net.name, filename)
		self.saver.save(sess, filename)
		print 'Wrote snapshot to: {:s}'.format(filename)

	def _train_model_software_baseline(self, sess, max_iters):
		"""
		Trains a model and implements the network training loop. 
		Inputs: tensorflow session, maximum number of iterations.
		This is the software base line code that trains using float32
		"""
		y = self.net.get_output('class_pred')
		y_ = self.net.get_output('y_')	
		cross_entropy = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

		train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
		correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		sess.run(tf.global_variables_initializer())
		for iter in range(max_iters):
			batch = self.dataset.train.next_batch(50)
			if cfg.DEBUG_TRAINING or cfg.DEBUG_ALL:
				if (iter+1) % 100 == 0:
					train_accuracy = accuracy.eval(feed_dict={self.net.x : batch[0], 
															  self.net.y_ : batch[1],
															  self.net.keep_prob : 1.0})
					print("Step : %d, training accuracy : %g"%(iter, train_accuracy))

			feed_dict = {self.net.x : batch[0], self.net.y_ : batch[1], self.net.keep_prob : 0.5}
			train_step.run(feed_dict=feed_dict)
		self._snapshot(sess, iter, 0)

	def _create_mask(self, percentRetrainable):
		"""
		Function to create a random mask to stop gradient flow through specific 
		prameters of the network while retraining the network. 
		This creates a list of ndarrays with values equal to 0/1 and dimention
		same as the variables in the net. 
		"""
		if cfg.DEBUG_LEVEL_1 or cfg.DEBUG_ALL:
			print 'Creating masks for stoping random gradient with \
					retention ratio = {}'.format(percentRetrainable)
		allShapes = [v.get_shape() for v in tf.trainable_variables()]
		masks = []
		for i in range(len(allShapes)):
			mask = np.random.rand(*allShapes[i])
			mask = np.where(mask < percentRetrainable, 1., 0.)	
			masks.append(mask)

		return masks

    def _find_or_train_baseline(self, sess):
        """
        Function looks for the baseline software models. 
        Incase it doesn't find that, it call the _train_model_software_baseline() to create baseline models. 
        """
		filename = self.net.name + '_iter_{:d}'.format(cfg.BASELINE_ITERS) + '.ckpt'
		if not os.path.isfile(os.path.join(self.output_dir, self.net.name,'baseline', filename + '.index')):
			print 'Baseline software models not found. Training software baseline'	
			self._train_model_software_baseline(sess, cfg.BASELINE_ITERS)
			self.pretrained_model = os.path.join(self.output_dir, self.net.name, 'baseline', filename)


	def _train_model_rram_variation(self, sess, max_iters, percentVar=20.0, percentRetrainable=0.05):
		"""
		Function to find max variation that can be rectified. 
		"""
		self._find_or_train_baseline(sess)

		y = self.net.get_output('class_pred')
		y_ = self.net.get_output('y_')	
		cross_entropy = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
		correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		sess.run(tf.global_variables_initializer())

		print ('Loading pretrained model weights from {:s}').format(self.pretrained_model)
		self.net.load(self.pretrained_model, sess, self.saver, True)

		baseline_accuracy = accuracy.eval(feed_dict={self.net.x : self.dataset.test.images, 
													self.net.y_ : self.dataset.test.labels,
													self.net.keep_prob : 1.0})
		print("Software baseline accuracy : %g"%(baseline_accuracy))
		print("Adding Write Variation ... = %g"%(percentVar))
		allParameters = [v.eval() for v in tf.trainable_variables()]
		allparameter_tensors = [v for v in tf.trainable_variables()]
		truncatedParams = readVariation(allParameters, percentVar)
		for i in range(len(allparameter_tensors)):
			allparameter_tensors[i].load(truncatedParams[i], sess)
			
		variation_accuracy = accuracy.eval(feed_dict={self.net.x : self.dataset.test.images, 
													self.net.y_ : self.dataset.test.labels,
													self.net.keep_prob : 1.0})
		print("Model with write variation accuracy : %g"%(variation_accuracy))

		print("Creating mask and retraining few parameters ... ")
		masks = self._create_mask(percentRetrainable)	
		optimizer = tf.train.GradientDescentOptimizer(0.01)
		grads = tf.gradients(cross_entropy, tf.trainable_variables())
		for i in range(len(grads)):
			grads[i] = grads[i]*masks[i]
		grads = list(zip(grads, tf.trainable_variables()))
		train_step = optimizer.apply_gradients(grads_and_vars=grads)
		
		last_snapshot_iter = -1
		for iter in range(max_iters):
			batch = self.dataset.train.next_batch(50)
			if cfg.DEBUG_TRAINING or cfg.DEBUG_ALL:	
				if (iter+1) % 100 == 0:
					train_accuracy = accuracy.eval(feed_dict={self.net.x : batch[0], 
															  self.net.y_ : batch[1],
															  self.net.keep_prob : 1.0})
					print("Step : %d, training accuracy : %g"%(iter, train_accuracy))

			feed_dict = {self.net.x : batch[0], self.net.y_ : batch[1], self.net.keep_prob : 0.5}
			train_step.run(feed_dict=feed_dict)	

			if (iter+1) % cfg.SNAPSHOT_ITERS == 0:
				last_snapshot_iter = iter
				self._snapshot(sess, iter, 2)	

		if last_snapshot_iter != iter:
			self._snapshot(sess, iter, 2)	

		return [baseline_accuracy, variation_accuracy]

	def _train_model_rram_epochs(self, sess, max_iters, percentVar=20.0, percentRetrainable=0.05):
		"""
		Function to find max variation that can be rectified. 
		"""
		print '_train_model_rram_epochs called for training ... '
        self._find_or_train_baseline(sess)
			
		y = self.net.get_output('class_pred')
		y_ = self.net.get_output('y_')	
		cross_entropy = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
		correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		sess.run(tf.global_variables_initializer())

		print ('Loading pretrained model weights from {:s}').format(self.pretrained_model)
		self.net.load(self.pretrained_model, sess, self.saver, True)

		baseline_accuracy = accuracy.eval(feed_dict={self.net.x : self.dataset.test.images, 
													self.net.y_ : self.dataset.test.labels,
													self.net.keep_prob : 1.0})
		print("Software baseline accuracy : %g"%(baseline_accuracy))
		print("Adding Write Variation ... = %g"%(percentVar))
		allParameters = [v.eval() for v in tf.trainable_variables()]
		allparameter_tensors = [v for v in tf.trainable_variables()]
		truncatedParams = readVariation(allParameters, percentVar)
		for i in range(len(allparameter_tensors)):
			allparameter_tensors[i].load(truncatedParams[i], sess)
			
		variation_accuracy = accuracy.eval(feed_dict={self.net.x : self.dataset.test.images, 
													self.net.y_ : self.dataset.test.labels,
													self.net.keep_prob : 1.0})
		print("Model with write variation accuracy : %g"%(variation_accuracy))

		print("Creating mask and retraining few parameters ... ")
		masks = self._create_mask(percentRetrainable)	
		optimizer = tf.train.GradientDescentOptimizer(0.01)
		grads = tf.gradients(cross_entropy, tf.trainable_variables())
		for i in range(len(grads)):
			grads[i] = grads[i]*masks[i]
		grads = list(zip(grads, tf.trainable_variables()))
		train_step = optimizer.apply_gradients(grads_and_vars=grads)
		
		iterations = []
		netAccuracy = []
		baseline = []
		degraded_accuracy = variation_accuracy
		for iter in range(max_iters):
			batch = self.dataset.train.next_batch(100)
			feed_dict = {self.net.x : batch[0], self.net.y_ : batch[1], self.net.keep_prob : 0.5}
			train_step.run(feed_dict=feed_dict)	
			if (iter+1)%100 == 0:
				train_accuracy = accuracy.eval(feed_dict={self.net.x : self.dataset.test.images, 
														self.net.y_ : self.dataset.test.labels,
														self.net.keep_prob : 1.0})
				print 'Net accuracy at iteration {} is {}'.format(iter, train_accuracy)
				netAccuracy.append(train_accuracy)
				iterations.append(iter+1)
				baseline.append(baseline_accuracy)

		return [netAccuracy, iterations, baseline, degraded_accuracy]

	def _train_model_rram(self, sess, max_iters, percentVar=20.0, percentRetrainable=0.5):
		"""
		Trains a model and implements the network training loop. 
		Inputs: tensorflow session, maximum number of iterations.
		This considers rram models and creates more realistic models
		# TODO : Does not converge at the moment. Look into it if the hardware 
		is to support on-chip training. 
		"""
        self._find_or_train_baseline(sess)
			
		y = self.net.get_output('class_pred')
		y_ = self.net.get_output('y_')	
		cross_entropy = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
		correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		sess.run(tf.global_variables_initializer())

		print ('Loading pretrained model weights from {:s}').format(self.pretrained_model)
		self.net.load(self.pretrained_model, sess, self.saver, True)

		baseline_accuracy = accuracy.eval(feed_dict={self.net.x : self.dataset.test.images, 
													self.net.y_ : self.dataset.test.labels,
													self.net.keep_prob : 1.0})
		print("Software baseline accuracy : %g"%(baseline_accuracy))
		print("Adding Write Variation ... ")
		filename = 'withVariation_' + self.net.name + '_iter_{:d}'.format(cfg.BASELINE_ITERS) + '.ckpt'
		if not os.path.isfile(os.path.join(self.output_dir, self.net.name,'variation', filename + '.index')):
			allParameters = [v.eval() for v in tf.trainable_variables()]
			allparameter_tensors = [v for v in tf.trainable_variables()]
			truncatedParams = readVariation(allParameters, percentVar)
			for i in range(len(allparameter_tensors)):
				allparameter_tensors[i].load(truncatedParams[i], sess)
			self._snapshot(sess, cfg.BASELINE_ITERS-1, 1)	
		else:
			filePath = os.path.join(self.output_dir, self.net.name,'variation', filename)
			self.net.load(filePath, sess, self.saver, True)	
			
		variation_accuracy = accuracy.eval(feed_dict={self.net.x : self.dataset.test.images, 
													self.net.y_ : self.dataset.test.labels,
													self.net.keep_prob : 1.0})
		print("Model with write variation accuracy : %g"%(variation_accuracy))

		print("Creating mask and retraining few parameters ... ")
		masks = self._create_mask(percentRetrainable)	

		if cfg.DEBUG_LEVEL_1 or cfg.DEBUG_ALL:
			allParameters = [v.eval() for v in tf.trainable_variables()]
			print '~~~ MASK ~~~ '
			print masks[0]
			print ' ~~~ before training parameters ~~~'
			print allParameters[3]
		
		#optimizer = tf.train.AdamOptimizer(1e-4)
		optimizer = tf.train.GradientDescentOptimizer(0.01)
		grads = tf.gradients(cross_entropy, tf.trainable_variables())
		for i in range(len(grads)):
			grads[i] = grads[i]*masks[i]
		grads = list(zip(grads, tf.trainable_variables()))
		train_step = optimizer.apply_gradients(grads_and_vars=grads)
		
		last_snapshot_iter = -1
		for iter in range(max_iters):
			batch = self.dataset.train.next_batch(50)
			if cfg.DEBUG_TRAINING or cfg.DEBUG_ALL:	
				if (iter+1) % 100 == 0:
					train_accuracy = accuracy.eval(feed_dict={self.net.x : batch[0], 
															  self.net.y_ : batch[1],
															  self.net.keep_prob : 1.0})
					print("Step : %d, training accuracy : %g"%(iter, train_accuracy))

			feed_dict = {self.net.x : batch[0], self.net.y_ : batch[1], self.net.keep_prob : 0.5}
			train_step.run(feed_dict=feed_dict)	

			if (iter+1) % cfg.SNAPSHOT_ITERS == 0:
				last_snapshot_iter = iter
				self._snapshot(sess, iter, 2)	

		if last_snapshot_iter != iter:
			self._snapshot(sess, iter, 2)

def train_net(network, dataset, output_dir, max_iters, percentVar, percentRetrainable):
	""" 
	trains a network. inputs: 
	network as tensorflow graph, dataset, output directory, max number of iterations.
	"""
	saver = tf.train.Saver(max_to_keep=100)
	sess = tf.InteractiveSession()
	sw = SolverWrapper(sess, saver, dataset, network, output_dir)
	print 'Solving ... '
	return sw._train_model_rram_epochs(sess, max_iters, percentVar, percentRetrainable)
	print 'Done Training'

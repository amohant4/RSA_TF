# file: train.py
# Author	: Abinash Mohanty
# Date		: 05/10/2017
# Project	: RRAM training NN

import tensorflow as tf
import os 
from rram_NN.config import cfg
from rram_NN.rram_modeling import truncate, addWriteVariation, readVariation
import numpy as np
import cPickle
import math
import random

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
		self._masks = None
		self.v_trainable = self._get_trainable() 
		self.v_non_trainable = self._get_non_trainable()
		self.optimizer = self.net.optimizer
		self.pretrained_model_tf = os.path.join(self.output_dir, self.net.name, 'baseline', \
					self.net.name + '_iter_{:d}'.format(cfg.BASELINE_ITERS) + '.ckpt')
		self.pretrained_model_pkl = os.path.join(self.output_dir, self.net.name, 'baseline', \
					self.net.name + '_iter_{:d}'.format(cfg.BASELINE_ITERS) + '.pkl')
		self.pretrained_variation_pkl = os.path.join(self.output_dir, self.net.name, 'variation', \
					'variation_' + self.net.name + '_iter_{:d}'.format(cfg.BASELINE_ITERS) + '.pkl')					
		self.summaryDir = os.path.join(output_dir, self.net.name,'summary')
		self._create_dirs()
		self.writer = tf.summary.FileWriter(self.summaryDir)	

	def _create_dirs(self):
		if not os.path.exists(self.output_dir):
			os.makedirs(self.output_dir)
		if not os.path.exists(os.path.join(self.output_dir, self.net.name)):
			os.makedirs(os.path.join(self.output_dir, self.net.name))
		if not os.path.exists(os.path.join(self.output_dir, self.net.name, 'baseline')):
			os.makedirs(os.path.join(self.output_dir, self.net.name, 'baseline'))
		if not os.path.exists(os.path.join(self.output_dir, self.net.name, 'variation')):
			os.makedirs(os.path.join(self.output_dir, self.net.name, 'variation'))
		if not os.path.exists(self.summaryDir):
			os.makedirs(self.summaryDir)

	def _snapshot(self, sess, iter, mode=1):
		"""
		Writes snapshot of the network to file in output directory.
		inputs: tensorflow session, current iteration number, mode.
		mode :	
			0 = baseline model
			1 = with variation model
			2 = retrained model to rectify variation	
		"""
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
		print("Creating baseline model ... ")
		accuracy = self.net.accuracy
		grads = self.net.gradients
		grads_and_vars = list(zip(grads, self.v_trainable))
		train_step = self.optimizer.apply_gradients(grads_and_vars=grads_and_vars,global_step=self.net.global_step)
		sess.run(tf.global_variables_initializer())
		merged = tf.summary.merge_all()
		for iter in range(max_iters):
			batch = self.dataset.train.next_batch(cfg.TRAIN_BATCH_SIZE)
			if cfg.DEBUG_TRAINING or cfg.DEBUG_ALL:
				if (iter+1) % 1000 == 0:
					train_accuracy = accuracy.eval(feed_dict={self.net.x : batch[0], 
															  self.net.y_ : batch[1],
															  self.net.keep_prob : 1.0})
					print("Step : %d, training accuracy : %g"%(iter, train_accuracy))

			feed_dict = {self.net.x : batch[0], self.net.y_ : batch[1], self.net.keep_prob : 0.5}
			#summary = train_step.run([merged], feed_dict=feed_dict)
			summary, _ = sess.run([merged, train_step], feed_dict=feed_dict)
			self.writer.add_summary(summary, iter)
		self._snapshot(sess, iter, 0)
		self._saveTrainedModel(sess, self.pretrained_model_pkl)

	def _saveTrainedModel(self, sess, location):
		"""
		Helper function to save models as python variables. 
		It is stored using cPickle as binary files. Prior to this 
		the variables in the models must be initialized.
		Args:
			sess: tensorflow session 
			location: file address where the models will be savedd
		"""	
		variables_names =[v.name for v in tf.global_variables()]
		values = sess.run(variables_names)
		netParams = dict(zip(variables_names, values))
		with open(location, 'wb') as fid:
			cPickle.dump(netParams, fid, cPickle.HIGHEST_PROTOCOL)
		print 'saved models at "{}"'.format(location)

	def _find_or_train_baseline(self, sess):
		"""
		Function looks for the baseline software models. 
		Incase it doesn't find that, it call the _train_model_software_baseline() to create baseline models. 
		"""
		filename = self.net.name + '_iter_{:d}'.format(cfg.BASELINE_ITERS) + '.ckpt'
		if not os.path.isfile(os.path.join(self.output_dir, self.net.name,'baseline', filename + '.index')):
			print 'Baseline software models not found. Training software baseline'	
			self._train_model_software_baseline(sess, cfg.BASELINE_ITERS)
		else:
			print 'Baseline models for {} found at {}'.format(self.net.name, os.path.join(self.output_dir, self.net.name,'baseline'))
			self._load_baseline_model(sess)
		return self._eval_net('Baseline')	

	def _create_mask_v0(self, percentRetrainable):
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
		keys = [v.name for v in tf.trainable_variables()]
		masks = {}
		for i in range(len(allShapes)):
			mask = np.random.rand(*allShapes[i])
			mask = np.where(mask < percentRetrainable/100.0, 1., 0.)	
			masks[keys[i]] = mask
		return masks

	def _create_mask(self, percentRetrain):
		"""
		Function to create a random mask to stop gradient flow through specific 
		prameters of the network while retraining the network. 
		This creates a list of ndarrays with values equal to 0/1 and dimention
		same as the variables in the net.  
		"""	
		if cfg.DEBUG_LEVEL_1 or cfg.DEBUG_ALL:
			print 'Creating masks for stoping random gradient with \
					retention ratio = {}'.format(percentRetrain)
		totalRetrained = 0
		totalParams = 0
		masks = {}
		for v in self.v_trainable:
			masks[v.name] = np.zeros(v.get_shape())
		keys = masks.keys()
		for key in keys:	
			shape = masks[key].shape
			dims = len(shape)
			mask = masks[key]
			if dims == 4:
				pecentPerDiagonal = float(shape[0]*shape[2]*shape[3])*100.0/float(shape[0]*shape[1]*shape[2]*shape[3])
				numDiagonals = int(math.ceil(percentRetrain / pecentPerDiagonal))
				for i in range(shape[3]):
					for j in range(shape[2]):
						for i in range(numDiagonals):
							if shape[0] == shape[1]:
								x = np.arange(shape[0])
								y = np.arange(shape[1])
								random.shuffle(x)
								random.shuffle(y)
								for m in range(len(x)):
									mask[x[m],y[m],j,i] = 1.0
				totalRetrained += shape[0]*shape[2]*shape[3]*numDiagonals 
				totalParams += shape[0]*shape[1]*shape[2]*shape[3]
				if cfg.DEBUG_ALL:
					print 'For layer ',key,' percentage per diagonal = ',pecentPerDiagonal
					print 'For layer ',key,' number of diagonals selected = ',numDiagonals
					print 'For layer ',key,' % of retrained parameters = ',float(numDiagonals)*100.0/shape[1]
			elif dims == 2:
				maxDim = max(shape[0],shape[1])
				pecentPerDiagonal = float(min(shape[0],shape[1]))*100.0/float(shape[0]*shape[1])
				dummy = np.zeros((maxDim, maxDim))
				x = np.arange(maxDim)
				y = np.arange(maxDim)
				numDiagonals = int(math.ceil(percentRetrain / pecentPerDiagonal))
				for k in range(numDiagonals):
					random.shuffle(x)
					random.shuffle(y)
					for m in range(len(x)):
						dummy[x[m],y[m]] = 1.0
				mask = dummy[:shape[0],:shape[1]]
				totalRetrained += numDiagonals*min(shape[0],shape[1]) 	
				totalParams += shape[0]*shape[1]	
				if cfg.DEBUG_ALL:
					print 'For layer ',key,' percentage per diagonal = ',pecentPerDiagonal
					print 'For layer ',key,' number of diagonals selected = ',numDiagonals
					print 'For layer ',key,' % of retrained parameters = ',float(numDiagonals)*100.0/max(shape[0],shape[1])
			elif dims == 1:
				x = np.arange(shape[0])
				random.shuffle(x)
				num = int(math.ceil(shape[0]*percentRetrain/100.0))
				for i in range(num):
					mask[x[i]] = 1.0	
				totalRetrained += num
				totalParams += shape[0]
				if cfg.DEBUG_ALL:		
					print 'For layer ',key,' % of retrained parameters = ',float(num)*100.0/float(shape[0])
		if cfg.DEBUG_ALL:
			totalPercentRetrained = float(totalRetrained)*100.0/float(totalParams)
			print 'Final Retrained parameter % = ',	totalPercentRetrained
		return masks

	def _get_trainable(self):
		v_trainable = [v for v in tf.trainable_variables()]
		return v_trainable	

	def _get_non_trainable(self):
		if cfg.DEBUG_TRAINING or cfg.DEBUG_ALL:
			print 'Separating trainable and non-trainable variables in the graph.'	
		v_all = [v for v in tf.global_variables()]
		v_trainable = [v for v in tf.trainable_variables()]
		v_no_train = list(set(v_all)-set(v_trainable))
		v_non_trainable = [v for v in v_no_train if v.name != 'global_step:0']
		return v_non_trainable

	def _load_model(self, path):
		with open(path) as fid:
			if cfg.DEBUG_TRAINING or cfg.DEBUG_ALL:
				print 'Loading variation models from : ', path
			params = cPickle.load(fid)	
		all_vars = tf.global_variables()	
		for i in range(len(self.v_non_trainable)):
			if cfg.DEBUG_TRAINING or cfg.DEBUG_ALL:
				print ' @@@ Loading from variation model for : ', self.v_non_trainable[i].name
			self.v_non_trainable[i].load(params[self.v_non_trainable[i].name])

	def _init_network(self, sess, percentRetrainable):
		"""
		Initializes the network.
		creates gradient mask / connectivity matrix for SRAM crossbar. 
		Args:
			sess: tensorflow session in which graph is present.
			percentRetrainable: percentage of trainable parameters in SRAM crossbar.	  
		"""
		sess.run(tf.global_variables_initializer())
		#self._load_model(self.pretrained_model_pkl)
		self._load_model(self.pretrained_variation_pkl)
		self._masks = self._create_mask(percentRetrainable)
		#self._masks = self._create_mask_v0(percentRetrainable)
		if cfg.DEBUG_TRAINING or cfg.DEBUG_ALL:
			for key in self._masks.keys():
				print key, ' -- ',self._masks[key].shape	
		for i in range(len(self.v_trainable)):
			parameters = self.v_trainable[i].eval()
			parameters = parameters * self._masks[self.v_trainable[i].name]
			self.v_trainable[i].load(parameters)	

	def _eval_net(self, description=''):
		"""
		This function evaluates the performance of the current network in the given session.
		Args:
			sess: Tensorflow session
			description: string describing the network for loggin.
		"""
		accuracy = self.net.accuracy.eval(feed_dict={self.net.x : self.dataset.test.images, 
													self.net.y_ : self.dataset.test.labels,
													self.net.keep_prob : 1.0})
		print description, ' Network accuracy : {}'.format(accuracy)
		return accuracy

	def _train_model(self, sess, max_iters):
		"""
		Function to train the model.
		Args:
			sess: tensorflow session.
			max_iters: maximum number of batches. 	 
		"""
		accuracy = self.net.accuracy
		grads = self.net.gradients
		keys = [v.name for v in self.v_trainable]
		for i in range(len(grads)):
			grads[i] = grads[i]*self._masks[keys[i]]
		grads_and_vars = list(zip(grads, self.v_trainable))
		train_step = self.optimizer.apply_gradients(grads_and_vars=grads_and_vars,global_step=self.net.global_step)
		merged = tf.summary.merge_all()
		last_snapshot_iter = -1
		for iter in range(max_iters):
			batch = self.dataset.train.next_batch(cfg.TRAIN_BATCH_SIZE)
			if cfg.DEBUG_TRAINING or cfg.DEBUG_ALL:	
				if (iter+1) % 1000 == 0:
					train_accuracy = accuracy.eval(feed_dict={self.net.x : batch[0], 
														  self.net.y_ : batch[1],
														  self.net.keep_prob : 1.0})
					print("Step : %d, training accuracy : %g"%(iter, train_accuracy))
			feed_dict = {self.net.x : batch[0], self.net.y_ : batch[1], self.net.keep_prob : 0.5}
			#train_step.run(feed_dict=feed_dict)
			summary, _ = sess.run([merged, train_step], feed_dict=feed_dict)
			self.writer.add_summary(summary, iter)
			if (iter+1) % cfg.SNAPSHOT_ITERS == 0:
				last_snapshot_iter = iter
				self._snapshot(sess, iter, 2)	
		if last_snapshot_iter != iter:
			self._snapshot(sess, iter, 2)

	def _test_new(self, sess, max_iters, percentVar, percentRetrainable):
		self._init_network(sess, percentRetrainable)
		#_ = self._eval_net('Baseline')
		#addWriteVariation(self.v_non_trainable, percentVar)
		#path = os.path.join(self.output_dir, self.net.name, 'variation', 'variation_'+self.net.name+'_iter_'+str(max_iters)+'.pkl')	
		#self._saveTrainedModel(sess, path)	
		_ = self._eval_net('Write Variation')
		self._train_model(sess, max_iters)
		return self._eval_net('Retrained')

def train_net(network, dataset, output_dir, max_iters, percentVar=50.0, percentRetrainable=5.0):
	""" 
	Trains a network for a given dataset. 
	Args:
		network: tensorflow network to train
		dataset: dataset for training and testing
		output_dir: directory to store checkpoints
		max_iters: maximum iterations to run the training process
		baselineWeights: path to the trained weights of the original network  
		percentVar: percentange of variation to be introduced in the weights
		percentRetrainable: percentage of parameters to retrain (Number of parameters in the SRAM array).
	"""
	saver = tf.train.Saver(max_to_keep=100)
	acc = 0.0
	sess = tf.InteractiveSession()
	sw = SolverWrapper(sess, saver, dataset, network, output_dir )
	print 'Solving ... '
	acc = sw._test_new(sess, max_iters, percentVar, percentRetrainable)
	#acc = sw._find_or_train_baseline(sess)
	graphWriter = tf.summary.FileWriter(sw.summaryDir, sess.graph)
	print 'Done Training'
	return acc

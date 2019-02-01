# file: train.py
# Author	: Abinash Mohanty
# Date		: 05/10/2017
# Project	: RRAM training NN

import tensorflow as tf
import os 
import sys
from rram_NN.config import cfg
from rram_NN.rram_modeling import addDefects, readVerifyTopN
import numpy as np
import cPickle
import math
import random
import matplotlib.pyplot as plt

# change dataset_name to dataset.name

class SolverWrapper(object):
	def __init__(self, sess, saver, dataset, network, output_dir, dataset_name, stddevVar):
		"""
		SolverWrapper constructor. Inputs are: 
			tensorflow session, tensorflow saver, dataset, network, output directory.
		"""
		self.net = network
		self.stddevVar = stddevVar
		self.dataset_name = dataset_name
		self.dataset = dataset
		self.output_dir = output_dir
		self.saver = saver
		self._masks = None
		self.v_trainable = self._get_trainable() 
		self.v_non_trainable = self._get_non_trainable()
		self.optimizer = self.net.optimizer
		self.pretrained_model_tf = os.path.join(self.output_dir, self.net.name, 'baseline', \
					self.net.name+'_'+self.dataset_name+'.ckpt')
		self.pretrained_model_pkl = os.path.join(self.output_dir, self.net.name, 'baseline', \
					self.net.name+'_'+self.dataset_name +'.pkl')
		self.pretrained_variation_pkl = os.path.join(self.output_dir, self.net.name, 'variation', \
					self.net.name +'_'+ self.dataset_name + '_' + str(stddevVar) +'.pkl')					
		self.summaryDir = os.path.join(output_dir, self.net.name,'summary')
		self._create_dirs()
		if cfg.DEBUG_ALL or cfg.WRITE_TO_SUMMARY:
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
		filename = prefix + self.net.name + '_iter_{:d}_'.format(iter+1) +self.dataset_name+ '.ckpt'
                if mode == 0:
		    filename = prefix + self.net.name +'_'+self.dataset_name+ '.ckpt'
		if mode == 0:
			filename = os.path.join(self.output_dir, self.net.name, 'baseline', filename)
		elif mode == 1:
			filename = os.path.join(self.output_dir, self.net.name,'variation', filename)
		elif mode == 2:
			filename = os.path.join(self.output_dir, self.net.name, filename)
		self.saver.save(sess, filename)
		if cfg.DEBUG_TRAINING or cfg.DEBUG_ALL:
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
		if cfg.DEBUG_ALL or cfg.DEBUG_TRAINING:
			merged = tf.summary.merge_all()
		for iter in range(max_iters):
			batch = self.dataset.train.next_batch(cfg.TRAIN.TRAIN_BATCH_SIZE)
			if cfg.DEBUG_TRAINING or cfg.DEBUG_ALL:
				if (iter+1) % 1000 == 0:
					train_accuracy = accuracy.eval(feed_dict={self.net.x : batch[0], 
										  self.net.y_ : batch[1],
										  self.net.phase : 0.0,
										  self.net.keep_prob : 1.0})
					print("Step : %d, training accuracy : %g"%(iter, train_accuracy))

			feed_dict = {self.net.x : batch[0], self.net.y_ : batch[1], self.net.phase : 1.0 ,self.net.keep_prob : 0.5}
			if cfg.DEBUG_ALL or cfg.WRITE_TO_SUMMARY:
				summary, _ = sess.run([merged, train_step], feed_dict=feed_dict)
				self.writer.add_summary(summary, iter)
			else:			
				_ = sess.run([train_step], feed_dict=feed_dict)
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

	def _find_or_train_baseline(self, sess, iters):
		"""
		Function looks for the baseline software models. 
		Incase it doesn't find that, it call the _train_model_software_baseline() to create baseline models. 
		"""
		filename = self.net.name + '_' + self.dataset_name + '.ckpt'
		if not os.path.isfile(os.path.join(self.output_dir, self.net.name,'baseline', filename + '.index')):
			print 'Baseline software models not found. Training software baseline'	
			self._train_model_software_baseline(sess, iters)
		else:
			print 'Baseline models for {} found at {}'.format(self.net.name, os.path.join(self.output_dir, self.net.name,'baseline'))
			self.saver.restore(sess, os.path.join(self.output_dir, self.net.name,'baseline', filename))
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

	def _create_mask_debug(self, percentRetrain):
		shapes = [v.get_shape().as_list() for v in tf.trainable_variables()]
		keys = [v.name for v in tf.trainable_variables()]
		masks = {}
		for i in range(len(shapes)):
			mask = np.zeros(shapes[i])
			masks[keys[i]] = mask
		return masks

	def _create_mask_topk(self, percentRetrain):
		if cfg.DEBUG_LEVEL_1 or cfg.DEBUG_ALL:
			print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
			print 'Creating masks for tarining top {}% of parameters '.format(percentRetrain)
			print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
		shapes = [v.get_shape().as_list() for v in tf.trainable_variables()]
		keys = [v.name for v in tf.trainable_variables()]
		masks = {}
		if cfg.DEBUG_ALL or cfg.DEBUG_TRAINING:	
			totalRetrained = 0
			totalParams = 0
		for i in range(len(shapes)):
			mask = np.zeros(shapes[i])
			dims = len(shapes[i])
			if dims == 4:
				pecentPerDiagonal = 100.0/float(min(shapes[i][0], shapes[i][1]))
				numDiagonals = int(math.ceil(percentRetrain / pecentPerDiagonal))
				x = np.arange(shapes[i][0])
				y = np.arange(shapes[i][1])
				for ii in range(numDiagonals):
					for j in range(shapes[i][3]):
						for k in range(shapes[i][2]):
							random.shuffle(x)
							random.shuffle(y)
							for m in range(min(shapes[i][0], shapes[i][1])):
								mask[x[m],y[m],k,j] = 1.0
				if cfg.DEBUG_ALL or cfg.DEBUG_TRAINING:		
					totalParams += shapes[i][0]*shapes[i][1]*shapes[i][2]*shapes[i][3]		
					totalRetrained += sum(sum(sum(sum(mask))))
					print 'For layer ',str(keys[i]),' percentage per diagonal = ',pecentPerDiagonal
					print 'For layer ',str(keys[i]),' number of diagonals selected = ',numDiagonals
					print 'For layer ',keys[i],' % of retrained parameters = ', float(sum(sum(sum(sum(mask)))))*100.0/float(shapes[i][0]*shapes[i][1]*shapes[i][2]*shapes[i][3])
					print '~~~ ~~~ ~~~ ~~~~ '		
			elif dims == 2:
				maxDim = max(shapes[i][0], shapes[i][1])
				pecentPerDiagonal = float(min(shapes[i][0],shapes[i][1]))*100/float(shapes[i][0]*shapes[i][1])
				numDiagonals = int(math.ceil(percentRetrain/pecentPerDiagonal))
				x = np.arange(maxDim)
				y = np.arange(maxDim)
				dummy = np.zeros((maxDim, maxDim))
				for k in range(numDiagonals):
					random.shuffle(x)
					random.shuffle(y)
					for m in range(len(x)):
						dummy[x[m],y[m]] = 1.0	
				mask = dummy[:shapes[i][0], :shapes[i][1]]
				if cfg.DEBUG_ALL or cfg.DEBUG_TRAINING:
					totalRetrained += sum(sum(mask))
					totalParams += shapes[i][0]*shapes[i][1]
					print 'For layer ',keys[i],' percentage per diagonal = ',pecentPerDiagonal
					print 'For layer ',keys[i],' number of diagonals selected = ',numDiagonals
					print 'For layer ',keys[i],' % of retrained parameters = ', float(sum(sum(mask)))*100.0/float(shapes[i][0]*shapes[i][1])
					print '~~~ ~~~ ~~~ ~~~~ '
			elif dims == 1:
				x = np.arange(shapes[i][0])
				random.shuffle(x)
				num = int(math.ceil(shapes[i][0]*percentRetrain/100.0))
				for p in range(num):
					mask[x[p]] = 1.0	
					if cfg.DEBUG_ALL:		
						totalRetrained += num
						totalParams += shapes[i][0]
				if cfg.DEBUG_ALL:		
					print 'For layer ',keys[i],' % of num retrained = ', num	 
					print 'For layer ',keys[i],' % of num totalParams = ', shapes[i][0]
					print 'For layer ',keys[i],' % of retrained parameters = ', float(sum(mask))*100.0/float(shapes[i][0])
					print '~~~ ~~~ ~~~ ~~~~ '
			masks[keys[i]] = mask	

		if cfg.DEBUG_ALL or cfg.DEBUG_ALL:
			totalPercentRetrained = float(totalRetrained)*100.0/float(totalParams)
			print 'Final Retrained parameter % = ',	totalPercentRetrained
		return masks

	def _create_mask(self, percentRetrain):
		"""
		Function to create a random mask to stop gradient flow through specific 
		prameters of the network while retraining the network. 
		This creates a list of ndarrays with values equal to 0/1 and dimention
		same as the variables in the net.  
		"""	
		if cfg.DEBUG_LEVEL_1 or cfg.DEBUG_ALL:
			print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
			print 'Creating masks for stoping random gradient with retention ratio = {}'.format(percentRetrain)
			print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
		shapes = [v.get_shape().as_list() for v in tf.trainable_variables()]
		keys = [v.name for v in tf.trainable_variables()]
		masks = {}
		if cfg.DEBUG_ALL or cfg.DEBUG_TRAINING:	
			totalRetrained = 0
			totalParams = 0
		for i in range(len(shapes)):
			mask = np.zeros(shapes[i])
			dims = len(shapes[i])
			if dims == 4:
				pecentPerDiagonal = 100.0/float(min(shapes[i][0], shapes[i][1]))
				numDiagonals = int(math.ceil(percentRetrain / pecentPerDiagonal))
				x = np.arange(shapes[i][0])
				y = np.arange(shapes[i][1])
				for ii in range(numDiagonals):
					for j in range(shapes[i][3]):
						for k in range(shapes[i][2]):
							random.shuffle(x)
							random.shuffle(y)
							for m in range(min(shapes[i][0], shapes[i][1])):
								mask[x[m],y[m],k,j] = 1.0
				if cfg.DEBUG_ALL or cfg.DEBUG_TRAINING:		
					totalParams += shapes[i][0]*shapes[i][1]*shapes[i][2]*shapes[i][3]		
					totalRetrained += sum(sum(sum(sum(mask))))
					print 'For layer ',str(keys[i]),' percentage per diagonal = ',pecentPerDiagonal
					print 'For layer ',str(keys[i]),' number of diagonals selected = ',numDiagonals
					print 'For layer ',keys[i],' % of retrained parameters = ', float(sum(sum(sum(sum(mask)))))*100.0/float(shapes[i][0]*shapes[i][1]*shapes[i][2]*shapes[i][3])
					print '~~~ ~~~ ~~~ ~~~~ '		
			elif dims == 2:
				maxDim = max(shapes[i][0], shapes[i][1])
				pecentPerDiagonal = float(min(shapes[i][0],shapes[i][1]))*100/float(shapes[i][0]*shapes[i][1])
				numDiagonals = int(math.ceil(percentRetrain/pecentPerDiagonal))
				x = np.arange(maxDim)
				y = np.arange(maxDim)
				dummy = np.zeros((maxDim, maxDim))
				for k in range(numDiagonals):
					random.shuffle(x)
					random.shuffle(y)
					for m in range(len(x)):
						dummy[x[m],y[m]] = 1.0	
				mask = dummy[:shapes[i][0], :shapes[i][1]]
				if cfg.DEBUG_ALL or cfg.DEBUG_TRAINING:
					totalRetrained += sum(sum(mask))
					totalParams += shapes[i][0]*shapes[i][1]
					print 'For layer ',keys[i],' percentage per diagonal = ',pecentPerDiagonal
					print 'For layer ',keys[i],' number of diagonals selected = ',numDiagonals
					print 'For layer ',keys[i],' % of retrained parameters = ', float(sum(sum(mask)))*100.0/float(shapes[i][0]*shapes[i][1])
					print '~~~ ~~~ ~~~ ~~~~ '
			elif dims == 1:
				x = np.arange(shapes[i][0])
				random.shuffle(x)
				num = int(math.ceil(shapes[i][0]*percentRetrain/100.0))
				for p in range(num):
					mask[x[p]] = 1.0	
					if cfg.DEBUG_ALL:		
						totalRetrained += num
						totalParams += shapes[i][0]
				if cfg.DEBUG_ALL:		
					print 'For layer ',keys[i],' % of num retrained = ', num	 
					print 'For layer ',keys[i],' % of num totalParams = ', shapes[i][0]
					print 'For layer ',keys[i],' % of retrained parameters = ', float(sum(mask))*100.0/float(shapes[i][0])
					print '~~~ ~~~ ~~~ ~~~~ '
			masks[keys[i]] = mask	

		if cfg.DEBUG_ALL or cfg.DEBUG_ALL:
			totalPercentRetrained = float(totalRetrained)*100.0/float(totalParams)
			print 'Final Retrained parameter % = ',	totalPercentRetrained
		return masks

	def _get_trainable(self):
		"""
		Function to return the trainable variables in the current graph.
		"""
		v_trainable = [v for v in tf.trainable_variables()]
		return v_trainable	

	def _get_non_trainable(self):
		"""
		Function to return the non_trainable variables in the current graph.
		It returns only the trainable parameters in the baseline neural network which are 
		non trainable in the SRAM branched neural network. 
		The returned values do not include the variables like global step, learning rate etc.
		"""
		if cfg.DEBUG_TRAINING or cfg.DEBUG_ALL:
			print '[',os.path.basename(sys.argv[0]),'] Separating trainable and non-trainable variables in the graph.'	
		v_all = [v for v in tf.global_variables()]
		v_trainable = [v for v in tf.trainable_variables()]
		v_no_train = list(set(v_all)-set(v_trainable))
		v_non_trainable = [v for v in v_no_train if v.name != 'global_step:0']
		return v_non_trainable

	def _load_model(self, path, variableList):
		"""
		Function to load model parameters from cPickle file. 
		the pickle file should be a dictoary with keys as the variable names
		and values as the variable values.
		Args:
			path: location of the pickle file
			variableList: list of tensors which are to be loaded.
		"""
		with open(path) as fid:
			if cfg.DEBUG_TRAINING or cfg.DEBUG_ALL:
				print '[',os.path.basename(sys.argv[0]),'] Models weight from : ', path
			params = cPickle.load(fid)	
		for i in range(len(variableList)):
			if cfg.DEBUG_TRAINING or cfg.DEBUG_ALL:
				print '[',os.path.basename(sys.argv[0]),'] Loading variables for : ', variableList[i].name, ' - ', variableList[i].shape
			variableList[i].load(params[variableList[i].name])

	def _init_baseline_network(self, sess):
		"""
		Initializes only the baseline network.
		creates gradient mask / connectivity matrix for SRAM crossbar. 
		Args:
			sess: tensorflow session in which graph is present.
		"""
		sess.run(tf.global_variables_initializer())
		self._load_model(self.pretrained_model_pkl, self.v_trainable)

	def _init_network(self, sess, percentRetrainable):
		"""
		Initializes the network.
		creates gradient mask / connectivity matrix for SRAM crossbar. 
		Args:
			sess: tensorflow session in which graph is present.
			percentRetrainable: percentage of trainable parameters in SRAM crossbar.	  
		"""
		sess.run(tf.global_variables_initializer())
		self._load_model(self.pretrained_model_pkl, self.v_non_trainable)
		self._masks = self._create_mask(percentRetrainable)
		#self._masks = self._create_mask_debug(percentRetrainable)
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
													self.net.phase: 0.0,
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
		if cfg.DEBUG_ALL or cfg.DEBUG_TRAINING:
			merged = tf.summary.merge_all()
		last_snapshot_iter = -1
		for iter in range(max_iters):
			batch = self.dataset.train.next_batch(cfg.TRAIN.TRAIN_BATCH_SIZE)
			if cfg.DEBUG_TRAINING or cfg.DEBUG_ALL:	
				if (iter+1) % 1000 == 0:
					train_accuracy = accuracy.eval(feed_dict={self.net.x : batch[0], 
														  self.net.y_ : batch[1],
														  self.net.phase: 0.0,
														  self.net.keep_prob : 1.0})
					print("Step : %d, training accuracy : %g"%(iter, train_accuracy))
			feed_dict = {self.net.x : batch[0], self.net.y_ : batch[1], self.net.phase : 1.0,self.net.keep_prob : 0.5}
			if cfg.DEBUG_ALL or cfg.WRITE_TO_SUMMARY:
				summary, _ = sess.run([merged, train_step], feed_dict=feed_dict)
				self.writer.add_summary(summary, iter)
			else:
				_ = sess.run([train_step], feed_dict=feed_dict)
			if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
				last_snapshot_iter = iter
				self._snapshot(sess, iter, 2)	
		if last_snapshot_iter != iter:
			self._snapshot(sess, iter, 2)

	def _train_model_v1(self, sess, max_iters):
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
		if cfg.DEBUG_ALL or cfg.DEBUG_TRAINING:
			merged = tf.summary.merge_all()
		last_snapshot_iter = -1
		acc = []
		iterations = []
		for iter in range(max_iters):
			batch = self.dataset.train.next_batch(cfg.TRAIN.TRAIN_BATCH_SIZE)
			if cfg.DEBUG_TRAINING or cfg.DEBUG_ALL:	
				if (iter+1) % 50 == 0:
					train_accuracy = accuracy.eval(feed_dict={self.net.x : self.dataset.test.images, 
														  self.net.y_ : self.dataset.test.labels,
														  self.net.phase: 0.0,
														  self.net.keep_prob : 1.0})
					print("Step : %d, training accuracy : %g"%(iter, train_accuracy))
					acc.append(train_accuracy)
					iterations.append(iter)
			feed_dict = {self.net.x : batch[0], self.net.y_ : batch[1], self.net.phase : 1.0,self.net.keep_prob : 0.5}
			if cfg.DEBUG_ALL or cfg.WRITE_TO_SUMMARY:
				summary, _ = sess.run([merged, train_step], feed_dict=feed_dict)
				self.writer.add_summary(summary, iter)
			else:
				_ = sess.run([train_step], feed_dict=feed_dict)
			if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
				last_snapshot_iter = iter
				self._snapshot(sess, iter, 2)	
		if last_snapshot_iter != iter:
			self._snapshot(sess, iter, 2)
		return acc, iterations

	def _add_variation_to_baseline(self, sess, stddevVar, num_levels=32, writeToPickle=True):
		self._init_baseline_network(sess)
		_ = self._eval_net('Baseline')
		"""
		addDefects(self.v_trainable, stddevVar, num_levels, cfg.RRAM.SA0, cfg.RRAM.SA1)
		acc = self._eval_net('Write Variation')
		"""
		#"""
		if not os.path.exists(self.pretrained_variation_pkl): 
			addDefects(self.v_trainable, stddevVar , num_levels, cfg.RRAM.SA0, cfg.RRAM.SA1)
			if writeToPickle:
				self._saveTrainedModel(sess, self.pretrained_variation_pkl)
		else:
			self._load_model(self.pretrained_variation_pkl, self.v_trainable)
		acc = self._eval_net('Write Variation')
		#"""
		return acc

	def _checkNVerifyTopN(self, sess, topN, stddev, netName, datasetName):
		#self._init_baseline_network(sess)
		sess.run(tf.global_variables_initializer())
		path = os.path.join(self.output_dir, self.net.name, 'baseline', \
					self.net.name+'_'+self.dataset_name +'_quatized.pkl')
		self._load_model(path, self.v_trainable)
		_ = self._eval_net('Quantized')
		print '_checkNVerifyTopN | stddev', str(stddev) 
		readVerifyTopN(self.v_trainable, topN, stddev, netName, datasetName)	
		acc = self._eval_net('Top N Read-Verified')
		return acc

	def _retrain_baseline(self, sess, max_iters, stddevVar, percentRetrainable):
		_ = self._add_variation_to_baseline(sess, stddevVar, 32, True)
		self._masks = self._create_mask(percentRetrainable)
		self._train_model(sess, max_iters)
		return self._eval_net('Retrained')

	def _iters_vs_accuracy(self, sess, max_iters, stddevVar, percentRetrainable):
		_ = self._add_variation_to_baseline(sess, stddevVar, 32, True)
		self._masks = self._create_mask(percentRetrainable)
		acc, iters = self._train_model_v1(sess, max_iters)
		_ = self._eval_net('Retrained')
		return acc, iters		

def plot_images(images, cls_true, cls_pred=None, smooth=True):
	assert len(images) == len(cls_true) == 9
	fig, axes = plt.subplots(3, 3)
	if cls_pred is None:
		hspace = 0.3
	else:
		hspace = 0.6
	fig.subplots_adjust(hspace=hspace, wspace=0.3)
	for i, ax in enumerate(axes.flat):
		if smooth:
			interpolation = 'spline16'
		else:
			interpolation = 'nearest'
		ax.imshow(images[i, :, :, :], interpolation=interpolation)
		cls_true_name = 'XXX'
		if cls_pred is None:
			xlabel = "True: {0}".format(cls_true_name)
		else:
			cls_pred_name = class_names[cls_pred[i]]
			xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)
		ax.set_xlabel(xlabel)
		ax.set_xticks([])
		ax.set_yticks([])
	plt.show()	

def verifyTopN(network, dataset, output_dir, stddevVar, dataset_name, num_levels, topN):
	saver = tf.train.Saver(max_to_keep=100)
	sess = tf.InteractiveSession()
	sw = SolverWrapper(sess, saver, dataset, network, output_dir, dataset_name, stddevVar)
	print 'Solving ... '
	print 'VerifyTopN | stddev', str(stddevVar) 
	acc = sw._checkNVerifyTopN(sess, topN, stddevVar, network.name, dataset_name)
	if cfg.WRITE_TO_SUMMARY or cfg.DEBUG_ALL:
		graphWriter = tf.summary.FileWriter(sw.summaryDir, sess.graph)	
	print 'Done Training'
	return acc

def train_net_v1(network, dataset, output_dir, iters, stddevVar, percentRetrainable, dataset_name, num_levels):
	saver = tf.train.Saver(max_to_keep=100)
	sess = tf.InteractiveSession()
	sw = SolverWrapper(sess, saver, dataset, network, output_dir, dataset_name, stddevVar)
	print 'Solving ... '
	acc, iters = sw._iters_vs_accuracy(sess, iters, stddevVar, percentRetrainable)
	if cfg.WRITE_TO_SUMMARY or cfg.DEBUG_ALL:
		graphWriter = tf.summary.FileWriter(sw.summaryDir, sess.graph)	
	print 'Done Training'
	return acc, iters


def train_net(network, dataset, output_dir, iters, stddevVar, percentRetrainable, dataset_name, num_levels):
	""" 
	Trains a network for a given dataset. 
	Args:
		network: tensorflow network to train
		dataset: dataset for training and testing
		output_dir: directory to store checkpoints
		iters: maximum iterations to run the training process
		baselineWeights: path to the trained weights of the original network  
		stddevVar: standard deviation of variation to be introduced in the weights as device parameters.
		percentRetrainable: percentage of parameters to retrain (Number of parameters in the SRAM array).
	"""
	saver = tf.train.Saver(max_to_keep=100)
	sess = tf.InteractiveSession()
	sw = SolverWrapper(sess, saver, dataset, network, output_dir, dataset_name, stddevVar)
	print 'Solving ... '
	#acc = sw._find_or_train_baseline(sess, iters)
	#acc = sw._add_variation_to_baseline(sess, stddevVar, num_levels, True)
	#acc = sw._test_new(sess, iters, stddevVar, percentRetrainable)
	acc = sw._retrain_baseline(sess, iters, stddevVar, percentRetrainable)
	#acc, iters = sw._iters_vs_accuracy(sess, iters, stddevVar, percentRetrainable)
	if cfg.WRITE_TO_SUMMARY or cfg.DEBUG_ALL:
		graphWriter = tf.summary.FileWriter(sw.summaryDir, sess.graph)	
	print 'Done Training'
	return acc

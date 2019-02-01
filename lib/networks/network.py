# file: network.py
# Date		: 05/08/2017
# Project	: RRAM training NN

import tensorflow as tf
import numpy as np
from rram_NN.config import cfg

DEFAULT_PADDING='SAME'

# def include_original(dec):
    # """ Meta decorator, which make the original function callable (via f._original() )"""
    # def meta_decorator(f):
        # decorated = dec(f)
        # decorated._original = f
        # return decorated
	# return meta_decorator

# @include_original
def layer(op):
	def layer_decorated(self, *args, **kwargs):
		name=kwargs.setdefault('name', self.get_unique_name(op.__name__))
		if len(self.inputs)==0:
			raise RuntimeError('No input variables found for layer %s.'%name)
		elif len(self.inputs)==1:
			layer_input=self.inputs[0]
		else:
			layer_input=list(self.inputs)
		layer_output=op(self,layer_input, *args, **kwargs)
		self.layers[name]=layer_output
		self.feed(layer_output)
		return self
	return layer_decorated	

class Network(object):
	def __init__(self,inputs,trainable=True):
		self.inputs=[]
		self.layers=dict(inputs)
		self.trainable=trainable
		self.setup()
	
	def setup(self):
		raise NotImplementedError('Must be subclassed.')

	def load(self, data_path, session, saver, ignore_missing=False):
		if data_path.endswith('.ckpt'):
			saver.restore(session, data_path)
		else:
			data_dict = np.load(data_path).item()
			for key in data_dict:
				with tf.variable_scope(key, reuse=True):
					for subkey in data_dict[key]:
						try:
							var = tf.get_variable(subkey)
							session.run(var.assign(data_dict[key][subkey]))
							print "assign pretrain model "+subkey+ " to "+key
						except ValueError:
							print "ignore "+key
							if not ignore_missing:
								raise

	def feed(self, *args):
		assert len(args) != 0
		self.inputs = []
		for layer in args:
			if isinstance(layer, basestring):
				try:
					layer = self.layers[layer]
					if cfg.DEBUG_ALL:	
						print layer
				except KeyError:
					print self.layers.keys()
					raise KeyError('Unknown layer name fed: %s'%layer)
			self.inputs.append(layer)
		return self

	def get_output(self, layer):
		try:
			layer = self.layers[layer]
		except KeyError:
			print self.layers.keys()
			raise KeyError('Unknown layer name fed: %s'%layer)
		return layer
		
	def get_unique_name(self, prefix):
		id = sum(t.startswith(prefix) for t,_ in self.layers.items())+1
		return '%s_%d'%(prefix, id)

	def make_var(self, name, shape, initializer=None, trainable=True, regularizer=None):
		return tf.get_variable(name, shape, initializer=initializer, trainable=trainable, regularizer=regularizer)
	
	def validate_padding(self, padding):
		assert padding in ('SAME', 'VALID')

	def variable_summaries(self, var):
		with tf.name_scope('summaries'):
			mean = tf.reduce_mean(var)
			tf.summary.scalar('mean',mean)
			with tf.name_scope('stddev'):
				stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
			tf.summary.scalar('stddev', stddev)
			tf.summary.scalar('max', tf.reduce_max(var))	
			tf.summary.scalar('min', tf.reduce_min(var))
			tf.summary.histogram('histogram', var)	
			
	@layer
	def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, biased=True,relu=True, padding=DEFAULT_PADDING, trainable=True):
		""" contribution by miraclebiu, and biased option"""
		self.validate_padding(padding)
		c_i = input.get_shape()[-1]
		convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
		with tf.variable_scope(name) as scope:
			init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
			init_biases = tf.constant_initializer(0.0)
			kernel = self.make_var('weights', [k_h, k_w, c_i, c_o], init_weights, trainable, \
			regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
			if biased:
				biases = self.make_var('biases', [c_o], init_biases, trainable)
				conv = convolve(input, kernel)
				if relu:
					bias = tf.nn.bias_add(conv, biases)
					return tf.nn.relu(bias)
				return tf.nn.bias_add(conv, biases)
			else:
				conv = convolve(input, kernel)
				if relu:
					return tf.nn.relu(conv)
				return conv			
		
	@layer
	def relu(self, input, name):
		return tf.nn.relu(input, name=name)

	@layer
	def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
		self.validate_padding(padding)
		return tf.nn.max_pool(input,
							  ksize=[1, k_h, k_w, 1],
							  strides=[1, s_h, s_w, 1],
							  padding=padding,
							  name=name)
						
	@layer
	def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
		self.validate(padding)
		return tf.nn.avg_pool(input, 
							  kszie=[1,k_h,k_w,1],
							  strides=[1,s_h,s_w,1],
							  padding=padding,
							  name=name)

	@layer
	def concat(self, inputs, axis, name):
		return tf.concat(axis=axis, values=inputs, name=name)

	@layer
	def lrn(self, input, radius, alpha, beta, name, bias=1.0):
		return tf.nn.local_response_normalization(input,
													depth_radius=radius,
													alpha=alpha,
													beta=beta,
													bias=bias,
													name=name)

	@layer
	def fc(self, input, num_out, name, relu=True, trainable=True, stddev=0.01):
		with tf.variable_scope(name) as scope:
			if isinstance(input, tuple):
				input=input[0]
			input_shape = input.get_shape()
			if input_shape.ndims == 4:
				dim = 1
				for d in input_shape[1:].as_list():
					dim *= d
				feed_in = tf.reshape(tf.transpose(input, [0,3,1,2]),[-1,dim])
			else:
				feed_in, dim = (input, int(input_shape[-1]))
		
			init_weights = tf.truncated_normal_initializer(0.0, stddev=stddev)
			init_biases = tf.constant_initializer(0.0)

			weights = self.make_var('weights', [dim, num_out], init_weights, trainable, \
				regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))			
			if cfg.WRITE_TO_SUMMARY or cfg.DEBUG_ALL:	
				self.variable_summaries(weights)
				
			biases = self.make_var('biases', [num_out], init_biases, trainable)
			if cfg.WRITE_TO_SUMMARY or cfg.DEBUG_ALL:	
				self.variable_summaries(biases)

			op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
			fc = op(feed_in, weights, biases, name=scope.name)
			return fc

	@layer
	def add(self,input,name):
		return tf.add(input[0],input[1], name=name)			
			
	@layer
	def batch_normalization(self,input,name,relu=True, is_training=0):
		"""contribution by miraclebiu"""
		trainingPhase = False
		if is_training == 1:
			trainingPhase = True
			
		if relu:
			temp_layer=tf.contrib.layers.batch_norm(input,scale=True,center=True,is_training=trainingPhase,scope=name)
			return tf.nn.relu(temp_layer)
		else:
			return tf.contrib.layers.batch_norm(input,scale=True,center=True,is_training=trainingPhase,scope=name)

	@layer
	def scale(self, input, c_in, name):
		with tf.variable_scope(name) as scope:

			alpha = tf.get_variable('alpha', shape=[c_in, ], dtype=tf.float32,
									initializer=tf.constant_initializer(1.0), trainable=True,
									regularizer=self.l2_regularizer(0.00001))
			beta = tf.get_variable('beta', shape=[c_in, ], dtype=tf.float32,
									initializer=tf.constant_initializer(0.0), trainable=True,
									regularizer=self.l2_regularizer(0.00001))
			return tf.add(tf.multiply(input, alpha), beta)

	def l2_regularizer(self, weight_decay=0.0005, scope=None):
		def regularizer(tensor):
			with tf.name_scope(scope, default_name='l2_regularizer', values=[tensor]):
				l2_weight = tf.convert_to_tensor(weight_decay,
												dtype=tensor.dtype.base_dtype,
												name='weight_decay')
				return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')
		return regularizer
			
	@layer
	def softmax(self, input, name):
		return tf.nn.softmax(input, name=name)

	@layer
	def dropout(self, input, keep_prob, name):
		return tf.nn.dropout(input, keep_prob, name=name)

	## Added for SRAM tests ~~~~~~~ .	
	@layer
	def sigmoid(self, input, name):
		 return tf.nn.sigmoid(input, name=name)		
		
	@layer
	def binaryActivation(self, input, name):
		with tf.variable_scope(name) as scope:		
			cond_name = scope.name+'_less'
			where_name = scope.name+'_where'
			cond = tf.less(input, tf.zeros(tf.shape(input)), name=cond_name)
			ba = tf.where(cond, tf.zeros(tf.shape(input)), tf.ones(tf.shape(input)), name=where_name)
			return ba

	@layer
	def pad_leNet(self, input, name):
		with tf.variable_scope(name) as scope:
			return tf.pad(input, [[0,0],[2,2],[2,2],[0,0]], mode='CONSTANT', name=scope.name)	
	@layer
	def reshape_leNet(self, input, name):
		with tf.variable_scope(name) as scope:
			return tf.reshape(input, [-1,28,28,1],name=scope.name)

	@layer
	def resize(self, input, new_height, new_width, name):
		with tf.variable_scope(name) as scope:
			return tf.image.resize_images(input, [new_height, new_width])

	@layer
	def reshape_and_resize(self, input, new_height, new_width, name):
		"""
		Function to take in a 3D image, resize it, and then flatten it to 1D tensor.
		"""
		with tf.variable_scope(name) as scope:
			images_ = tf.image.resize_images(input, [new_height, new_width])		 
			dim = new_height*new_width		 
			return tf.reshape(images_,[-1, dim], name=scope.name)

	@layer
	def sum_fc_ops(self, input, name):
		with tf.variable_scope(name) as scope:
			return input[0] + input[1]

	def pre_process_image_train(self, image, img_size_h, img_size_w):	
		with tf.variable_scope('pre_process_train') as scope:
			image = tf.random_crop(image, size=[img_size_h, img_size_w, 3])
			image = tf.image.random_flip_left_right(image)
			image = tf.image.random_hue(image, max_delta=0.05)
			image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
			image = tf.image.random_brightness(image, max_delta=0.2)
			image = tf.image.random_saturation(image, lower=0.0, upper=2.0)
			image = tf.image.per_image_standardization(image)		
			image.set_shape([img_size_h, img_size_w, 3])
			return image		
	
	def pre_process_image_test(self, image, img_size_h, img_size_w):
		with tf.variable_scope('pre_process_test') as scope:
			image = tf.image.resize_image_with_crop_or_pad(image,
											   target_height=img_size_h,
											   target_width=img_size_w)
			image = tf.image.per_image_standardization(image)	
			image.set_shape([img_size_h, img_size_w, 3])	
			return image
	
	@layer
	def pre_process(self, input, name):
		with tf.variable_scope('name') as scope:
			images = tf.cond(input[1] > 0, lambda: tf.map_fn(lambda image: self.pre_process_image_train(image, 24, 24), input[0]), 
						lambda: tf.map_fn(lambda image: self.pre_process_image_test(image, 24, 24), input[0]))	
			return images

	


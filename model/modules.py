import tensorflow as tf 
import numpy as np 

def weight_init(shape, name = None):
	'''
	Returns: 4D tensor of shape [filter_height, filter_width, in_channels, out_channels]
	It just initializes the weights of the filters using the truncated normal distro
	'''
	W = tf.get_variable(name, initializer = tf.truncated_normal(shape = shape, stddev = 0.1))
	return W


def bias_init(shape, name = None):	
	'''
	Defining the bias for each filter count set to 0.1
	'''
	B = tf.get_variable(name, shape = shape, initializer = tf.constant_initializer(0.1))
	return B

def activation(act_fun='relu', input_volume):
	if input_volume:
		if act_fun == 'relu':
			try:
				return tf.nn.relu(input_volume)
			except:
				print("Error for the activation")
		elif act_fun == 'sigmoid':
			try:
				return tf.nn.sigmoid(input_volume)
			except:
				print("Error for the activation")

def conv2d(idx, name, input_volume, kernel_size, activation):
	"""
	Perform 2D convolutions 
	Input: 
		input_volume shape [batch, in_height, in_width, in_channels]
		stride : int
		padding : int
		idx : index number
		kernel_size : [filter_height, filter_width, in_channels, out_channels]
		filter 
	Output: output_volume shape [batch, out_height, out_width, out_channels]
	"""

	print("Layer {}, Name {}, Input Shape {}\n".format(str(idx), str(name), str(input_volume)))

	with tf.variable_scope(name):
		weights = weight_init(kernel_size, 'W')
		bias = bias_init(kernel_size[3], 'B')
		# strides is the list of int (1d tensor) for each dimension
		strides = [1, stride, stride, 1]
		conv = tf.nn.conv2d(input = input_volume, filter = weights, strides = strides, padding = 'SAME')
		res = tf.add(conv, bias)

		# to apply the relu do this:
		if not activation:
			return res
		else:
			return activation(activation, res)

def maxpool2d(idx, input_volume, kernel_size = 2, stride = 2):
	strides = [1, stride, stride, 1]
	ksize = [1,kernel_size, kernel_size, 1]
	maxpool = tf.nn.max_pool(value = input_volume, ksize = ksize, strides = stride, padding = 'SAME')
	return maxpool

def upsampling2d(idx, name, input_volume, kernel_size = 2, name = None):
	"""
	Read this blog http://www.mazhixian.me/2018/01/27/upsampling-for-2D-convolution-by-tensorflow/
	Using the keras upsampling 
	"""
	upsample = tf.keras.layers.UpSampling2D(size = kernel_size)(input_volume)
	return upsample

def merge(idx, convlist, dim, name):
	"""
	To concat the given list of tensors across the dimension specified
	"""
	print("Concatenating with index {}".format(str(idx)))
	try:
		concat2d = tf.concat(values = convlist, axis = dim)
		return concat2d
	except:
		print("Error with the concatenation")	
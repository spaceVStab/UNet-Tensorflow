import tensorflow as tf 
import numpy as np 

from modules import *

class UNet(object):
	def __init__(self):
		"""
		Implementing the UNet Architecture
		"""		
		# channel last
		self.x = width
		self.y = height
		self.channel = 3

		self.inputs = tf.placeholder(tf.float32, shape = [None, self.x, self.y, 3])
		# inputs shape [batch_size, x, y, channel]
		# kernel_size shape [kernel_x, kernel_y, in_channels, out_channels]
		self.conv2d_1_1 = conv2d(1, name = 'conv2d_1_1', input_volume = self.inputs, kernel_size = [3,3,3,64], activation = 'relu')
		self.conv2d_1_2 = conv2d(2, name = 'conv2d_1_2', input_volume = self.conv2d_1_1, kernel_size = [3,3,64,64], activation = 'relu')
		self.maxpool2d_1 = maxpool2d(3, input_volume = self.conv2d_1_2)

		self.conv2d_2_1 =  conv2d(4, name = 'conv2d_2_1', input_volume = self.maxpool2d_1, kernel_size = [3,3,64,128], activation = 'relu')
		self.conv2d_2_2 =  conv2d(5, name = 'conv2d_2_2', input_volume = self.conv2d_2_1, kernel_size = [3,3,128,128], activation = 'relu')
		self.maxpool2d_2 =  maxpool2d(6, input_volume = self.conv2d_2_2)

		self.conv2d_3_1 =  conv2d(7, name = 'conv2d_3_1', input_volume = self.maxpool2d_2, kernel_size = [3,3,128,256], activation = 'relu')
		self.conv2d_3_2 =  conv2d(8, name = 'conv2d_3_2', input_volume = self.conv2d_3_1, kernel_size = [3,3,256,256], activation = 'relu')
		self.maxpool2d_3 =  maxpool2d(9, input_volume = self.conv2d_3_2)

		self.conv2d_4_1 =  conv2d(10, name = 'conv2d_4_1', input_volume = self.maxpool2d_3, kernel_size = [3,3,256,512], activation = 'relu')
		self.conv2d_4_2 =  conv2d(11, name = 'conv2d_4_2', input_volume = self.conv2d_4_1, kernel_size = [3,3,512,512], activation = 'relu')
		# apply dropout
		self.maxpool2d_4 =  maxpool2d(12, input_volume = self.conv2d_4_2)

		self.conv2d_5_1 =  conv2d(13, name = 'conv2d_5_1', input_volume = self.maxpool2d_4, kernel_size = [3,3,512,1024], activation = 'relu')
		self.conv2d_5_2 =  conv2d(14, name = 'conv2d_5_2', input_volume = self.conv2d_5_1, kernel_size = [3,3,1024,1024], activation = 'relu')
		# apply dropout

		self._up2d_6 =  upsampling2d(15, input_volume = self.conv2d_5_2)
		self.up2d_6 =  conv2d(16, name = 'up2d_6', input_volume = self._up2d_6, kernel_size = [3,3,1024,512], activation = 'relu')
		self.merge_6 =  merge(17, [self.up2d_6, self.conv2d_4_2], dim = 3)
		self.conv2d_6_1 =  conv2d(18, name = 'conv2d_6_1', input_volume = self.merge_6, kernel_size = [3,3,1024,512], activation = 'relu')
		self.conv2d_6_2 =  conv2d(19, name = 'conv2d_6_2', input_volume = self.conv2d_6_1, kernel_size = [3,3,512,512], activation = 'relu')


		self._up2d_7 =  upsampling2d(20, input_volume = self.conv2d_6_2)
		self.up2d_7 =  conv2d(21, name = 'up2d_7', input_volume = self._up2d_7, kernel_size = [3,3,512,256], activation = 'relu')
		self.merge_7 =  merge(22, [self.up2d_7, self.conv2d_3_2], dim = 3)
		self.conv2d_7_1 =  conv2d(23, name = 'conv2d_7_1', input_volume = self.merge_7, kernel_size = [3,3,512,256], activation = 'relu')
		self.conv2d_7_2 =  conv2d(24, name = 'conv2d_7_2', input_volume = self.conv2d_7_1, kernel_size = [3,3,256,256], activation = 'relu')

		self._up2d_8 =  upsampling2d(25, input_volume = self.conv2d_7_2)
		self.up2d_8 =  conv2d(26, name = 'up2d_8', input_volume = self._up2d_8, kernel_size = [3,3,256,128], activation = 'relu')
		self.merge_8 =  merge(27, [self.up2d_8, self.conv2d_2_2], dim = 3)
		self.conv2d_8_1 =  conv2d(28, name = 'conv2d_8_1', input_volume = self.merge_8, kernel_size = [3,3,256,128], activation = 'relu')
		self.conv2d_8_2 =  conv2d(29, name = 'conv2d_8_2', input_volume = self.conv2d_8_1, kernel_size = [3,3,128,128], activation = 'relu')

		self._up2d_9 =  upsampling2d(25, input_volume = self.conv2d_8_2)
		self.up2d_9 =  conv2d(26, name = 'up2d_9', input_volume = self._up2d_9, kernel_size = [3,3,128,64], activation = 'relu')
		self.merge_9 =  merge(27, [self.up2d_9, self.conv2d_1_2], dim = 3)
		self.conv2d_9_1 =  conv2d(28, name = 'conv2d_9_1', input_volume = self.merge_9, kernel_size = [3,3,128,64], activation = 'relu')
		self.conv2d_9_2 =  conv2d(29, name = 'conv2d_9_2', input_volume = self.conv2d_9_1, kernel_size = [3,3,64,64], activation = 'relu')

		self.conv2d_10_1 =  conv2d(30, name = 'conv2d_10_1', input_volume = self.conv2d_9_2, kernel_size = [3,3,64,2], activation = 'relu')
		self.conv2d_10_2 =  conv2d(31, name = 'conv2d_10_2', input_volume = self.conv2d_10_1, kernel_size = [1,1,2,1], activation = 'sigmoid')

		return 

	def train_data(self):
		""" 
		To traint the data
		"""
		pass

	def predict_test_data(self):
		"""
		To predict the segmentation out of test data
		"""
		pass

	def evaluate(self):
		pass





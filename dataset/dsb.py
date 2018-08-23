# this is the script to handle the dataset for the data science bowl 2018; divergent nuclei images

import numpy as np 
import skimage.io 
import matplotlib.pyplot as plt 
from skimage import transform 
import os
from tqdm import tqdm 
import tensorflow as tf 
from subprocess import check_output

# use the options 
INPUT_DIR = '/path/to/image/data'

def read_image(image_id):
	image_file = INPUT_DIR + "/" + str(image_id) + "/images/" + str(image_id) + ".png"
	mask_file = INPUT_DIR + "/" + str(image_id) + "/masks/*.png"
	image = skimage.io.imread(image_file)
	masks = skimage.io.imread_collection(mask_file).concatenate()
	height, width, _ = image.shape
	num_mask = masks.shape[0]
	# defining the labels as numpy array to feed in masks
	labels = np.zeros((height, width), np.uint16)
	for i in range(num_mask):
		labels[masks[index] > 0] = i + 1
	return image, labels

image_ids = check_output(["ls",INPUT_DIR]).decode('utf8').split()

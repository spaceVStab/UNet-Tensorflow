# this is the script to handle the dataset for the data science bowl 2018; divergent nuclei images

import numpy as np 
from skimage.io import imread, imshow, imread_collection, concatenate_images
import matplotlib.pyplot as plt 
from skimage import transform 
import os
from tqdm import tqdm 
import tensorflow as tf 
from subprocess import check_output

"""
path/to/dataset => home/DSB
home/DSB
	-> train
		-> {sample_id_folder}
			-> images
				-> *.png
			-> masks
				-> *.png
	-> test
		-> {sample_id_folder}
			-> images
				-> *.png
			-> masks
				-> *.png
"""

INPUT_DIR = 'path/to/dataset'
TRAIN_PATH = os.path.join(INPUT_DIR, 'train')
TEST_PATH = os.path.join(INPUT_DIR, 'test')
img_h = 128
img_w = 128
img_c = 3

train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

def get_train_data():
	images = np.zeros((len(train_ids), img_h, img_w, img_c), dtype = np.uint8)
	labels = np.zeros((len(train_ids), img_h, img_w, 1), dtype = np.bool)

	print("train images and masks")

	for n, _id in tqdm(enumerate(train_ids), total=len(train_ids)):
	    path = os.path.join(TRAIN_PATH, str(_id))
	    img = imread(os.path.join(path,'images','{}.png'.format(_id)))[:,:,:img_c]
	    img = resize(img, (img_h, img_w), mode = 'constant', preserve_range=True)
	    images[n] = img
	    
	    mask = np.zeros((img_height,img_width,1), dtype = np.bool)
	    for maskpath in next(os.walk(os.path.join(path,'masks')))[2]:
	        _mask = imread(os.path.join(path, 'masks', maskpath))
	        _mask = np.expand_dims(resize(_mask, (img_h, img_w), mode = 'constant', preserve_range=True), axis = -1)
	        mask = np.maximum(mask, _mask)
	    labels[n] = mask

	return images, labels	    

def get_test_data():
	images = np.zeros((len(test_ids), img_h, img_w, img_c), dtype=np.uint8)
	test_img_sizes = []

	for n, _id in tqdm(enumerate(test_ids), total=len(test_ids)):
	    path = os.path.join(TEST_PATH, str(_id))
	    img = imread(os.path.join(path,'images','{}.png'.format(_id)))[:,:,:img_c]
	    test_img_sizes.append([img.shape[0], img.shape[1]])
	    img = resize(img, (img_h, img_w), mode = 'constant', preserve_range=True)
	    images[n] = img

	return images	    

def sample_train_img_mask(image_id):
	image_path = os.path.join(TRAIN_PATH,'{}/images/{}.png'.format(str(image_id)))
	image = skimage.io.imread(image_path)
	skimage.io.imshow(image)

	mask_path = os.path.join(TRAIN_PATH, "{}/masks/*.png".format(str(image_id)))
	masks = skimage.io.imread_collection(mask_path).concatenate()
	skimage.io.imshow_collection(masks)
	plt.show()
	return 
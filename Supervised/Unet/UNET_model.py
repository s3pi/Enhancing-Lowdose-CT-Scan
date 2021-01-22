import random
from keras.layers.core import *
import sys
import cv2
import nibabel as nib
import _pickle as cPickle
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose,ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
import numpy as np
import scipy.misc
import numpy.random as rng
from PIL import Image, ImageDraw, ImageFont
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Input, concatenate, Add
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D,Convolution2D
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.models import model_from_json,model_from_config,load_model
from keras.optimizers import SGD,RMSprop,adam,Adam
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.preprocessing import image
from keras import backend as K
from keras.initializers import random_uniform, RandomNormal
from sklearn.utils import shuffle
# from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from collections import OrderedDict as od
import tensorflow as tf
import os
import math
from contextlib import redirect_stdout
import csv
import pylab as plt

def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
    	padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
    	padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x
  
def get_unet(input_tensor, n_filters = 32, dropout = 0.1, batchnorm = True):
	# Contracting Path
	c1 = conv2d_block(input_tensor, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
	p1 = MaxPooling2D((2, 2))(c1)
	p1 = Dropout(dropout)(p1)

	c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
	p2 = MaxPooling2D((2, 2))(c2)
	p2 = Dropout(dropout)(p2)

	c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
	p3 = MaxPooling2D((2, 2))(c3)
	p3 = Dropout(dropout)(p3)

	c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
	p4 = MaxPooling2D((2, 2))(c4)
	p4 = Dropout(dropout)(p4)

	c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)

	# Expansive Path
	u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
	u6 = concatenate([u6, c4])
	u6 = Dropout(dropout)(u6)
	c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)

	u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
	u7 = concatenate([u7, c3])
	u7 = Dropout(dropout)(u7)
	c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)

	u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
	u8 = concatenate([u8, c2])
	u8 = Dropout(dropout)(u8)
	c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)

	u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
	u9 = concatenate([u9, c1])
	u9 = Dropout(dropout)(u9)
	c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)

	outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
	model = Model(inputs=[input_tensor], outputs=[outputs])

	return model

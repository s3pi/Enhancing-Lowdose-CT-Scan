# from UNET_model import get_unet
from tensorflow.keras.layers import Input,Dense,Flatten,Dropout,Reshape,Conv3D,MaxPooling2D,UpSampling2D,Conv2DTranspose,ZeroPadding2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Input, concatenate, Add, Permute
from tensorflow.keras.models import Model, Sequential
from contextlib import redirect_stdout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import BatchNormalization
import os
import numpy as np 
from sklearn.utils import shuffle
import math
from skimage.metrics import structural_similarity as ssim
import cv2
import random
import time
# import Model_ViT
import tensorflow as tf
# from ViT import vit, utils, layers
import copy


def make_model(d, h, w, c):
    # Making all the modules of the model architecture
    inp_shape = (d, h, w, c)
    x = Input(inp_shape)
    lamda = tf.Variable(initial_value=0., trainable=True)
    
    # x is the represents input_tensor
    v_of_x = Conv3D(filters=c, kernel_size=3, padding='same', activation='relu')(x)
    v_of_x = tf.reshape(v_of_x, (-1, d, h*w, c)) # N = h*w
    v_of_x = tf.transpose(v_of_x, perm=[0, 3, 2, 1]) #channel first
    v_of_x_T = tf.transpose(v_of_x, perm=[0, 1, 3, 2])

    c_hat = c - less_than_c_by
    k_of_x = Conv3D(filters=c_hat, kernel_size=1, padding='same', activation='relu')(x)
    q_of_x = Conv3D(filters=c_hat, kernel_size=1, padding='same', activation='relu')(x)
    k_of_x = tf.reshape(k_of_x, (-1, d, h*w, c_hat)) # N = h*w
    k_of_x = tf.transpose(k_of_x, perm=[0, 3, 2, 1]) # channel first
    q_of_x = tf.reshape(q_of_x, (-1, d, h*w, c_hat))
    q_of_x = tf.transpose(q_of_x, perm=[0, 3, 2, 1])

    k_of_x_T = tf.transpose(k_of_x, perm=[0, 1, 3, 2]) 
    q_of_x_T = tf.transpose(q_of_x, perm=[0, 1, 3, 2]) 
    
    plane_att = tf.matmul(k_of_x, q_of_x_T)
    plane_att = tf.nn.softmax(plane_att)
    depth_att = tf.matmul(q_of_x_T, k_of_x)
    depth_att = tf.nn.softmax(depth_att)

    plane_att = tf.matmul(plane_att, v_of_x)
    depth_att = tf.matmul(depth_att, v_of_x_T)
    depth_att = tf.transpose(depth_att, perm=[0, 1, 3, 2])
   
    fuse_att = lamda * (plane_att+depth_att) + v_of_x
    fuse_att = tf.transpose(fuse_att, perm=[0, 3, 2, 1])
    fuse_att = tf.reshape(fuse_att, (-1, d, h, w, c))

    model = Model(inputs=[x], outputs=[fuse_att])

    with open('self_attention.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()

    return model


# #--------------------------------------------------------------------------------------------------------
# # Arguments:
# # train_subs = list(set(os.listdir(pro_low_dose_data_path)) - set(valid_subs + test_subs))
# #--------------------------------------------------------------------------------------------------------
# # log_file_path = local_path + "/log_file.txt"
# # with open(log_file_path, 'a') as log_file:
# #     log_file.write("-----------------------------------------\nProgram Starts")
# #--------------------------------------------------------------------------------------------------------
less_than_c_by = 0
model = make_model(d = 3, h = 512, w = 512, c = 32)
# #--------------------------------------------------------------------------------------------------------

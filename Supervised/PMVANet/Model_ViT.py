import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, concatenate, BatchNormalization, PReLU, MaxPool2D, GlobalAveragePooling2D, Dropout, Dense, Lambda
from tensorflow.keras.regularizers import l2
from ViT import vit, utils

def normalisation_layer(x):
    return(tf.nn.l2_normalize(x, 1, 1e-10))
# image_size = (256, 512)
def create_model_1(input_shape):
    model = vit.vit_l32_1(
    image_size=(input_shape[1], input_shape[2]),
    activation='linear',
    pretrained=False,
    include_top=False,
    pretrained_top=False
    )
    return model

def create_model_2(input_shape):
	model = vit.vit_l32_2(
	image_size=(input_shape[1], input_shape[2]),
	activation='linear',
	pretrained=False,
	include_top=False,
	pretrained_top=False
	)
	return model

def create_model_3(input_shape):
    model = vit.vit_l32_3(
    image_size=(input_shape[1], input_shape[2]),
    activation='linear',
    pretrained=False,
    include_top=False,
    pretrained_top=False
    )
    return model


def create_model_4(input_shape):
    model = vit.vit_l32_4(
    image_size=(input_shape[1], input_shape[2]),
    activation='linear',
    pretrained=False,
    include_top=False,
    pretrained_top=False
    )
    return model
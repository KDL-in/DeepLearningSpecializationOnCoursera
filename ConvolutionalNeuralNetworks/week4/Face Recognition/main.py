from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
import sys



import funcs
import importlib

importlib.reload(funcs)
from funcs import *



np.set_printoptions(threshold=sys.maxsize)



with tf.Session() as test:
    tf.set_random_seed(1)
    y_true = (None, None, None)
    y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed = 1),
              tf.random_normal([3, 128], mean=1, stddev=1, seed = 1),
              tf.random_normal([3, 128], mean=3, stddev=4, seed = 1))
    loss = triplet_loss(y_true, y_pred)
    
    print("loss = " + str(loss.eval()))

FRmodel = faceRecoModel(input_shape=(3, 96, 96))
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)

# 模型可视化
# from keras.utils import plot_model
# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
# plot_model(FRmodel, to_file='model.svg')
# SVG(model_to_dot(FRmodel).create(prog='dot', format='svg'))
FRmodel.summary()
# transpose的使用
A = np.array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7]],

           [[ 8,  9, 10, 11],
            [12, 13, 14, 15]]])
# 如何读懂
# 多维数组本质上是一个一维数组，比如A有两个元素，每个元素都是个二维数组
# 0 2 1表示，0维不动，1 2维转置，相当于其中二维转置
# 其实更加抽象的读法是 l h w c -> 3 0 1 2 直接读就可以了c h w c
A.transpose((0,2,1))

database = {}
database["danielle"] = img_to_encoding("images/danielle.png", FRmodel)
database["younes"] = img_to_encoding("images/younes.jpg", FRmodel)
database["tian"] = img_to_encoding("images/tian.jpg", FRmodel)
database["andrew"] = img_to_encoding("images/andrew.jpg", FRmodel)
database["kian"] = img_to_encoding("images/kian.jpg", FRmodel)
database["dan"] = img_to_encoding("images/dan.jpg", FRmodel)
database["sebastiano"] = img_to_encoding("images/sebastiano.jpg", FRmodel)
database["bertrand"] = img_to_encoding("images/bertrand.jpg", FRmodel)
database["kevin"] = img_to_encoding("images/kevin.jpg", FRmodel)
database["felix"] = img_to_encoding("images/felix.jpg", FRmodel)
database["benoit"] = img_to_encoding("images/benoit.jpg", FRmodel)
database["arnaud"] = img_to_encoding("images/arnaud.jpg", FRmodel)


verify("images/camera_0.jpg", "younes", database, FRmodel)
verify("images/camera_2.jpg", "kian", database, FRmodel)
who_is_it("images/camera_0.jpg", database, FRmodel)
# 代码有问题，可能有些版本或者完整性问题，直接做线上的吧
# 


import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf


model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
print(model)
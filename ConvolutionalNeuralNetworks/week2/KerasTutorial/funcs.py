import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *
def sep(s):
    print(f"\n================{s}=================\n")
# GRADED FUNCTION: HappyModel

def HappyModel(input_shape):
    """
    Implementation of the HappyModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    
    ### START CODE HERE ###
    # Feel free to use the suggested outline in the text above to get started, and run through the whole
    # exercise (including the later portions of this notebook) once. The come back also try out other
    # network architectures as well. 
    
    # placeholder
    X_input = Input(input_shape)
    # padding
    X = ZeroPadding2D((3,3))(X_input)
    # convolve
    X = Conv2D(32,(7,7),strides=(1,1),name='conv0')(X)
    # normalize
    X = BatchNormalization(axis=3,name='bn0')(X)
    # active
    X =Activation('relu')(X)
    # pooling
    X = MaxPooling2D((2,2),name='max_pool')(X)
    # fullyconnect
    X = Flatten()(X)
    X = Dense(1,activation='sigmoid',name='fc')(X)
    model = Model(inputs = X_input,outputs = X,name = 'HappyModel')

    ### END CODE HERE ###
    
    return model
# import the necessary packages
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers import Input
from keras.regularizers import l2
from keras import backend as K

class SmallVGGNet:
    @staticmethod
    def build(height, width, depth, classes, reg = 0.0005):
        # set the input shape to match the channel ordering
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            channel_dim = 1
        else:
            input_shape = (height, width, depth)
            channel_dim = -1

        # first (and only) CONV => RELU => POOL block
        inpt = Input(shape = input_shape)
        x = Conv2D(32, (3, 3), padding = "same", kernel_regularizer = l2(reg))(inpt)
        x = Activation("relu")(x)
        x = BatchNormalization(axis = channel_dim)(x)
        x = MaxPooling2D(pool_size = (2, 2))(x)
        x = Dropout(0.25)(x)

        # first CONV => RELU => CONV => RELU => POOL block
        x = Conv2D(64, (3, 3), padding = "same", kernel_regularizer = l2(reg))(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis = channel_dim)(x)
        x = Conv2D(64, (3, 3), padding = "same", kernel_regularizer = l2(reg))(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis = channel_dim)(x)
        x = MaxPooling2D(pool_size = (2, 2))(x)
        x = Dropout(0.25)(x)

        # second CONV => RELU => CONV => RELU => POOL block
        x = Conv2D(128, (3, 3), padding = "same", kernel_regularizer = l2(reg))(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis = channel_dim)(x)
        x = Conv2D(128, (3, 3), padding = "same", kernel_regularizer = l2(reg))(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis = channel_dim)(x)
        x = MaxPooling2D(pool_size = (2, 2))(x)
        x = Dropout(0.25)(x)

        # flatten layer
        fl_out = Flatten()(x)
        
        # binary classifier
        bin_classifier = Dense(64, kernel_regularizer = l2(reg))(fl_out)
        bin_classifier = Activation("relu")(bin_classifier)
        bin_classifier = BatchNormalization(axis = channel_dim)(bin_classifier)
        bin_classifier = Dense(1, kernel_regularizer = l2(reg))(bin_classifier)
        bin_classifier = Activation("sigmoid", name = "bin_classifier")(bin_classifier)

        # regression head
        reg_head = Dense(64, kernel_regularizer = l2(reg))(fl_out)
        reg_head = Activation("relu")(reg_head)
        reg_head = BatchNormalization(axis = channel_dim)(reg_head)
        reg_head = Dense(1, name = "reg_head", kernel_regularizer = l2(reg))(reg_head)

        # return the constructed network architecture
        model = Model(inputs = inpt, outputs = [bin_classifier, reg_head])
        return model

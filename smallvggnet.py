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
from keras import backend as K

class SmallVGGNet:
    @staticmethod
    def build(height, width, depth, classes):
        # set the input shape to match the channel ordering
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            channel_dim = 1
        else:
            input_shape = (height, width, depth)
            channel_dim = -1

        # first (and only) CONV => RELU => POOL block
        inpt = Input(shape = input_shape)
        x = Conv2D(32, (3, 3), padding = "same")(inpt)
        x = Activation("relu")(x)
        x = BatchNormalization(axis = channel_dim)(x)
        x = MaxPooling2D(pool_size = (3, 3))(x)
        x = Dropout(0.25)(x)

        # first CONV => RELU => CONV => RELU => POOL block
        x = Conv2D(64, (3, 3), padding = "same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis = channel_dim)(x)
        x = Conv2D(64, (3, 3), padding = "same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis = channel_dim)(x)
        x = MaxPooling2D(pool_size = (2, 2))(x)
        x = Dropout(0.25)(x)

        # second CONV => RELU => CONV => RELU => POOL block
        x = Conv2D(128, (3, 3), padding = "same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis = channel_dim)(x)
        x = Conv2D(128, (3, 3), padding = "same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis = channel_dim)(x)
        x = MaxPooling2D(pool_size = (2, 2))(x)
        x = Dropout(0.25)(x)

        # first (and only) FC layer
        x = Flatten()(x) # Change to GlobalMaxPooling2D
        x = Dense(1024)(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis = channel_dim)(x)
        FC_out = Dropout(0.5)(x)

        # binary classifier
        bin_classifier = Dense(1)(FC_out)
        bin_classifier = Activation("sigmoid")(bin_classifier)

        # regression head
        reg_head = Dense(1)(FC_out)

        # return the constructed network architecture
        model = Model(inputs = inpt, outputs = [bin_classifier, reg_head])
        return model

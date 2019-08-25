from keras.models import Model, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os


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
        bin_classifier = Activation("sigmoid",name = 'bin_classifier')(bin_classifier)

        # regression head
        reg_head = Dense(1,name = 'reg_head')(FC_out)

        model = Model(inputs=inpt, outputs=[bin_classifier, reg_head])
        return model


lossweights = {'reg_head': 1.0, 'bin_classifier': 1.0}
losses = {'reg_head': 'mse', 'bin_classifier': 'binary_crossentropy'}

image_file_path = 'CamCann_Dataset/image_data.npy'
resized_image_path  = 'CamCann_Dataset/Resized_Images'
csv_path = 'modified.csv'

##============Processing The CSV==============##
print("[INFO]Processing The CSV")


dataframe = pd.read_csv(csv_path, header= None)

genders = dataframe.iloc[:, 1]
lb = LabelBinarizer()
genders = lb.fit_transform(genders)
image_name = dataframe.iloc[:, 3]
age = dataframe.iloc[:, 2]
resized = []

# for file in os.listdir(resized_image_path):
#     resized.append(str(file))
# dataframe = dataframe[dataframe[2].isin(resized)]
# dataframe.to_csv('/home/harsha/CammCann/CamCann_Dataset/modified.csv', header = None)

##============Splitting The Data=============##

print("[INFO]Splitting The Data")

img_data = np.load(image_file_path)
img_train, img_valid, o2_train, o2_valid, o1_train, o1_valid = train_test_split(img_data, genders, age, test_size=0.2, random_state= 24 )

##==========Creating The Checkpoints=========##

filepath = "weight-improvement-{epoch:02d}-{loss:4f}.hd5"
checkpoint = ModelCheckpoint(filepath,monitor='val_bin_classifier_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

##============Setting Up the model============##

epochs = 50
batch_size = 64

model = SmallVGGNet.build(128, 128, 3, 1)
model.compile(optimizer= 'nadam', loss= losses, loss_weights = lossweights, metrics=["accuracy"])
model.summary()

#=============Training the Model==============##
print("[INFO]Training the Model")

model.fit(img_train, {'reg_head': o1_train, 'bin_classifier': o2_train}, batch_size = batch_size, epochs=epochs, callbacks= callbacks_list,  validation_data=(img_valid, {'reg_head': o1_valid, 'bin_classifier': o2_valid}))



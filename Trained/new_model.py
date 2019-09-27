# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:14:20 2019

@author: tanma
"""
import numpy as np, pandas as pd
from keras.layers import Dense, BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras.layers.core import Activation
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.applications.xception import Xception
from keras.ImagePreprocessing import ImageDataGenerator

model = Xception(include_top=False, weights='imagenet', input_shape=(128,128,3), pooling = 'max')

for layer in model.layers:
    layer.trainable = False

x = model.output

bin_classifier = Dense(64, kernel_regularizer = l2(0.0005))(x)
bin_classifier = Activation("relu")(bin_classifier)
bin_classifier = BatchNormalization(axis = -1)(bin_classifier)
bin_classifier = Dense(1, kernel_regularizer = l2(0.0005))(bin_classifier)
bin_classifier = Activation("sigmoid", name = "bin_classifier")(bin_classifier)

reg_head = Dense(64, kernel_regularizer = l2(0.0005))(x)
reg_head = Activation("relu")(reg_head)
reg_head = BatchNormalization(axis = -1)(reg_head)
reg_head = Dense(1, name = "reg_head", kernel_regularizer = l2(0.0005))(reg_head)

base_model = Model(inputs = model.input, outputs = [bin_classifier, reg_head])

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
checkpoint = ModelCheckpoint(filepath,monitor='bin_classifier_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

##============Setting Up the model============##

epochs = 50
batch_size = 16

base_model.compile(optimizer= 'nadam', loss= losses, loss_weights = lossweights, metrics=["accuracy"])
model.summary()

#=============Training the Model==============##
print("[INFO]Training the Model")

train_datagen = ImageDataGenerator(
rescale=1./255,
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

base_model.fit(img_train, {'reg_head': o1_train, 'bin_classifier': o2_train}, batch_size = batch_size, epochs=epochs, callbacks= callbacks_list,  validation_data=(img_valid, {'reg_head': o1_valid, 'bin_classifier': o2_valid}))

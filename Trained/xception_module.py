# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:49:14 2019

@author: tanma
"""
import numpy as np
from keras.layers import Dense, BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras.layers.core import Activation
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.applications.xception import Xception

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

images = np.load('imfdb_images.npy')
gender = np.load('imfdb_gender_labels.npy')
age = np.load('imfdb_age_labels.npy')

# normalize the input images into the range [0, 1]
images = images.astype("float") / 255.0

# split the data into train and test sets
(x_train, x_test, gender_train, gender_test, age_train, age_test) = train_test_split(images, gender, age, test_size = 0.2, random_state = 24)

# convert the labels from string to vectors
lb = LabelBinarizer()
gender_train = lb.fit_transform(gender_train)
gender_test = lb.fit_transform(gender_test)

# initialize the loss functions and the loss weights
loss_weights = {'reg_head': 1.0, 'bin_classifier': 2.0}
losses = {'reg_head': 'mse', 'bin_classifier': 'binary_crossentropy'}

# compiling the model
print("[INFO] compiling model...")
base_model.compile(optimizer = "nadam", loss = losses, loss_weights = loss_weights, metrics = ["accuracy"])

# initialize the callbacks
filepath = "weight-improvement-{epoch:02d}-{loss:4f}.hd5"
checkpoint = ModelCheckpoint(filepath, monitor = 'val_bin_classifier_acc', verbose = 1)
callbacks = [checkpoint]

# train the model
print("[INFO] training the model")
epochs = 50
batch_size = 128
base_model.fit(x_train, {'reg_head': age_train, 'bin_classifier': gender_train}, batch_size = batch_size, epochs=epochs, callbacks= callbacks,  validation_data=(x_test, {'reg_head': age_test, 'bin_classifier': gender_test}))
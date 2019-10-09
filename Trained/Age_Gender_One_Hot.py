# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 16:42:13 2019

@author: ADMIN
"""

import numpy as np, pandas as pd
from keras.layers import Dense, BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras.layers.core import Activation
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.applications.densenet import DenseNet121
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

model = DenseNet121(include_top=False, weights='imagenet', input_shape=(128,128,3), pooling = 'max')

model.trainable = False

set_trainable = False
for layer in model.layers:
  if layer.name.startswith("conv5"):
    set_trainable = True
  if set_trainable:
    layer.trainable = True
  else:
    layer.trainable = False

x = model.output

bin_classifier = Dense(64, kernel_regularizer = l2(0.05))(x)
bin_classifier = Activation("relu")(bin_classifier)
bin_classifier = BatchNormalization(axis = -1)(bin_classifier)
bin_classifier = Dense(1, kernel_regularizer = l2(0.5))(bin_classifier)
bin_classifier = Activation("sigmoid", name = "bin_classifier")(bin_classifier)

reg_head = Dense(64, kernel_regularizer = l2(0.5))(x)
reg_head = Activation("relu")(reg_head)
reg_head = BatchNormalization(axis = -1)(reg_head)
reg_head = Dense(9, name = "reg_head", kernel_regularizer = l2(0.5), activation = 'softmax')(reg_head)

base_model = Model(inputs = model.input, outputs = [bin_classifier, reg_head])

images = np.load('imfdb/imfdb_images.npy')
gender = np.load('imfdb/imfdb_gender_labels.npy')
age = np.load('imfdb/imfdb_age_labels.npy')

df = pd.DataFrame(columns = ['Values','Groups','Categories'])
df['Values'] = age.flatten()
labels = ["{0} - {1}".format(i, i + 6) for i in range(18, 66, 6)]
codes = [i for i in range(8)]
df['Groups'] = pd.cut(df.Values, range(18, 67, 6), right=False, labels=labels)
df['Categories'] = pd.cut(df.Values, range(18, 67, 6), right=False, labels=codes)

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(df['Categories'].values)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
age = onehot_encoder.fit_transform(integer_encoded)

images = images.astype("float") / 255.0

(x_train, x_test, gender_train, gender_test, age_train, age_test) = train_test_split(images, gender, age, test_size = 0.3, random_state = 24)

lb = LabelBinarizer()
gender_train = lb.fit_transform(gender_train)
gender_test = lb.fit_transform(gender_test)

loss_weights = {'reg_head': 10., 'bin_classifier': 1.}
losses = {'reg_head': 'categorical_crossentropy', 'bin_classifier': 'binary_crossentropy'}

base_model.compile(optimizer = "nadam", loss = losses, loss_weights = loss_weights, metrics = ["accuracy"])

filepath = "weight-improvement-{epoch:02d}-{loss:4f}.hd5"
checkpoint = ModelCheckpoint(filepath, monitor = 'val_reg_head_acc', verbose = 1, mode = 'max', save_best_only=True)
callbacks = [checkpoint]

epochs = 25
batch_size = 64

base_model.fit(x_train, {'reg_head': age_train, 'bin_classifier': gender_train}, batch_size = batch_size, epochs=epochs, callbacks= callbacks,  validation_data=(x_test, {'reg_head': age_test, 'bin_classifier': gender_test}))

base_model.save("base_model.h5")
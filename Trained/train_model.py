# import the necessary packages
from camcann.nn.conv import SmallVGGNet
from camcann.callbacks import TrainingMonitor
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import numpy as np
import argparse
import os

# construct the argument parser to parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True, help = "path to input dataset")
ap.add_argument("-o", "--output", required = True, help = "path to store output model files")
args = vars(ap.parse_args())

# load the data
print("[INFO] loading data...")
IMAGES_PATH = os.path.sep.join([args["dataset"], "images.npy"])
LABELS_PATH = os.path.sep.join([args["dataset"], "labels.npy"])
images = np.load(IMAGES_PATH)
labels = np.load(LABELS_PATH, allow_pickle = True)
gender = labels[0, :]
age = labels[1, :]

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
model = SmallVGGNet.build(128, 128, 3, 1)
model.compile(optimizer = "nadam", loss = losses, loss_weights = loss_weights, metrics = ["accuracy"])

# initialize the callbacks
OUTPUT_PATH = os.path.sep.join([args["output"], "model_{epoch:03d}.h5"])
checkpoint = ModelCheckpoint(OUTPUT_PATH, monitor = 'val_bin_classifier_acc', verbose = 1)
callbacks = [checkpoint]

# train the model
print("[INFO] training the model")
epochs = 50
batch_size = 128
model.fit(x_train, {'reg_head': age_train, 'bin_classifier': gender_train}, batch_size = batch_size, epochs=epochs, callbacks= callbacks,  validation_data=(x_test, {'reg_head': age_test, 'bin_classifier': gender_test}))

# import the necessary packages
from keras.applications import ResNet50
from keras.models import load_model, Model
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2

# construct the argument parser to parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--features", required = True, help = "path to features")
ap.add_argument("-i", "--image", required = True, help = "path to query image")
# ap.add_argument("-m", "--model", required = True, help = "path to model weights")
args = vars(ap.parse_args())

def euclidean_distance(arr_1, arr_2):
    return np.squeeze(np.sqrt(np.sum(np.square(np.subtract(arr_1, arr_2)), axis = -1)))

def extract_features(image):
    # initialize the pretrained ResNet model
    print("[INFO] initializing ResNet model")
    model = ResNet50(include_top = False, weights = "imagenet", input_shape = (128, 128, 3))

    # extract the facial features
    print("[INFO] extracting features")
    features = model.predict(image)

    # return the extracted features
    return features

# load the query image and resize it
print("[INFO] loading query image...")
query_image = cv2.imread(args["image"])

# load the features database
print("[INFO] loading features...")
features = np.load(args["features"])

# extract the features
query_image_features = extract_features(query_image)

# calculate the euclidean distance, sort the array in descending order and get the top 16 matches
dist = euclidean_distance(query_image_features, features)
dist = np.argsort(dist)
dist = dist[:16]

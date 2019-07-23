from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import cv2
import cvlib as cv
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = ap.parse_args()

image = cv2.imread(args.image)

if image is None:
    print("Could not read input image")
    exit()

model = load_model("gender_detection.model")

face, confidence = cv.detect_face(image)

classes = ['man','woman']

for idx, f in enumerate(face):
       
    (startX, startY) = f[0], f[1]
    (endX, endY) = f[2], f[3]

    cv2.rectangle(image, (startX,startY), (endX,endY), (0,255,0), 2)

    face_crop = np.copy(image[startY:endY,startX:endX])

    face_crop = cv2.resize(face_crop, (96,96))
    face_crop = face_crop.astype("float") / 255.0
    face_crop = img_to_array(face_crop)
    face_crop = np.expand_dims(face_crop, axis=0)

    conf = model.predict(face_crop)[0]
    print(conf)
    print(classes)

    idx = np.argmax(conf)
    label = classes[idx]

if label == 'man':
    cv2.imwrite('male_'+args.image,image)
else:
    cv2.imwrite('female_'+args.image,image)

cv2.destroyAllWindows()
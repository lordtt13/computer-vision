# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 13:15:02 2019

@author: tanma
"""
import cv2
import cvlib as cv
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

model = load_model('gender_detection.model')

def integrator(image):
    image = cv2.imread(image)
    face, confidence = cv.detect_face(image)
    
    for idx, f in enumerate(face):
       
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]
    
        cv2.rectangle(image, (startX,startY), (endX,endY), (0,255,0), 2)
    
        face_crop = np.copy(image[startY:endY,startX:endX])
    
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)
        # cv2.imwrite(args.image, face_crop)
    
        tags = model.predict(face_crop)[0]
        # print(conf)
        # print(classes)
    
        # idx = np.argmax(conf)
        # label = classes[idx]]
        
        return tags
    
print(integrator('..//download.jpeg'))
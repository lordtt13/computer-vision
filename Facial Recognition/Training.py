# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 03:12:37 2019

@author: enTropy
"""

import cv2
import numpy as np
import os
from PIL import Image
import argparse

##============ Constructing the Argument Parser ==================##
parser = argparse.ArgumentParser()
parser.add_argument("-p","--path", required = True, help = "Path of Images and Labels")
parser.add_argument("-x","--xmlpath", requred = True, help = "Path of the xml file for CascadeClassifier")
args = vars(parser.parse_args())
##================================================================##

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(args["xmlpath"])

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') 
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids

print ("\nTraining Faces")
faces,ids = getImagesAndLabels(args["path"])
recognizer.train(faces, np.array(ids))
recognizer.write('trained_model.yml') 

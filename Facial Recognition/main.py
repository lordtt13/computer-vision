# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 02:35:23 2019

@author: enTropy
"""

import cv2
import numpy as np
import os 

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trained_model.yml')
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
id = 0

names = ['None', 'Anjali', 'Atharva', 'Unnikrishnan', 'Anmol'] 

cap = cv2.VideoCapture(0)
cap.set(3, 640) 
cap.set(4, 480)
min_W = 0.1*cap.get(3)
min_H = 0.1*cap.get(4)
while True:
    ret, img =cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor = 1.2,minNeighbors = 5,minSize = (int(min_W), int(min_H)),)
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))        
        cv2.putText(img, str(id), (x+5,y-5),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 1)      
    cv2.imshow('camera',img) 
    k = cv2.waitKey(10) & 0xff
    if k == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

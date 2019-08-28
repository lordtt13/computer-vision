# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 00:55:13 2019

@author: enTropy
"""
import cv2
import imutils
from imutils.video import VideoStream
import os 
n=0
id_no=input("Enter the ID of the Person to Enroll : ")
detector=cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
vid=VideoStream(src=0).start()
while True:
    frame=vid.read()
    temp=frame.copy()
    frame=imutils.resize(frame, width=400)
    rects=detector.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in rects:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)    
    cv2.imshow("Frame", frame)
    key=cv2.waitKey(1) & 0xFF
    if key==ord("c"):
        cv2.imwrite("C:/Users/Unnikrishnan Menon/Desktop/Facial Recognition/Dataset/User."+id_no+".{}.png".format(str(n)),frame[y:y+h,x:x+w])
        n+=1
    elif key==ord("q"):
        break        
print("{} Images have been stored in the dataset!".format(n))
cv2.destroyAllWindows()
vid.stop()

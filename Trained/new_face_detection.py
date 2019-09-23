from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
from datetime import datetime
import pytz
import csv


# construct the argument parser to parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required = True, help = "path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required = True, help = "path to Caffe pretrained model")
ap.add_argument("-c", "--confidence", type = float, default = 0.5, help = "minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load the serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream
print("[INFO] initializing video stream...")
cap = cv2.VideoCapture("india.mp4")
time.sleep(2.0)
c=0
d=0
l=[]
z=[]

# loop over the frames from the video stream
while True:

    fps = FPS().start()
    # capture the frame and resize it to have a min size of 400
    ret, frame = cap.read()
    frame = imutils.resize(frame, width = 400)
    
    #Calculate time
    now = datetime.now()
    tz = pytz.timezone('Asia/Kolkata')
    now = now.replace(tzinfo=tz)
    time = now.astimezone(tz)
    t = str(time)[11:19]
    date = str(time)[0:10]
   


    # grab the frame dimensions and convert it into a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections
    net.setInput(blob)
    detections = net.forward()
    c=0

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence associated with each prediction
        confidence = detections[0, 0, i, 2]

        # filter out the weak detections
        if confidence > args["confidence"]:
            # compute the bounding box coordinates of the face object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype("int")
            

            # draw the bounding box on the image along with the confidence probability
            text = "{:.2f}%".format(confidence * 100)
            c=c+1
            y = start_y - 10 if start_y - 10 > 10 else start_y + 10
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
            cv2.putText(frame, text, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.putText(frame, str(c), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    fps.update()
    # show the frame
    if(d==0):
        z.append(c)
        print(z)
        d=1
    if(c>0 and c!=z[-1]):
        l.append([c, t, date])
        z.append(c)
        print(c)
        
    cv2.imshow("Output", frame)
    key = cv2.waitKey(1) & 0xFF

    # if 'q' was pressed, break from loop
    if key == ord('q'):
        break

fps.stop()

print("[INFO] Average FPS = {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
cap.release()

with open('result.csv','w') as f:
    writer = csv.writer(f)
    writer.writerows(l)
f.close()

# import the necessary packages
import numpy as np
import argparse
import cv2

# construct the argument parser to parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "path to input image")
ap.add_argument("-p", "--prototxt", required = True, help = "path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required = True, help = "path to Caffe pretrained model")
ap.add_argument("-c", "--confidence", type = float, default = 0.5, help = "minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load the serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# load an input image and construct an input blob from the image
# resize the blob to 128x128 and normalize it
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

# pass the blob through the network and obtain the detections
print("[INFO] computing face detections...")
net.setInput(blob)
detections = net.forward()

# initialize an empty list to store the detected faces
faces = []

# loop over the detections
for i in range(0, detections.shape[2]):
    # extract the confidence associated with each prediction
    confidence = detections[0, 0, i, 2]

    # filter out the weak detections
    if confidence > args["confidence"]:
        # compute the bounding box coordinates of the face object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (start_x, start_y, end_x, end_y) = box.astype("int")

        # extract the face from the image
        face = image[start_y:end_y, start_x:end_x, :]
        faces.append(face)

        # # draw the bounding box on the image along with the confidence probability
        # text = "{:.2f}%".format(confidence * 100)
        # y = start_y - 10 if start_y - 10 > 10 else start_y + 10
        # cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
        # cv2.putText(image, text, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# save the faces to disk
for (i, face) in enumerate(faces):
    face = cv2.resize(face, (128, 128))
    cv2.imwrite("face_{}.jpg".format(i + 1), face)

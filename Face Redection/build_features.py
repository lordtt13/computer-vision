# import the necessary packages
from keras.applications import ResNet50
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# construct the argument parser to parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required = True, help = "path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required = True, help = "path to Caffe pretrained model")
ap.add_argument("-f", "--features", required = True, help = "path to store extracted features")
ap.add_argument("-c", "--confidence", type = float, default = 0.5, help = "minimum probability to filter weak detections")
args = vars(ap.parse_args())

def extract_faces():
    # load the serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    # initialize the video stream
    print("[INFO] initializing video stream...")
    vs = VideoStream(src = 0).start()
    time.sleep(2.0)

    # loop over the frames from the video stream
    faces = []
    while True:

        fps = FPS().start()
        # capture the frame and resize it to have a min size of 400
        frame = vs.read()
        frame = imutils.resize(frame, width = 400)

        # grab the frame dimensions and convert it into a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence associated with each prediction
            confidence = detections[0, 0, i, 2]

            # filter out the weak detections
            if confidence > args["confidence"]:
                # compute the bounding box coordinates of the face object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (start_x, start_y, end_x, end_y) = box.astype("int")

                # add the extracted face
                face = frame[start_y - 5: end_y + 5, start_x - 5: end_x + 5]
                faces.append(face)
                #
                # # draw the bounding box on the image along with the confidence probability
                # text = "{:.2f}%".format(confidence * 100)
                # y = start_y - 10 if start_y - 10 > 10 else start_y + 10
                # cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
                # cv2.putText(frame, text, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        fps.update()
        # show the frame
        cv2.imshow("Output", frame)
        key = cv2.waitKey(1) & 0xFF

        # if 'q' was pressed, break from loop
        if key == ord('q'):
            break

    fps.stop()
    print("[INFO] Average FPS = {:.2f}".format(fps.fps()))

    # destroy all created windows
    cv2.destroyAllWindows()
    vs.stop()

    # reshape the faces to (128, 128) and convert from BGR to RGB
    for (i, face) in enumerate(faces):
        faces[i] = cv2.cvtColor(cv2.resize(face, (128, 128)), cv2.COLOR_BGR2RGB)

    faces = np.array(faces)

    # save the extracted faces to disk
    FACES_PATH = os.path.sep.join([args["features"], "extracted_faces.npy"])
    np.save(FACES_PATH, faces)

    return faces

def extract_features():
    # extract faces from video stream (press 'q' to stop the video stream)
    faces = extract_faces()

    # initialize the pretrained ResNet model
    print("[INFO] initializing ResNet model")
    model = ResNet50(include_top = False, weights = "imagenet", input_shape = (128, 128, 3))

    # extract the facial features
    print("[INFO] extracting features")
    features = model.predict(faces)

    # save the extracted features to disk
    FEATURES_PATH = os.path.sep.join([args["features"], "extracted_features.npy"])
    np.save(FEATURES_PATH, features)


if __name__ == "__main__":
    extract_features()

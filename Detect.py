# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2 as cv
import os
import time

result = []

ethnicities = ["Asian", "Black", "Hispanic", "Indian", "White"]
ages = ["Child", "Teen", "Young_Adult", "Adult", "Elderly"]
genders = ["Female", "Male"]
for ethnicity in ethnicities:
    for age in ages:
        for gender in genders:
            result.append(ethnicity + "-" + age + "-" + gender)

lowConfidence = 0.5

# face detectinon function


def detectAndPredict(frame, faceNet, predictNet):

    (h, w) = frame.shape[:2]
    blob = cv.dnn.blobFromImage(frame, 1.0, (224, 224),
                                (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > lowConfidence:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel ordering, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv.cvtColor(face, cv.COLOR_BGR2RGB)
            face = cv.resize(face, (48, 48))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = predictNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    return (locs, preds)


# load our serialized face detector model from disk
prototxtPath = r"deploy.prototxt"
weightsPath = r"res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
predictNet = load_model("detector.model")

# initialize the video stream
vs = VideoStream(src=0).start()

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it to have a maximum width of 900 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=900)

    # detect faces in the frame and determine if they are wearing a face mask or not
    (locs, preds) = detectAndPredict(frame, faceNet, predictNet)

    # loop over the detected face locations and their corresponding locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        #(mask, withoutMask) = pred
        print(np.argmax(pred))
		
        print(result[np.argmax(pred)])
        label = pred

        color = (0, 255, 0)

        # include the probability in the label
        #label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output frame

        cv.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # show the output frame
    cv.imshow("Age, Ethnicity, Gender Detection -- q to quit", frame)
    key = cv.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv.destroyAllWindows()
vs.stop()

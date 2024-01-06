import numpy as np
import time
import cv2
import os
#import imgtools

filepath = "yolo-coco"
dataNamesFile = "coco.names"
weightsFile = "yolov3.weights"
cfgFile = "yolov3.cfg"
'''
filepath = "yolo-9k"
dataNamesFile = "9k.names"
weightsFile = "yolo9000.weights"
cfgFile = "yolo9000.cfg"
'''
confidenceVal = 0.3
threshold = 0.3  # non-maxima suppression threshold

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([filepath, dataNamesFile])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([filepath, weightsFile])
configPath = os.path.sep.join([filepath, cfgFile])

# load our YOLO object detector trained on COCO dataset (80 classes)
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
print("[INFO] loaded Yolo")

def processScene(image, playRate, showImage):  # showImage is used to show / hide processed image
    # load our input image and grab its spatial dimensions
    # image = cv2.imread(imgFile)
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (128, 128),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # show timing information on YOLO
    #print("[INFO] Yolo processed scene in {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)

            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confidenceVal:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidenceVal, threshold)

    # i = 0
    # for i in classIDs:
    #	print('classID = {}'.format(LABELS[i]))
    listarr = []

    if showImage == 1:

        #image = imgtools.getLinesImageSimple(image)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                listarr.append(
                    {"class": LABELS[classIDs[i]], "confidence": round(confidences[i], 3), "x": x, "y": y, "w": w,
                     "h": h})

                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in COLORS[classIDs[i]]]
                image = np.ascontiguousarray(image, dtype=np.uint8)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)

        # show the output image


        cv2.imshow("Object Detection", image)
        cv2.waitKey(playRate)
        #cv2.waitKey(1)
    else:  # showImage is 0
        if len(idxs) > 0:
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                listarr.append(
                    {"class": LABELS[classIDs[i]], "confidence": round(confidences[i], 3), "x": x, "y": y, "w": w,
                     "h": h})

    return H, W, listarr

# def processScene: ends

# getObjectsInScene(imgFile1)

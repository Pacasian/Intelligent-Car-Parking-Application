import numpy as np
import argparse
import time
import cv2
import os
count =0
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")

args = vars(ap.parse_args())

labelsPath = os.path.sep.join(["yolo-coco","coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")


# checking the number plate of the car
def numPlate():

    return



np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")


weightsPath = os.path.sep.join(["yolo-coco","yolov3.weights"])
configPath = os.path.sep.join(["yolo-coco","yolov3.cfg"])



net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


image = cv2.imread(args["image"])
(H, W) = image.shape[:2]

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                             swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()


boxes = []
confidences = []
classIDs = []

for output in layerOutputs:

    for detection in output:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]

        if confidence > 0.5:
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))

            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

if len(idxs) > 0:
    for i in idxs.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        color = [int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
       # print(text.split(":")[0])
        if (text.split(":")[0]) == "car":
            crop_img = image[y:y + h, x:x + w]
            print("car")
            # frame=numPlate(crop_img)
        elif (text.split(":")[0]) == "truck":
            count=count+1
            print("truck")
            print("stop")
        elif (text.split(":")[0]) == "stop sign":
            print("stop sign")
        else:
            pass
        #print(count)
        cv2.putText(image, text.split(":")[0], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

cv2.imshow("Image", image)

cv2.waitKey(0)

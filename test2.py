import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier("Model2/keras_model.h5", "Model2/labels.txt")

offset = 20
imgSize = 300

folder = "Data/C"
counter = 0

labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure that the cropped image coordinates do not exceed the dimensions of the input image
        y_min, y_max = max(y - offset, 0), min(y + h + offset, img.shape[0])
        x_min, x_max = max(x - offset, 0), min(x + w + offset, img.shape[1])
        imgCrop = img[y_min:y_max, x_min:x_max]

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = cv2.resize(imgCrop, (imgSize, imgSize))

        prediction, index = classifier.getPrediction(imgCrop, draw=False)
        print(prediction, index)

        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)

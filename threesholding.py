import cv2 as cv
import numpy as np
img=cv.imread('Photo/img2.jpg')
img=cv.resize(img, (int(img.shape[1]*0.1), int(img.shape[0]*0.1)))
cv.imshow('Original', img)

gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)

haar_cascade=cv.CascadeClassifier('haar_face.xml')
faces_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=3)

for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y),(x+w, y+h), 0,1)

cv.imshow("Face Detect", img)
cv.waitKey(0)
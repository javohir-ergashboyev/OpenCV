import cv2 as cv
import os
import numpy as np

people=[]
# create people list manually
for i in os.listdir(r'C:\Users\Javohir\Desktop\train'):
    people.append(i)
main_path=r'C:\Users\Javohir\Desktop\train'

haar_cascade=cv.CascadeClassifier('haar_face.xml')

face_recogniser=cv.face.LBPHFaceRecognizer.create()
face_recogniser.read('face_trained.yml')

img=cv.imread(r'C:\Users\Javohir\Desktop\train\khabib\211.jpg')
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)

face_rect=haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

for (x,y,w,h) in face_rect:
    face_roi=gray[x:x+w,y:y+h]
    label ,confidence=face_recogniser.predict(face_roi)
    cv.putText(img, str(people[label]), (x,y), cv.FONT_HERSHEY_COMPLEX, 1.0, 255,1)
    cv.rectangle(img, (x,y), (x+w, y+h), 255,1)
cv.imshow('Detected', img)
cv.waitKey(0)
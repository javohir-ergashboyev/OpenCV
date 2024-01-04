import os
import numpy as np
import cv2 as cv

people=[]
# create people list manually
for i in os.listdir(r'C:\Users\Javohir\Desktop\train'):
    people.append(i)
main_path=r'C:\Users\Javohir\Desktop\train'

haar_cascade=cv.CascadeClassifier('haar_face.xml')

features=[]
labels=[]
def create_train():
    for person in people:
        path=os.path.join(main_path, person)
        label=people.index(person)

        for img in os.listdir(path):
            img_path=os.path.join(path,img)

            img_array=cv.imread(img_path)
            img_array=cv.resize(img_array, (int(img_array.shape[1]*0.5), int(img_array.shape[0]*0.5)))
           
            gray=cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            face_rect=haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in face_rect:
                face_roi=gray[x:x+w, y:y+h]
                features.append(face_roi)
                labels.append(label)

create_train()

features=np.array(features, dtype='object')
labels=np.array(labels)
np.save('features.npy',features)
np.save('labels.npy',labels)
face_recogniser=cv.face.LBPHFaceRecognizer.create()
face_recogniser.train(features,labels)
face_recogniser.save('face_trained.yml')

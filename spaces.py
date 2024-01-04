import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img=cv.imread('Photo\img2.jpg')
img=cv.resize(img,(int(img.shape[1]*0.1),int(img.shape[0]*0.1)))

gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
hist=cv.calcHist([gray],[0],None,[256],[0,256])
cv.imshow('Gray',gray)
plt.figure()
plt.xlabel('Bins')
plt.ylabel('Pixels')
plt.title('Gray Histogram')
plt.plot(hist)
plt.show()

cv.imshow('original',img)
cv.waitKey(0)

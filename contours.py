import cv2 as cv
import numpy as np
# original
image=cv.imread('Photo/img1.jpg')
image=cv.resize(image,(int(image.shape[1]*0.1),int(image.shape[0]*0.1)))
# cv.imshow("original", image)
# blank
blank=np.zeros(image.shape)
# cv.imshow('blank',blank)
# gray
gray=cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# cv.imshow('gray', gray)
# conrours with canny
# canny=cv.Canny(image, 125,175)
# cv.imshow("canny", canny)
# contours with thresh
ret, thresh=cv.threshold(gray,125,255,cv.THRESH_BINARY)
cv.imshow('thresh',thresh)


contours, hyrercy=cv.findContours(thresh,cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
cv.drawContours(blank,contours,-1,(255,0,0),1)
cv.imshow("Contours", blank)
cv.waitKey(0)

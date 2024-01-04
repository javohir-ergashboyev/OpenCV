import cv2 as cv
import numpy as np

capture = cv.VideoCapture(0)


def resize(frame, scale=0.75):
    wth = int(frame.shape[1] * scale)
    ht = int(frame.shape[0] * scale)
    dims = (wth, ht)
    return cv.resize(frame, dims, interpolation=cv.INTER_AREA)


img = cv.imread('Photo/img1.jpg')
img = resize(img, 0.1)


# cv.imshow('Nae', img)
# # greys
# img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow("Gray img", img2)
# # blured img
# img3 = cv.GaussianBlur(img, (7, 7), cv.BORDER_DEFAULT)
# cv.imshow('Blured', img3)
# # crop imgg
# img4 = img[100:100, 50:50]
# cv.imshow('Cropped', img4)
#
# cv.waitKey(0)


# blank = np.zeros((500, 500, 3), dtype='uint8')
# cv.rectangle(blank, (0, 0), (250, 250), (0, 255, 0), thickness=-1)
# cv.circle(blank, (250, 250), 50, (0, 0, 255), thickness=3)
# cv.line(blank, (0, 0), (250, 250), (255, 255, 255), thickness=5)
# cv.putText(blank, 'This is text', (50, 300), cv.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), thickness=2)
# cv.imshow('blank', blank)
# cv.waitKey(0)
#pip 
#
# while True:
#     _, frame = capture.read()
#     frame2=resize(frame)
#     cv.imshow('Video', frame2)
#     if cv.waitKey(20) and 0xFF == ord('q'):
#         break
#
# capture.release()
# cv.destroyAllWindows()

# Image Transmission
# translate
def translate(img, x, y):
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    dimension = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimension)


cv.imshow('translated', translate(img, 100, -100))


# rotate
def rotate(img, angle, rotPoint=None):
    (wth, hgt) = img.shape[:2]
    if rotPoint is None:
        rotPoint = (wth // 2, hgt // 2)
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    return cv.warpAffine(img, rotMat, (wth, hgt))


cv.imshow('Rotated', rotate(img, 45))
cv.waitKey(0)

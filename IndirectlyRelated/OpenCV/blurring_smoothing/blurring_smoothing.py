import cv2
import numpy as np

image = cv2.imread('filtering/man.jpg')

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_red = np.array([30,150,50])
upper_red = np.array([255,255,180])

mask = cv2.inRange(hsv, lower_red, upper_red)
res = cv2.bitwise_and(image,image, mask = mask)

kernel = np.ones((15, 15), np.float32) / 225
smoothed = cv2.filter2D(res, -1, kernel)

blur = cv2.GaussianBlur(res, (5, 5), 0)
median = cv2.medianBlur(res, 5)
bilateral = cv2.bilateralFilter(res, 15, 75, 75)

cv2.imshow('image',image)
cv2.imshow('mask',mask)
cv2.imshow('res',res)
cv2.imshow('smoothed',smoothed)
cv2.imshow('median',median)
cv2.imshow('bilateral',bilateral)


cv2.waitKey()
cv2.destroyAllWindows()

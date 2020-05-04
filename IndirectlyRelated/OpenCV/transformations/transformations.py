import cv2
import numpy as np

image = cv2.imread('filtering/man.jpg')

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_red = np.array([30,150,50])
upper_red = np.array([255,255,180])

mask = cv2.inRange(hsv, lower_red, upper_red)
res = cv2.bitwise_and(image,image, mask = mask)

kernel = np.ones((5,5),np.uint8)

opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

cv2.imshow('Original',image)
cv2.imshow('Mask',mask)
cv2.imshow('res',res)
cv2.imshow('Opening',opening)
cv2.imshow('Closing',closing)

cv2.waitKey()
cv2.destroyAllWindows()

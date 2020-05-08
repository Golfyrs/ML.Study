import numpy as np
import cv2

image = cv2.imread('corner_detection/sample.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

corners = cv2.goodFeaturesToTrack(gray, 0, 0.15, 0)
corners = np.int0(corners)

for corner in corners:
    x,y = corner.ravel()
    cv2.circle(image, (x,y), 3, 255, -1)
    
cv2.imshow('corner', image)
cv2.waitKey()
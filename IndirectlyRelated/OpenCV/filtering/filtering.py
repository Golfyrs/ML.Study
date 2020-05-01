import cv2
import numpy as np

image = cv2.imread('filtering/man.jpg')

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_red = np.array([30,150,50])
upper_red = np.array([255,255,180])

mask = cv2.inRange(hsv, lower_red, upper_red)
res = cv2.bitwise_and(image,image, mask= mask)

cv2.imshow('image',image)
cv2.imshow('mask',mask)
cv2.imshow('res',res)



# lower mask (0-10)
lower_red = np.array([0,50,50])
upper_red = np.array([10,255,255])
mask0 = cv2.inRange(hsv, lower_red, upper_red)

# upper mask (170-180)
lower_red = np.array([170,50,50])
upper_red = np.array([180,255,255])
mask1 = cv2.inRange(hsv, lower_red, upper_red)

# join my masks
mask = mask0+mask1

# set my output img to zero everywhere except my mask
output = image.copy()
output[np.where(mask==0)] = 0

cv2.imshow('output', output)

cv2.waitKey()
cv2.destroyAllWindows()

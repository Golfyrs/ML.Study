import numpy as np
import cv2

image_1 = cv2.imread('arithmetics/matplotlib.png')
image_2 = cv2.imread('arithmetics/mainsvmimage.png')

# addition = image_1 + image_2
# addition = cv2.add( image_1, image_2 )
weighted = cv2.addWeighted(image_1, 0.6, image_2, 0.4, 0)
cv2.imshow('weighted', weighted)



height, width, channels = image_1.shape
region_of_image = image_1[0:height, 0:width]

img2gray = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 220, 255, cv2.THRESH_BINARY_INV)

mask_inv = cv2.bitwise_not(mask)
background = cv2.bitwise_and(region_of_image, region_of_image, mask=mask_inv)
foreground = cv2.bitwise_and(image_2, image_2, mask=mask)

dst = cv2.add(background, foreground)
image_1[0:height, 0:width] = dst

cv2.imshow('mask', mask)
cv2.imshow('mask_inv', mask_inv)
cv2.imshow('background', background)
cv2.imshow('foreground', foreground)
cv2.imshow('dst', dst)


cv2.waitKey(0)
cv2.destroyAllWindows()
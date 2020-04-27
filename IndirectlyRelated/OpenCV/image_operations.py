import numpy as np
import cv2

image = cv2.imread('watch.jpg', cv2.IMREAD_COLOR)

px = image[55, 55]

print(px)

width = image.shape[1]
height = image.shape[0]
print(height)

image[100, :] = [255, 255, 255]

region_of_image = image[ 500:700, 500:700 ]
image[0:200, 0:200] = region_of_image

cv2.imshow('Image', image)
cv2.imshow('roi', region_of_image)
cv2.waitKey(0)
cv2.destroyAllWindows() 
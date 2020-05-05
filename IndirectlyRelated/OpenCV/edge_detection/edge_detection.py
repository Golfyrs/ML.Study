import cv2
import numpy as np

image = cv2.imread('filtering/man.jpg')

lapplacian = cv2.Laplacian(image, cv2.CV_64F)

sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
edges = cv2.Canny(image, 200, 200)


cv2.imshow('original', image)
cv2.imshow('lapplacian', lapplacian)
cv2.imshow('sobelx', sobelx)
cv2.imshow('sobely', sobely)
cv2.imshow('edges', edges)

cv2.waitKey()
cv2.destroyAllWindows()
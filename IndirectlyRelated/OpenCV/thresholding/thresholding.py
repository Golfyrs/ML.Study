import cv2
import numpy as np

image = cv2.imread('thresholding/bookpage.jpg')
retval, theshold = cv2.threshold(image, 8, 255, cv2.THRESH_BINARY)

grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
retval, theshold2 = cv2.threshold(grayscaled, 8, 255, cv2.THRESH_BINARY)
gaus = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 155, 1)
retval, otsu = cv2.threshold(grayscaled, 125, 255, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)

cv2.imshow('original', image)
cv2.imshow('threshold', theshold)
cv2.imshow('theshold2', theshold2)
cv2.imshow('gaus', gaus)
cv2.imshow('otsu', otsu)
cv2.waitKey(0)
cv2.destroyAllWindows()
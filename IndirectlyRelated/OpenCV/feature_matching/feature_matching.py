import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('search/fishing2.jpg',0)
template = cv2.imread('search/float.jpg',0)

orb = cv2.ORB(1000, 1.2)

kp1, des1 = orb.detectAndCompute(image, None)
kp2, des2 = orb.detectAndCompute(template, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x : x.distance)

result = cv2.drawMatches(image, kp1, template, kp2, matches[:10], None, flags=2)
plt.imshow(result)
plt.show()

import numpy as np
import cv2

image = cv2.imread('watch.jpg', cv2.IMREAD_COLOR)

# NOTE! cv2 work with BGR
# Draw line in image. Start line at (0, 0) and end at (150, 150).
# Line color is (255, 100, 0) blue = 255, Green = 100, Red = 0 and line thickness is 15.
cv2.line(image, (0, 0), (150, 150), (255, 100, 0), 15)
cv2.rectangle(image, (15, 25), (200, 100), (0, 255, 0), -1)
cv2.circle(image, (100, 70), 55, (0, 0, 255), -1)

points = np.array( [5, 5, 20, 20, 40, 20, 20, 40], np.int32 )
points = points.reshape( (-1, 2) )
# Or you can avoid reshape:
# points = np.array( [[5, 5], [20, 20], [40, 20], [20, 40]], np.int32 )
cv2.polylines(image, [points], True, (255,182,193), 5)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image, 'Some text', (100, 300), font, 10, (255, 255, 255), 10, cv2.LINE_4)


cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows() 

import cv2
import numpy as np
import matplotlib.pyplot as plt

# IMREAD_UNCHANGED - -1
# IMREAD_GRAYSCALE = 0
# IMREAD_COLOR - 1
image = cv2.imread('watch.jpg', cv2.IMREAD_GRAYSCALE) # Filter IMREAD_GRAYSCALE that will be applied on loaded image.
image = cv2.resize(image, (600, 600))

# Open window and resize window with cv2.
# cv2.imshow('Image', image)
# cv2.resizeWindow('Image', 600,600)

# Leave the window open until text is entered.
# cv2.waitKey()
# cv2.destroyAllWindows()

# Save image to file.
cv2.imwrite('watchgray.png', image)

# Show image via matplotlib.
plt.imshow(image, cmap='gray', interpolation='bicubic')
plt.plot( [50, 100], [80, 100], 'b', linewidth=5)
plt.show()

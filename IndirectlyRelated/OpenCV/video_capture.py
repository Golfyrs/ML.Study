import cv2
import numpy as np

cap = cv2.VideoCapture('input.avi') # Load video

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480)) # Record video to file.

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)

    cv2.waitKey(1)
    if 0xFF == ord('q'):
        break

cap.release()
out.release()

cv2.destroyAllWindows()

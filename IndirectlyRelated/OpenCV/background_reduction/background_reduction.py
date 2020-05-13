import numpy as np
import cv2

cap = cv2.VideoCapture('background_reduction/people-walking.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()

while(1):
    ret, frame = cap.read()

    FG_mask = fgbg.apply(frame)
 
    cv2.imshow('fgmask', frame)
    cv2.imshow('frame', FG_mask)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    

cap.release()
cv2.destroyAllWindows()
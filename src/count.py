import numpy as np
import cv2

cap = cv2.VideoCapture('project_video.mp4')
count = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count%100 == 0:
        print("count={}".format(count))

cap.release()
print("total count={}".format(count))
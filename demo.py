#!/usr/bin/env python

import cv2
from core.detectors import CornerNet_Squeeze
from core.vis_utils import draw_bboxes
import time

cap = cv2.VideoCapture(0)
if cap is None:
    print("Camera is not ready ")
    exit()


detector = CornerNet_Squeeze()

while True:

    ret,frame = cap.read()
    if not ret:
        pass
    #image = cv2.imread("demo.jpg")

    t0 = time.time()

    #bboxes = detector(image)
    bboxes = detector(frame)
    image  = draw_bboxes(frame, bboxes)
    #cv2.imwrite("demo_out.jpg", image)
    fps = 1.0/(time.time()-t0)
    print("[FPS]: ",fps)

    cv2.imshow("wind",image)

    cv2.waitKey(1)








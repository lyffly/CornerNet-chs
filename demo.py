#!/usr/bin/env python

import cv2
from core.detectors import CornerNet_Squeeze
from core.vis_utils import draw_bboxes
import time

# 打开编号为0的相机
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera is not ready ")
    exit()

# 加载模型
detector = CornerNet_Squeeze()

while True:

    ret,frame = cap.read()
    if not ret:
        pass
    #image = cv2.imread("demo.jpg")

    t0 = time.time()

    #bboxes = detector(image)
    bboxes = detector(frame)
    # 画框
    image  = draw_bboxes(frame, bboxes)
    #cv2.imwrite("demo_out.jpg", image)
    
    # 计算 FPS
    fps = 1.0/(time.time()-t0)
    print("[FPS]: ",fps)

    cv2.imshow("cornernet",image)

    cv2.waitKey(1)








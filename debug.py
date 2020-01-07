# -*- coding: utf-8 -*-
import cv2
import numpy as np

"""
用于调试的一些函数
显示图片

"""


def img_show(filename):
    cv2.imshow("img_show", filename)
    cv2.waitKey(0)


def draw_box_contours(oldimg, box):  # 在原图像上画框
    box = np.int0(box)
    oldimg = cv2.drawContours(oldimg, [box], 0, (0, 0, 255), 2)
    return oldimg


# -*- coding: utf-8 -*-
import cv2
import numpy as np
from debug import *
import debug
import sys
import os
import json
from matplotlib import pyplot as plt
from PIL import Image
MAX_WIDTH = 1000  # 图片最大宽度
"""
文件的一些输入输出操作
"""


filename = "C:/study/计算机视觉/Python_VLPR-master/test/Yes_img/1_3.jpg"

def img_read(filename):
    return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
    # 以uint8方式读取filename标识的图片， 放入imdecode中，cv2.IMREAD_COLOR读取彩色照片



# -*- coding: utf-8 -*-
import cv2
import numpy as np
from numpy.linalg import norm
import sys
import os
import json
from matplotlib import pyplot as plt
from PIL import Image

SZ = 20  # 训练图片长宽
MAX_WIDTH = 1000  # 原始图片最大宽度
Min_Area = 2000  # 车牌区域允许最大面积
PROVINCE_START = 1000


class StatModel(object):
    def load(self, fn):
        self.model = self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)


class SVM(StatModel):
    def __init__(self, C=1, gamma=0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    # 训练svm
    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    # 字符识别
    def predict(self, samples):
        r = self.model.predict(samples)
        return r[1].ravel()  # 扁平化为一维数组，对原数组的引用（会改变原数组的值）


# 来自opencv的sample，用于svm训练
def deskew(img):  # 计算HOG前需要使用图片的二阶矩对其进行抗扭斜处理
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


# 来自opencv的sample，用于svm训练
def preprocess_hog(digits):  # 计算图像的HOG描述符
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)  # 笛卡尔坐标转换为极坐标
        bin_n = 16
        bin = np.int32(bin_n * ang / (2 * np.pi))  # 计算得到每个像素的梯度的方向和大小，把这个梯度转换成16位的整数
        bin_cells = bin[:10, :10], bin[10:, :10], bin[:10, 10:], bin[10:, 10:]  # 分成四个小方块
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        # 对每个小方块计算它们的朝向直方图，使用梯度的大小作为权重，得到一个含16个成员的向量
        hist = np.hstack(hists)
        #  4个小方块共64个成员的向量组成了特征向量
        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)


provinces = [
    "zh_cuan", "川",
    "zh_e", "鄂",
    "zh_gan", "赣",
    "zh_gan1", "甘",
    "zh_gui", "贵",
    "zh_gui1", "桂",
    "zh_hei", "黑",
    "zh_hu", "沪",
    "zh_ji", "冀",
    "zh_jin", "津",
    "zh_jing", "京",
    "zh_jl", "吉",
    "zh_liao", "辽",
    "zh_lu", "鲁",
    "zh_meng", "蒙",
    "zh_min", "闽",
    "zh_ning", "宁",
    "zh_qing", "青",
    "zh_qiong", "琼",
    "zh_shan", "陕",
    "zh_su", "苏",
    "zh_sx", "晋",
    "zh_wan", "皖",
    "zh_xiang", "湘",
    "zh_xin", "新",
    "zh_yu", "豫",
    "zh_yu1", "渝",
    "zh_yue", "粤",
    "zh_yun", "云",
    "zh_zang", "藏",
    "zh_zhe", "浙"
]

dic = {
    "川":"zh_cuan",
    "鄂":"zh_e",
    "赣":"zh_gan",
    "甘":"zh_gan1",
    "贵":"zh_gui",
    "桂":"zh_gui1",
    "黑":"zh_hei",
    "沪":"zh_hu",
    "冀":"zh_ji",
    "津":"zh_jin",
    "京":"zh_jing",
    "吉":"zh_jl",
    "辽":"zh_liao",
    "鲁":"zh_lu",
    "蒙":"zh_meng",
    "闽":"zh_min",
    "宁":"zh_ning",
    "青":"zh_qing",
    "琼":"zh_qiong",
    "陕":"zh_shan",
    "苏":"zh_su",
    "晋":"zh_sx",
    "皖":"zh_wan",
    "湘":"zh_xiang",
    "新":"zh_xin",
    "豫":"zh_yu",
    "渝":"zh_yu1",
    "粤":"zh_yue",
    "云":"zh_yun",
    "藏":"zh_zang",
    "浙":"zh_zhe"


}



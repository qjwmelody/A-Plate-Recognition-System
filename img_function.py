# -*- coding: utf-8 -*-
import cv2
import numpy as np
import sys
import os
import json
from matplotlib import pyplot as plt
from PIL import Image
from debug import *
from img_read import *
import svm_recg
import cnn_recg_character
import cnn_recg_digit
SZ = 20  # 训练图片长宽
MAX_WIDTH = 1000  # 原始图片最大宽度
Min_Area = 2000  # 车牌区域允许最小面积
color_Min_Area = 1000
PROVINCE_START = 1000

"""
定义了主要的类
车牌识别类
"""
def cv2_to_PIL(cv_img):
    img = Image.fromarray(cv_img)
    img = img.resize((32, 40))
    return img

def point_limit(point):  # 点坐标标准化
    if point[0] < 0:
        point[0] = 0
    if point[1] < 0:
        point[1] = 0


# 根据设定的阈值和图片直方图，找出波峰，用于分隔字符
def find_waves(threshold, histogram):
    up_point = -1  # 上升点
    is_peak = False
    if histogram[0] > threshold:
        up_point = 0
        is_peak = True
    wave_peaks = []
    for i, x in enumerate(histogram):
        if is_peak and x < threshold:
            if i - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point, i))
        elif not is_peak and x >= threshold:
            is_peak = True
            up_point = i
    if is_peak and up_point != -1 and i - up_point > 4:
        wave_peaks.append((up_point, i))
    return wave_peaks


def seperate_card(img, waves):  # 根据波峰分割灰度图，每个峰被认为是一个字符
    part_cards = []
    for wave in waves:
        part_cards.append(img[:, wave[0]:wave[1]])
    return part_cards


def img_contours(oldimg,box):
    box = np.int0(box)
    oldimg = cv2.drawContours(oldimg, [box], 0, (0, 0, 255), 2)
    if oldimg.size:
        cv2.imshow("img_contours", oldimg)
        cv2.waitKey(0)


class CardPredictor:
    def __init__(self):
        pass

    def __del__(self):
        self.save_traindata()

    def train_svm(self):
        # 识别英文字母和数字
        self.model = svm_recg.SVM(C=1, gamma=0.5)
        # 识别中文
        self.modelchinese = svm_recg.SVM(C=1, gamma=0.5)
        if os.path.exists("svm.dat"):
            self.model.load("svm.dat")
        else:
            chars_train = []
            chars_label = []

            for root, dirs, files in os.walk("train\\chars2"):
                if len(os.path.basename(root)) > 1:
                    continue
                root_int = ord(os.path.basename(root))
                for filename in files:
                    filepath = os.path.join(root, filename)
                    digit_img = cv2.imread(filepath)
                    digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                    chars_train.append(digit_img)
                    # chars_label.append(1)
                    chars_label.append(root_int)

            chars_train = list(map(svm_recg.deskew, chars_train))
            # map函数接收一个函数f和一个list， 把函数f依次作用在list的每个元素上，得到一个新的list并且返回
            chars_train = svm_recg.preprocess_hog(chars_train)
            # chars_train = chars_train.reshape(-1, 20, 20).astype(np.float32)
            chars_label = np.array(chars_label)
            print(chars_train.shape)
            self.model.train(chars_train, chars_label)
        if os.path.exists("svmchinese.dat"):
            self.modelchinese.load("svmchinese.dat")
        else:
            chars_train = []
            chars_label = []
            for root, dirs, files in os.walk("train\\charsChinese"):
                if not os.path.basename(root).startswith("zh_"):
                    continue
                pinyin = os.path.basename(root)
                index = svm_recg.provinces.index(pinyin) + PROVINCE_START + 1  # 1是拼音对应的汉字
                for filename in files:
                    filepath = os.path.join(root, filename)
                    digit_img = cv2.imread(filepath)
                    digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                    chars_train.append(digit_img)
                    # chars_label.append(1)
                    chars_label.append(index)
            chars_train = list(map(svm_recg.deskew, chars_train))
            chars_train = svm_recg.preprocess_hog(chars_train)
            # chars_train = chars_train.reshape(-1, 20, 20).astype(np.float32)
            chars_label = np.array(chars_label)
            print(chars_train.shape)
            self.modelchinese.train(chars_train, chars_label)

    def save_traindata(self):
        if not os.path.exists("svm.dat"):
            self.model.save("svm.dat")
        if not os.path.exists("svmchinese.dat"):
            self.modelchinese.save("svmchinese.dat")


    def img_first_pre(self, car_pic_file):
        # if type(car_pic_file) == type(""):
        #   img = img_read(car_pic_file)
        img = car_pic_file
        pic_height, pic_width = img.shape[:2]  # 获取图片高度和宽度
        if pic_width > MAX_WIDTH:
            resize_rate = MAX_WIDTH / pic_width
            img = cv2.resize(img, (MAX_WIDTH, int(pic_height * resize_rate)), interpolation=cv2.INTER_AREA)
        # 缩小图片
        blur = 5
        img = cv2.GaussianBlur(img, (blur, blur), 0)
        # 高斯模糊
        oldimg = img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 转化为灰度图像
        return img, oldimg

    def operator_comp(self, img, operator):
        if operator == "sobel":
            # sobel算子
            x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
            y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
            absX = cv2.convertScaleAbs(x)
            absY = cv2.convertScaleAbs(y)
            edge = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        elif operator == "canny":
            edge = cv2.Canny(img, 20, 150)
            # for debug
            cv2.namedWindow("Canny", 0)
            cv2.resizeWindow("Canny", 640, 480)
            cv2.imshow("Canny", edge)
            # canny算子
        elif operator == "laplacian":
            gray_lap = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
            edge = cv2.convertScaleAbs(gray_lap)
            # for debug
            cv2.namedWindow("Laplacian", 0)
            cv2.resizeWindow("Laplacian", 640, 480)
            cv2.imshow("Laplacian", edge)
            # laplacian算子
        else:
            edge = cv2.Canny(img, 100, 200)
            """
            # for debug
            cv2.namedWindow("Original", 0)
            cv2.resizeWindow("Original", 640, 480)
            cv2.imshow("Original", img)
            
            # for debug
            cv2.namedWindow("GaussianBlur", 0)
            cv2.resizeWindow("GaussianBlur", 640, 480)
            cv2.imshow("GaussianBlur", img)
            
            # for debug
            cv2.namedWindow("Gray", 0)
            cv2.resizeWindow("Gray", 640, 480)
            cv2.imshow("Gray", img)
            
            # for debug
            cv2.namedWindow("Sobel", 0)
            cv2.resizeWindow("Sobel", 640, 480)
            cv2.imshow("Sobel", edge)
            """
        ret, thresh = cv2.threshold(edge, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 二值化
        matrix = np.ones((4, 19), np.uint8)
        close_edge = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, matrix)
        open_edge = cv2.morphologyEx(close_edge, cv2.MORPH_OPEN, matrix)
        # 形态学操作

        return open_edge

    def img_findContours(self, pre_img, oldimg):
        img, contours, hierarchy = cv2.findContours(pre_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > Min_Area]  # 排除面积最小的点
        # print("findContours len = ", len(contours))
        # 排除面积最小的点
        car_contours = []
        background = oldimg.copy()
        for cnt in contours:
            rec = cv2.minAreaRect(cnt)  # rec[0]为中心坐标
            width, height = rec[1]  # rec[1]为矩形的长宽
            if width < height:
                width, height = height, width
            ration = width / height

            if 2 < ration < 5.5:
                car_contours.append(rec)
                box = cv2.boxPoints(rec)  # 获取最小外接矩形的四个顶点
                background = draw_box_contours(background, box)
        # 显示已经框好的图像

        if background.size:
            # for debug
            cv2.namedWindow("card_contours", 0)
            cv2.resizeWindow("card_contours", 480, 360)
            cv2.imshow("card_contours", background)
            cv2.waitKey(0)

        return car_contours

    def img_find_colorContours(self, pre_img, oldimg):
        img, contours, hierarchy = cv2.findContours(pre_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > color_Min_Area]  # 排除面积最小的点
        # print("findContours len = ", len(contours))
        # 排除面积最小的点
        car_contours = []
        background = oldimg.copy()
        for cnt in contours:
            rec = cv2.minAreaRect(cnt)  # rec[0]为中心坐标
            width, height = rec[1]  # rec[1]为矩形的长宽
            if width < height:
                width, height = height, width
            ration = width / height

            if 2 < ration < 5.5:
                car_contours.append(rec)
                box = cv2.boxPoints(rec)  # 获取最小外接矩形的四个顶点
                background = draw_box_contours(background, box)
        # 显示已经框好的图像

        if background.size:
            # for debug
            cv2.namedWindow("card_color_contours", 0)
            cv2.resizeWindow("card_color_contours", 480, 360)
            cv2.imshow("card_color_contours", background)
            cv2.waitKey(0)

        return car_contours

    def img_Transform(self, car_contours, oldimg, pic_width, pic_height):  # 矩形矫正
        car_imgs = []
        for car_rect in car_contours:  # rec[2]为旋转角度
            #  旋转角度θ是水平轴（x轴）逆时针旋转，与碰到的矩形的第一条边的夹角。
            #  并且这个边的边长是width，另一条边边长是height
            #  在opencv中，坐标系原点在左上角，相对于x轴，逆时针旋转角度为负，顺时针旋转角度为正。
            #  在这里，θ∈（-90度，0]。
            if -1 < car_rect[2] < 1:
                angle = 1
                # 对于角度为-1 1之间时，默认为1
            else:
                angle = car_rect[2]
            car_rect = (car_rect[0], (int(car_rect[1][0]+5), int(car_rect[1][1]+5)), angle)  # 长宽各加5，可修改
            box = cv2.boxPoints(car_rect)

            height_point = right_point = [0, 0]  # 确定上下左右边界，循环比较
            left_point = low_point = [pic_width, pic_height]
            for point in box:
                if left_point[0] > point[0]:
                    left_point = point
                if low_point[1] > point[1]:
                    low_point = point
                if height_point[1] < point[1]:
                    height_point = point
                if right_point[0] < point[0]:
                    right_point = point

            if left_point[1] <= right_point[1]:  # 正角度，即从x轴正方向逆时针旋转
                new_right_point = [right_point[0], height_point[1]]  # 拉平下底边
                pts2 = np.float32([left_point, height_point, new_right_point])  # 字符只是高度需要改变
                pts1 = np.float32([left_point, height_point, right_point])
                M = cv2.getAffineTransform(pts1, pts2)  # 三个点确定一个仿射变换， 求得变换矩阵
                dst = cv2.warpAffine(oldimg, M, (pic_width, pic_height))  # 对原图像整体进行仿射变换
                point_limit(new_right_point)
                point_limit(height_point)
                point_limit(left_point)
                car_img = dst[int(left_point[1]):int(height_point[1]), int(left_point[0]):int(new_right_point[0])]
                #  cv2.imshow("", car_img)  # for debug
                #  cv2.waitKey(0)
                car_imgs.append(car_img)

            elif left_point[1] > right_point[1]:  # 负角度
                new_left_point = [left_point[0], height_point[1]]
                pts2 = np.float32([new_left_point, height_point, right_point])  # 字符只是高度需要改变
                pts1 = np.float32([left_point, height_point, right_point])
                M = cv2.getAffineTransform(pts1, pts2)
                dst = cv2.warpAffine(oldimg, M, (pic_width, pic_height))
                point_limit(right_point)
                point_limit(height_point)
                point_limit(new_left_point)
                car_img = dst[int(right_point[1]):int(height_point[1]), int(new_left_point[0]):int(right_point[0])]
                #  cv2.imshow("", car_img)  # for debug
                #  cv2.waitKey(0)
                car_imgs.append(car_img)

            # for debug
        """
        for new_rec in car_imgs:
            if new_rec.size:
                cv2.imshow("rectangle transform", new_rec)
                cv2.waitKey(0)
        """

        return car_imgs

    def accurate_place(self, card_img_hsv, limit1, limit2, color):
        row_num, col_num = card_img_hsv.shape[:2]
        xl = col_num
        xr = 0
        yh = 0
        yl = row_num
        row_num_limit = 21
        col_num_limit = col_num * 0.8 if color != "green" else col_num * 0.5  # 绿色有渐变
        for i in range(row_num):
            count = 0
            for j in range(col_num):
                H = card_img_hsv.item(i, j, 0)
                S = card_img_hsv.item(i, j, 1)
                V = card_img_hsv.item(i, j, 2)
                if limit1 < H <= limit2 and 34 < S and 46 < V:
                    count += 1
            if count > col_num_limit:
                if yl > i:
                    yl = i
                if yh < i:
                    yh = i
        for j in range(col_num):
            count = 0
            for i in range(row_num):
                H = card_img_hsv.item(i, j, 0)
                S = card_img_hsv.item(i, j, 1)
                V = card_img_hsv.item(i, j, 2)
                if limit1 < H <= limit2 and 34 < S and 46 < V:
                    count += 1
            if count > row_num - row_num_limit:
                if xl > j:
                    xl = j
                if xr < j:
                    xr = j
        return xl, xr, yh, yl
    
    def img_color(self, card_imgs, only_color):
        colors = []
        for card_index, card_img in enumerate(card_imgs):
            green = yellow = blue = black = white = 0
            if not card_img.size:
                continue
            card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
            # 有转换失败的可能，原因来自于上面矫正矩形出错
            if card_img_hsv is None:
                continue
            row_num, col_num = card_img_hsv.shape[:2]
            card_img_count = row_num * col_num  # 总的像素值

            for i in range(row_num):  # 思想是：对于每一个像素点，考察其HSV分量，对其进行分类；一幅图像中统计出来最多的颜色被视为该图像的颜色，待研究
                for j in range(col_num):
                    H = card_img_hsv.item(i, j, 0)
                    S = card_img_hsv.item(i, j, 1)
                    V = card_img_hsv.item(i, j, 2)
                    if 11 < H <= 34 and S > 34:
                        yellow += 1
                    elif 35 < H <= 99 and S > 34:
                        green += 1
                    elif 99 < H <= 124 and S > 34:
                        blue += 1

                    if 0 < H < 180 and 0 < S < 255 and 0 < V < 46:
                        black += 1
                    elif 0 < H < 180 and 0 < S < 43 and 221 < V < 225:
                        white += 1
            color = "no"

            limit1 = limit2 = 0
            if yellow * 2 >= card_img_count:
                color = "yellow"
                limit1 = 11
                limit2 = 34  # 有的图片有色偏绿
            elif green * 2 >= card_img_count:
                color = "green"
                limit1 = 35
                limit2 = 99
            elif blue * 2 >= card_img_count:
                color = "blue"
                limit1 = 100
                limit2 = 124  # 有的图片有色偏紫
            elif black + white >= card_img_count * 0.7:
                color = "bw"
            colors.append(color)
            card_imgs[card_index] = card_img
            if only_color:
                continue
            if limit1 == 0:
                continue
            xl, xr, yh, yl = self.accurate_place(card_img_hsv, limit1, limit2, color)
            if yl == yh and xl == xr:
                continue
            need_accurate = False
            if yl >= yh:
                yl = 0
                yh = row_num
                need_accurate = True
            if xl >= xr:
                xl = 0
                xr = col_num
                need_accurate = True

            if color == "green":
                card_imgs[card_index] = card_img
            else:
                card_imgs[card_index] = card_img[yl:yh, xl:xr] if color != "green" or yl < (yh - yl) // 4 else card_img[yl - (yh - yl) // 4:yh, xl:xr]

            if need_accurate:
                card_img = card_imgs[card_index]
                card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
                xl, xr, yh, yl = self.accurate_place(card_img_hsv, limit1, limit2, color)
                if yl == yh and xl == xr:
                    continue
                if yl >= yh:
                    yl = 0
                    yh = row_num
                if xl >= xr:
                    xl = 0
                    xr = col_num
            if color == "green":
                card_imgs[card_index] = card_img
            else:
                card_imgs[card_index] = card_img[yl:yh, xl:xr] if color != "green" or yl < (yh - yl) // 4 else card_img[yl - ( yh - yl) // 4:yh, xl:xr]

        return colors, card_imgs
    
    def img_color_contours(self, pre_img, oldimg):
        """
        :param pre_img: 预处理好的图像
        :param oldimg: 原图像
        :return: 已经定位好的车牌
        """
        pic_height, pic_width = pre_img.shape[:2]
        self.train_svm()
        card_contours = self.img_findContours(pre_img, oldimg)
        card_imgs = self.img_Transform(card_contours, oldimg, pic_width, pic_height)
        colors, car_imgs = self.img_color(card_imgs, 0)
        predict_result = []
        roi = None
        card_color = None
        standard_part = []

        for i, color in enumerate(colors):
            if color in ("blue", "yellow", "green"):
                card_img = card_imgs[i]
                if not card_img.size:
                    continue
                gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
                # 黄、绿车牌字符比背景暗、与蓝车牌刚好相反，所以黄、绿车牌需要反向
                if color == "green" or color == "yellow":
                    gray_img = cv2.bitwise_not(gray_img)
                ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # for debug
                """
                if gray_img.size:
                    cv2.imshow("", gray_img)
                    cv2.waitKey(0)
                """

                x_histogram = np.sum(gray_img, axis=1)  # axis=1表示x方向,按照行向量内部进行相加 axis=0表示y方向
                x_min = np.min(x_histogram)
                x_average = np.sum(x_histogram) / x_histogram.shape[0]
                x_threshold = (x_min + x_average) / 2
                # for debug
                arr = x_histogram.flatten()
                #fig = plt.figure()
                #plt.title("x_histogram")
                #n, bins, patches = plt.hist(arr, bins=256, density=1, facecolor='green', alpha=0.75)
                #plt.show()

                wave_peaks = find_waves(x_threshold, x_histogram)
                # for debug
                #fig = plt.figure()
                #plt.title("x_wave_peaks")
                #n, bins, patches = plt.hist(wave_peaks, bins=256, density=1, facecolor='green', alpha=0.75)
                #plt.show()

                if len(wave_peaks) == 0:
                    # print("peak less 0:")
                    continue
                # 认为水平方向，最大的波峰为车牌区域
                wave = max(wave_peaks, key=lambda x: x[1] - x[0])
                gray_img = gray_img[wave[0]:wave[1]]
                # 查找垂直直方图波峰
                row_num, col_num = gray_img.shape[:2]
                # 去掉车牌上下边缘1个像素，避免白边影响阈值判断
                gray_img = gray_img[1:row_num - 1]
                y_histogram = np.sum(gray_img, axis=0)  # 得到y方向的直方图
                y_min = np.min(y_histogram)
                y_average = np.sum(y_histogram) / y_histogram.shape[0]
                y_threshold = (y_min + y_average) / 5  # U和0要求阈值偏小，否则U和0会被分成两半
                wave_peaks = find_waves(y_threshold, y_histogram)

                # for debug
                #fig = plt.figure()
                #plt.title("y_histogram")
                #n, bins, patches = plt.hist(y_histogram, bins=256, density=1, facecolor='green', alpha=0.75)
                #plt.show()

                # for debug
                #fig = plt.figure()
                #plt.title("y_wave_peaks")
                #n, bins, patches = plt.hist(wave_peaks, bins=256, density=1, facecolor='green', alpha=0.75)
                #plt.show()

                if len(wave_peaks) <= 6:
                    # print("peak less 1:", len(wave_peaks))
                    continue

                wave = max(wave_peaks, key=lambda x: x[1] - x[0])  # x[1]-x[0]表示一个波峰的大小
                max_wave_dis = wave[1] - wave[0]
                # 判断是否是左侧车牌边缘
                if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis / 3 and wave_peaks[0][0] == 0:  # 左侧边缘，较小波峰
                    wave_peaks.pop(0)  # 从列表中移除该项

                # 组合分离汉字
                cur_dis = 0
                for i, wave in enumerate(wave_peaks):
                    if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
                        break
                    else:
                        cur_dis += wave[1] - wave[0]
                if i > 0:
                    wave = (wave_peaks[0][0], wave_peaks[i][1])
                    wave_peaks = wave_peaks[i + 1:]
                    wave_peaks.insert(0, wave)
                point = wave_peaks[2]
                point_img = gray_img[:, point[0]:point[1]]
                if np.mean(point_img) < 255 / 5:
                    wave_peaks.pop(2)

                if len(wave_peaks) <= 6:
                    # print("peak less 2:", len(wave_peaks))
                    continue

                # for debug

                arr = np.array(wave_peaks)
                fig = plt.figure()
                plt.title("final_wave_peaks")
                n, bins, patches = plt.hist(arr, bins=256, density=1, facecolor='green', alpha=0.75)
                plt.show()

                part_cards = seperate_card(gray_img, wave_peaks)
                #  for debug

                fig = plt.figure()
                plt.title("seperate card")
                plt.axis('off')
                i = 0
                for part in part_cards:
                    i = i+1
                    fig.add_subplot(1, len(part_cards), i)
                    plt.axis('off')
                    plt.imshow(part, cmap="gray")
                plt.show()
                fig = plt.figure()
                plt.title("standard card parts")
                plt.axis('off')

                for i, part_card in enumerate(part_cards):
                    # 可能是固定车牌的铆钉
                    if np.mean(part_card) < 255 / 5:
                        # print("a point")
                        continue
                    part_card_old = part_card

                    w = abs(part_card.shape[1] - SZ) // 2  # 缩放至标准大小

                    part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])  # 填充左右边界
                    part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)  # 缩放至标准大小
                    # for debug
                    """
                    fig.add_subplot(1, len(part_cards), i + 1)
                    plt.axis('off')
                    plt.imshow(part_card, cmap="gray")
                    """
                    standard_part.append(part_card)

                    part_card = svm_recg.preprocess_hog([part_card])
                    if i == 0:
                        resp = self.modelchinese.predict(part_card)  # 获取最大可能的预测
                        character = svm_recg.provinces[int(resp[0]) - PROVINCE_START]
                    else:
                        resp = self.model.predict(part_card)
                        character = chr(resp[0])

                    # Now apply cnn
                    """
                    if i == 0:
                        part_card = cv2_to_PIL(part_card)
                        character = cnn_recg_character.cnn_predict(part_card)
                    else:
                        part_card = cv2_to_PIL(part_card)
                        character = cnn_recg_digit.cnn_predict(part_card)
                    """
                    # 防止川被分割成川1
                    if i == 1 and character == "1":
                        standard_part = standard_part[:-1]
                        continue
                    # 判断最后一个数是否是车牌边缘，假设车牌边缘被认为是1
                    if character == "1" and i == len(part_cards) - 1:
                        if part_card_old.shape[0] / part_card_old.shape[1] >= 7 and len(predict_result) >= 7:  # 1太细，认为是边缘
                            standard_part = standard_part[:-1]
                            continue
                    predict_result.append(character)
                if len(predict_result) > 7:
                    # predict_result = predict_result[:-1]
                    standard_part = standard_part[:-1]

                # plt.show()
                roi = card_img
                card_color = color
                break

        return predict_result, roi, card_color, standard_part  # 识别到的字符、定位的车牌图像、车牌颜色


    def img_only_color(self, pre_img, oldimg):
        """
        :param pre_img: 预处理好的图像
        :param oldimg: 原图像文件
        :return: 已经定位好的车牌
        """
        pic_height, pic_width = pre_img.shape[:2]
        self.train_svm()
        lower_blue = np.array([100, 110, 110])
        upper_blue = np.array([130, 255, 255])
        lower_yellow = np.array([15, 55, 55])
        upper_yellow = np.array([50, 255, 255])
        lower_green = np.array([50, 50, 50])
        # lower_green = np.array([110, 13, 224])
        upper_green = np.array([100, 255, 255])
        hsv = cv2.cvtColor(oldimg, cv2.COLOR_BGR2HSV)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)  # 设置阈值，去除背景部分 
        # 就是将低于lower和高于upper的部分分别变成0，lower～upper之间的值变成255 
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        output = cv2.bitwise_and(hsv, hsv, mask=mask_blue + mask_yellow + mask_green)

        matrix = np.ones((20, 20), np.uint8)
        # my trial

        yellow_close_edge = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, matrix)
        yellow_open_edge = cv2.morphologyEx(yellow_close_edge, cv2.MORPH_OPEN, matrix)
        green_close_edge = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, matrix)
        green_open_edge = cv2.morphologyEx(green_close_edge, cv2.MORPH_OPEN, matrix)
        blue_close_edge = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, matrix)
        blue_open_edge = cv2.morphologyEx(blue_close_edge, cv2.MORPH_OPEN, matrix)

        # 根据阈值找到对应颜色
        # for debug
        """
        fig = plt.figure()
        fig.add_subplot(2, 2, 1)
        plt.title("blue_mask")
        plt.axis('off')
        plt.imshow(mask_blue, cmap="gray")
        fig.add_subplot(2, 2, 2)
        plt.title("yellow_mask")
        plt.axis('off')
        plt.imshow(mask_yellow, cmap="gray")
        fig.add_subplot(2, 2, 3)
        plt.title("green_mask")
        plt.axis('off')
        plt.imshow(mask_green, cmap="gray")
        """
        output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        # for debug
        """
        fig.add_subplot(2, 2, 4)
        plt.title("hsv_mask")
        plt.axis('off')
        plt.imshow(output, cmap="gray")
        plt.show()
        """
        matrix = np.ones((20, 20), np.uint8)
        close_edge = cv2.morphologyEx(output, cv2.MORPH_CLOSE, matrix)
        open_edge = cv2.morphologyEx(close_edge, cv2.MORPH_OPEN, matrix)
        """
        # for debug
        fig = plt.figure()
        fig.add_subplot(2, 2, 1)
        plt.axis("off")
        plt.title("close_edge")
        plt.imshow(close_edge, cmap="gray")
        fig.add_subplot(2, 2, 2)
        plt.axis("off")
        plt.title("open_edge")
        plt.imshow(open_edge, cmap="gray")
        """
        # 增加一个固定阈值二值化
        ret, binary = cv2.threshold(open_edge, 110, 255, cv2.THRESH_BINARY)
        """
        fig.add_subplot(2, 2, 3)
        plt.axis("off")
        plt.title("open_edge_binary")
        plt.imshow(binary, cmap="gray")
        plt.show()
        """
        card_contours = self.img_find_colorContours(binary, oldimg)
        card_imgs = self.img_Transform(card_contours, oldimg, pic_width, pic_height)
        colors, car_imgs = self.img_color(card_imgs, 1)


        predict_result = []
        roi = None
        card_color = None
        standard_part = []

        for i, color in enumerate(colors):
            if color in ("blue", "yellow", "green"):
                card_img = card_imgs[i]
                if not card_img.size:
                    continue

                # cv2.imshow("card_img", card_img)
                # cv2.waitKey(0)
                gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
                # for debug
                """
                fig = plt.figure()
                fig.add_subplot(1, 2, 1)
                plt.axis('off')
                plt.title('original gray image')
                plt.imshow(gray_img, cmap="gray")
                """
                # 黄、绿车牌字符比背景暗、与蓝车牌刚好相反，所以黄、绿车牌需要反向
                if color == "green" or color == "yellow":
                    gray_img = cv2.bitwise_not(gray_img)
                ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # for debug
                """
                fig.add_subplot(1, 2, 2)
                plt.axis('off')
                plt.title('after OTSU')
                plt.imshow(gray_img, cmap="gray")
                plt.show()
                if gray_img.size:
                    cv2.imshow("gray img", gray_img)
                    cv2.waitKey(0)
                """
                x_histogram = np.sum(gray_img, axis=1)
                x_min = np.min(x_histogram)
                x_average = np.sum(x_histogram) / x_histogram.shape[0]
                x_threshold = (x_min + x_average) / 2

                # for debug
                """
                arr = x_histogram.flatten()
                fig = plt.figure()
                plt.title("x_histogram")
                n, bins, patches = plt.hist(arr, bins=256, density=1, facecolor='green', alpha=0.75)
                plt.show()
                """
                wave_peaks = find_waves(x_threshold, x_histogram)
                # for debug
                """
                fig = plt.figure()
                plt.title("x_wave_peaks")
                n, bins, patches = plt.hist(wave_peaks, bins=256, density=1, facecolor='green', alpha=0.75)
                plt.show()
                """
                if len(wave_peaks) == 0:
                    # print("peak less 0:")
                    continue
                # 认为水平方向，最大的波峰为车牌区域
                wave = max(wave_peaks, key=lambda x: x[1] - x[0])
                gray_img = gray_img[wave[0]:wave[1]] if color != "green" else gray_img[wave[0]-50:wave[1]]


                # 查找垂直直方图波峰
                row_num, col_num = gray_img.shape[:2]
                # 去掉车牌上下边缘1个像素，避免白边影响阈值判断
                gray_img = gray_img[1:row_num - 1]
                y_histogram = np.sum(gray_img, axis=0)
                y_min = np.min(y_histogram)
                y_average = np.sum(y_histogram) / y_histogram.shape[0]
                y_threshold = (y_min + y_average) / 5  # U和0要求阈值偏小，否则U和0会被分成两半
                wave_peaks = find_waves(y_threshold, y_histogram)
                if len(wave_peaks) < 6:
                    # print("peak less 1:", len(wave_peaks))
                    continue

                # for debug
                """
                fig = plt.figure()
                arr = np.array(y_histogram)
                plt.title("y_histogram")
                n, bins, patches = plt.hist(arr, bins=256, density=1, facecolor='green', alpha=0.75)
                plt.show()
                
                # for debug
                fig = plt.figure()
                arr = np.array(wave_peaks)
                plt.title("y_wave_peaks")
                n, bins, patches = plt.hist(arr, bins=256, density=1, facecolor='green', alpha=0.75)
                plt.show()
                """
                wave = max(wave_peaks, key=lambda x: x[1] - x[0])
                max_wave_dis = wave[1] - wave[0]
                # 判断是否是左侧车牌边缘
                if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis / 3 and wave_peaks[0][0] == 0:
                    wave_peaks.pop(0)

                # 组合分离汉字
                cur_dis = 0
                for i, wave in enumerate(wave_peaks):
                    if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
                        break
                    else:
                        cur_dis += wave[1] - wave[0]
                if i > 0:
                    wave = (wave_peaks[0][0], wave_peaks[i][1])
                    wave_peaks = wave_peaks[i + 1:]
                    wave_peaks.insert(0, wave)

                point = wave_peaks[2]
                point_img = gray_img[:, point[0]:point[1]]
                if np.mean(point_img) < 255 / 5:
                    wave_peaks.pop(2)

                if len(wave_peaks) <= 6:
                    # print("peak less 2:", len(wave_peaks))
                    continue

                # for debug

                arr = np.array(wave_peaks)
                fig = plt.figure()
                plt.title("final_wave_peaks")
                n, bins, patches = plt.hist(arr, bins=256, density=1, facecolor='green', alpha=0.75)
                plt.show()

                # for debug
                if gray_img.size:
                    cv2.imshow("accurate height", gray_img)
                    cv2.waitKey(0)

                part_cards = seperate_card(gray_img, wave_peaks)
                #  for debug

                fig = plt.figure()
                plt.title("seperate card")
                plt.axis('off')
                i = 0
                for part in part_cards:
                    i = i + 1
                    fig.add_subplot(1, len(part_cards), i)
                    plt.axis('off')
                    plt.imshow(part, cmap="gray")
                plt.show()
                fig = plt.figure()
                plt.title("standard card parts")
                plt.axis('off')

                for i, part_card in enumerate(part_cards):
                    # 可能是固定车牌的铆钉
                    if np.mean(part_card) < 255 / 5:
                        # print("a point")
                        continue
                    part_card_old = part_card

                    w = abs(part_card.shape[1] - SZ) // 2

                    part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)
                    # for debug

                    fig.add_subplot(1, len(part_cards), i + 1)
                    plt.axis('off')
                    plt.imshow(part_card, cmap="gray")

                    standard_part.append(part_card)

                    part_card = svm_recg.preprocess_hog([part_card])

                    if i == 0:
                        resp = self.modelchinese.predict(part_card)
                        character = svm_recg.provinces[int(resp[0]) - PROVINCE_START]
                    else:
                        resp = self.model.predict(part_card)
                        character = chr(resp[0])

                    # Now apply cnn

                    if i == 0:
                        part_card = cv2_to_PIL(part_card)
                        character = cnn_recg_character.cnn_predict(part_card)
                    else:
                        part_card = cv2_to_PIL(part_card)
                        character = cnn_recg_digit.cnn_predict(part_card)

                    # 防止川被分割成川1
                    if i == 1 and character == "1":
                        standard_part = standard_part[:-1]
                        continue
                    # 判断最后一个数是否是车牌边缘，假设车牌边缘被认为是1
                    if character == "1" and i == len(part_cards) - 1 and len(predict_result) >= 7:
                        if part_card_old.shape[0] / part_card_old.shape[1] >= 7:  # 1太细，认为是边缘
                            standard_part = standard_part[:-1]
                            continue
                    predict_result.append(character)
                if len(predict_result) > 7:
                    # predict_result = predict_result[:-1]
                    standard_part = standard_part[:-1]
                # plt.show()
                roi = card_img
                card_color = color
                break
        return predict_result, roi, card_color, standard_part  # 识别到的字符、定位的车牌图像、车牌颜色

"""
 Image slant correction
"""
# 导入所需的模块
import math
import cv2 as cv
from matplotlib import pyplot as plt
from openCV_functions import gray_gauss, plt_show_raw


# 提取车牌部分图片
def license_image(slant_image):
    image_gray = gray_gauss(slant_image)  # 高斯&灰度降噪
    # Sobel 算子边缘检测 (Y方向上的检测)
    Sobel_x = cv.Sobel(image_gray, cv.CV_16S, 1, 0)  # x, y
    absX = cv.convertScaleAbs(Sobel_x)  # 转为unit 8
    image_edge = absX
    # plt_show_grey(image_edge)
    # 自适应的阈值处理
    ret, image = cv.threshold(image_edge, 0, 255, cv.THRESH_OTSU)
    # plt_show_grey(image)
    # 闭运算， 白色部位看成一个整体
    kernelX = cv.getStructuringElement(cv.MORPH_OPEN, (16, 6))
    print(kernelX)
    image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernelX, iterations=1)
    # plt_show_grey(image)
    # 去除一些白点
    kernelX = cv.getStructuringElement(cv.MORPH_RECT, (10, 1))
    kernelY = cv.getStructuringElement(cv.MORPH_RECT, (1, 10))
    # 膨胀, 腐蚀
    image = cv.dilate(image, kernelX)
    image = cv.erode(image, kernelX)
    # 腐蚀, 膨胀
    image = cv.erode(image, kernelY)
    image = cv.dilate(image, kernelY)
    # plt_show_grey(image)
    # 中值滤波去噪点
    image = cv.medianBlur(image, 19)
    # plt_show_grey(image)
    # 轮廓检测
    contours, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # 绘制轮廓
    image_rawEdge = slant_image.copy()
    cv.drawContours(image_rawEdge, contours, -1, (0, 255), 5)
    plt_show_raw(image_rawEdge)
    return image_rawEdge, contours


# 筛选出车牌位置的轮廓
def boundingRect_draw(slant_image):
    image_rawEdge, contour = license_image(slant_image)
    # 3:1 or 4:1 的长宽比开判断
    for index, item in enumerate(contour):
        # cv2.boundingRect用一个最小的矩形，把找到的形状包起来
        rect = cv.boundingRect(item)
        x = rect[0]
        y = rect[1]
        weight = rect[2]
        height = rect[3]
        # 440mm×140mm
        if (weight > (height * 2.5)) and (weight < (height * 5)):
            print(index)
            image2 = slant_image.copy()
            cv.drawContours(image2, contour, 1, (0, 0, 255), 5)
            plt_show_raw(image2)
            return image, contour


# 显示出车牌位置
def boundingRect_show(slant_image):
    image_rawEdge, contours = license_image(slant_image)
    for item in contours:
        # cv2.boundingRect用一个最小的矩形，把找到的形状包起来
        rect = cv.boundingRect(item)
        x = rect[0]
        y = rect[1]
        weight = rect[2]  # 宽度
        height = rect[3]  # 高度
        # 440mm×140mm
        if (weight > (height * 2.5)) and (weight < (height * 5)):
            c = rect
            i = item
            image = slant_image[y:y + height, x:x + weight]
            # 图像保存
            plt_show_raw(image)
            cv.imwrite('../images/test000.png', image)
            return image


# 计算斜率
def calculate_slope(image, contour):
    cnt = contour[1]
    image3 = image.copy()

    h, w = image3.shape[:2]
    [vx, vy, x, y] = cv.fitLine(cnt, cv.DIST_L2, 0, 0.01, 0.01)
    print([vx, vy, x, y])

    k = vy / vx  # Calculate Slope
    b = y - k * x  # Calculate Y-intercept
    print(k, b)
    left = b
    right = k * w + b
    img = cv.line(image3, (w, right), (0, left), (0, 255, 0), 2)
    plt_show_raw(img)
    print((w, right))
    print((0, left))
    return k


# License Image Skew Correction
def image_rotation(k):
    a = math.atan(k)
    a = math.degrees(a)
    image4 = original_image.copy()
    # 图像旋转
    h, w = image4.shape[:2]
    print(h, w)
    # 第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
    M = cv.getRotationMatrix2D((w / 2, h / 2), a, 0.8)
    # 第三个参数：变换后的图像大小
    correct_image = cv.warpAffine(image4, M, (int(w * 1.1), int(h * 1.1)))
    plt_show_raw(correct_image)

    # 显示纠正后的车牌图像
    adjusted_image = boundingRect_show(correct_image)
    return adjusted_image


original_image = cv.imread("C:/Users/72946/Desktop/Project/License_Plate_Recognition_Project/test images/ModelS.png")
image = original_image.copy()
carLicense_image, contour = boundingRect_draw(image)  # 返回定位后的车牌和轮廓
k = calculate_slope(carLicense_image, contour)
rawImage = image_rotation(k)

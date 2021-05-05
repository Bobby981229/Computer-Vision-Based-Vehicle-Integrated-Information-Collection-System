# 导入所需的模块
import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


# 显示图片
def cv_show(name, img):
    cv.imshow(name, img)
    cv.waitKey()
    cv.destroyAllWindows()


# plt 显示彩色图像
def plt_show_raw(img):
    b, g, r = cv.split(img)  # 拆分
    img = cv.merge([r, g, b])  # 组合
    plt.imshow(img)  # 显示图像
    plt.show()


# plt 显示灰度图像
def plt_show_grey(img):
    plt.imshow(img, cmap='gray')  # 指定为灰度图
    plt.show()


# 图像去噪灰度处理
def gray_gauss(image):
    image = cv.GaussianBlur(image, (3, 3), 0)
    gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    return gray_image


# 提取车牌部分图片
def license_image(images):
    image_gray = gray_gauss(images)  # 高斯&灰度降噪
    plt_show_grey(image_gray)
    # Sobel 算子边缘检测 (Y方向上的检测)
    Sobel_x = cv.Sobel(image_gray, cv.CV_16S, 1, 0)  # x, y
    absX = cv.convertScaleAbs(Sobel_x)  # 转为unit 8
    image_edge = absX
    plt_show_grey(image_edge)
    # 自适应的阈值处理
    ret, image = cv.threshold(image_edge, 0, 255, cv.THRESH_OTSU)
    plt_show_grey(image)
    # 闭运算， 白色部位看成一个整体
    kernelX = cv.getStructuringElement(cv.MORPH_OPEN, (17, 5))
    image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernelX, iterations=3)
    plt_show_grey(image)
    # 去除一些白点
    kernelX = cv.getStructuringElement(cv.MORPH_RECT, (20, 1))
    kernelY = cv.getStructuringElement(cv.MORPH_RECT, (1, 19))
    # 膨胀, 腐蚀
    image = cv.dilate(image, kernelX)
    image = cv.erode(image, kernelX)
    # 腐蚀, 膨胀
    image = cv.erode(image, kernelY)
    image = cv.dilate(image, kernelY)
    plt_show_grey(image)
    # 中值滤波去噪点
    image = cv.medianBlur(image, 15)
    plt_show_grey(image)
    # 轮廓检测
    contours, hierarchy = cv.findContours(
        image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # 绘制轮廓
    image_rawEdge = images.copy()
    cv.drawContours(image_rawEdge, contours, -1, (0, 255), 5)
    plt_show_raw(image_rawEdge)
    # 筛选出车牌位置的轮廓
    # 3:1 or 4:1 的长宽比开作为判断依据
    for item in contours:
        # cv.boundingRect是一个矩形
        rect = cv.boundingRect(item)
        x = rect[0]
        y = rect[1]
        width = rect[2]  # 宽度
        height = rect[3]  # 高度
        if (width > (height * 3)) and (width < (height * 4)):
            image = images[y:y + height, x:x + width]
            plt_show_raw(image)
            # cv.imwrite('../images/test1_result.png', image)
            return image


# 车牌字母和数字分离
def license_spilt(image):
    gray_image = gray_gauss(image)
    ret, image = cv.threshold(gray_image, 0, 255, cv.THRESH_OTSU)
    plt_show_grey(image)
    # 计算二值图像黑白点的个数，处理绿牌照问题，让车牌号码始终为白色
    area_white = 0
    area_black = 0
    height, width = image.shape
    for i in range(height):
        for j in range(width):
            if image[i, j] == 255:
                area_white += 1
            else:
                area_black += 1
    if area_white > area_black:
        ret, image = cv.threshold(
            image, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY_INV)
        plt_show_grey(image)
    # 闭运算,是白色部分练成整体
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    image = cv.dilate(image, kernel)
    contours, hierarchy = cv.findContours(
        image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # 筛选出各个字符的位置的轮廓
    words = []
    word_images = []
    for item in contours:
        # cv2.boundingRect用一个最小的矩形，把找到的形状包起来
        word = []
        rect = cv.boundingRect(item)
        x = rect[0]
        y = rect[1]
        weight = rect[2]
        height = rect[3]
        word.append(x)
        word.append(y)
        word.append(weight)
        word.append(height)
        words.append(word)
    words = sorted(words, key=lambda s: s[0], reverse=False)
    i = 0
    for word in words:
        if (word[3] > (word[2] * 1.8)) and (word[3] < (word[2] * 3.5)):
            i = i + 1
            split_image = image[word[1]:word[1] +
                                word[3], word[0]:word[0] + word[2]]
            word_images.append(split_image)
            # cv.imwrite('../words/' + str(i) + '.png', split_image)
    # 绿牌要改为8，蓝牌为7，显示所用
    for k, z in enumerate(word_images):
        plt.subplot(1, i, k + 1)
        plt.imshow(word_images[k], cmap='gray')
    plt.show()

    return word_images


# 图像匹配前预处理
def pre_processing(path):
    # 显示原图
    img_raw = cv.imread(path)
    plt_show_raw(img_raw)
    # 高斯降噪
    img_Gaussian = cv.GaussianBlur(img_raw, (3, 3), 0)
    # 灰度处理, 二值化处理
    img_gray = cv.cvtColor(img_Gaussian, cv.COLOR_RGB2GRAY)
    plt_show_grey(img_gray)
    # 自适应阈值处理
    retval, img_threshold = cv.threshold(img_gray, 0, 255, cv.THRESH_OTSU)
    plt_show_grey(img_threshold)
    return img_raw, img_gray, img_threshold


# 读取文件夹下所有图片的，输入参数是文件名
def read_images(image_name):
    referImg_list = []  # 存储路径
    for fileName in os.listdir(image_name):  # 拿出文件夹下所有文件
        referImg_list.append(image_name + '/' + fileName)  # 存入list中
    return referImg_list


template = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
            'X', 'Y', 'Z',
            '藏', '川', '鄂', '甘', '赣', '贵', '桂', '黑', '沪', '吉', '冀', '津', '晋', '京', '辽', '鲁', '蒙', '闽', '宁',
            '青', '琼', '陕', '苏', '皖', '湘', '新', '渝', '豫', '粤', '云', '浙']


# 模板列表（匹配车牌的字符和数字）
def get_chinese_words():
    chin_words = []
    for i in range(34, 64):
        c_word = read_images('../refer_images/' + template[i])
        chin_words.append(c_word)
    return chin_words


# 英文模板列表（只匹配车牌的第二个字符）
def get_english_words():
    eng_words = []
    for i in range(10, 34):
        e_word = read_images('../refer_images/' + template[i])
        eng_words.append(e_word)
    return eng_words


# 英文数字模板列表（匹配车牌后面的字符）
def get_eng_num_words():
    eng_num_word = []
    for i in range(0, 34):
        word = read_images('../refer_images/' + template[i])
        eng_num_word.append(word)
    return eng_num_word


# 读取一个模板地址与图片进行匹配，返回得分
def template_score(template, image):
    # fromfile()函数读回数据时需要用户指定元素类型，并对数组的形状进行适当的修改
    template_img = cv.imdecode(np.fromfile(template, dtype=np.uint8), 1)
    template_img = cv.cvtColor(template_img, cv.COLOR_RGB2GRAY)  # 灰度处理
    ret, template_img = cv.threshold(
        template_img, 0, 255, cv.THRESH_OTSU)  # 阈值处理
    image_input = image.copy()
    height, width = image_input.shape  # 目标图片和模板的尺寸要相同
    template_img = cv.resize(template_img, (width, height))  # 改变指定图片的尺寸大小
    # cv.TM_CCOEFF 计算相关系数, 返回值越大越相似
    result = cv.matchTemplate(image_input, template_img,
                              cv.TM_CCOEFF)  # 模板匹配 (图片, 模板, 参数)
    return result[0][0]


# 使用处理过的图像，与样本模板进行匹配
def template_matching(word_images):
    chin_words_list = get_chinese_words()  # 获取中文样本模板图片
    eng_words_list = get_english_words()  # 获取英文样本模板图片
    eng_num_words_list = get_eng_num_words()  # 获取英文和数字样本模板图片

    results = []  # 最终车牌结果
    for index, word_image in enumerate(word_images):
        if index == 0:  # 第一个字符
            best_score = []  # 最高匹配度值
            for words in chin_words_list:  # 循环字符
                score = []  # 存储所有的相似度数值
                for word in words:  # 进行模板遍历匹配
                    result = template_score(word, word_image)
                    score.append(result)  # 储存相似度结果
                best_score.append(max(score))  # 获取最大相似度值
            i = best_score.index(max(best_score))  # 获取最大值的下标
            r = template[34 + i]
            results.append(r)  # 添加在results[]中
            continue
        if index == 1:  # 第二个字符
            best_score = []
            for eng_word_list in eng_words_list:
                score = []
                for eng_word in eng_word_list:
                    result = template_score(eng_word, word_image)
                    score.append(result)
                best_score.append(max(score))
            i = best_score.index(max(best_score))
            r = template[10 + i]
            results.append(r)
            continue
        else:  # 匹配其他字符
            best_score = []
            for eng_num_word_list in eng_num_words_list:
                score = []
                for eng_num_word in eng_num_word_list:
                    result = template_score(eng_num_word, word_image)
                    score.append(result)
                best_score.append(max(score))
            i = best_score.index(max(best_score))
            r = template[i]
            results.append(r)
            continue
    return results

import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


# Shown image
def cv_show(name, img):
    cv.imshow(name, img)
    cv.waitKey()
    cv.destroyAllWindows()


# plt Show origin image
def plt_show_raw(img):
    b, g, r = cv.split(img)  # split image by three channel
    img = cv.merge([r, g, b])  # merge
    plt.imshow(img)
    plt.show()


# plt Display grayscale image
def plt_show_grey(img):
    plt.imshow(img, cmap='gray')  # Specifies as a grayscale image
    plt.show()


# Image denoising and grayscale processing
def gray_gauss(image):
    image = cv.GaussianBlur(image, (3, 3), 0)
    gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    return gray_image


def license_image(images):
    """
    Get license plate picture
    :param images: Vehicle image
    :return: licence plate image
    """
    image_gray = gray_gauss(images)  # Gaussian & grayscale noise reduction
    plt_show_grey(image_gray)
    # Sobel operator edge detection (detection in the Y direction)
    Sobel_x = cv.Sobel(image_gray, cv.CV_16S, 1, 0)  # x, y
    absX = cv.convertScaleAbs(Sobel_x)  # 转为unit 8
    image_edge = absX
    plt_show_grey(image_edge)
    # Adaptive threshold processing
    ret, image = cv.threshold(image_edge, 0, 255, cv.THRESH_OTSU)
    plt_show_grey(image)
    # Closed, white parts seen as a whole
    kernelX = cv.getStructuringElement(cv.MORPH_OPEN, (17, 5))
    image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernelX, iterations=3)
    plt_show_grey(image)
    # Removal of some white spots
    kernelX = cv.getStructuringElement(cv.MORPH_RECT, (20, 1))
    kernelY = cv.getStructuringElement(cv.MORPH_RECT, (1, 19))
    # Expansion and corrosion
    image = cv.dilate(image, kernelX)
    image = cv.erode(image, kernelX)
    # Corrosion and expansion
    image = cv.erode(image, kernelY)
    image = cv.dilate(image, kernelY)
    plt_show_grey(image)
    # Median filtering to remove noise
    image = cv.medianBlur(image, 15)
    plt_show_grey(image)
    # Contours detection
    contours, hierarchy = cv.findContours(
        image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # Draw the contours
    image_rawEdge = images.copy()
    cv.drawContours(image_rawEdge, contours, -1, (0, 255), 5)
    plt_show_raw(image_rawEdge)
    # Get the outline of the license plate position
    # Aspect ratio of 3:1 or 4:1 to judgement
    for item in contours:
        rect = cv.boundingRect(item)
        x = rect[0]
        y = rect[1]
        width = rect[2]  # width
        height = rect[3]  # high
        if (width > (height * 3)) and (width < (height * 4)):
            image = images[y:y + height, x:x + width]
            plt_show_raw(image)
            # cv.imwrite('../images/test1_result.png', image)
            return image


def license_spilt(image):
    """
    Separation of licence plate letters and numbers
    :param image: licence plate image
    :return: characters images
    """
    gray_image = gray_gauss(image)
    ret, image = cv.threshold(gray_image, 0, 255, cv.THRESH_OTSU)
    plt_show_grey(image)
    # Calculate the number of black and white dots
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
    # Close operation, the white part is practiced as a whole
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    image = cv.dilate(image, kernel)
    contours, hierarchy = cv.findContours(
        image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # Filter the outline of the position of the individual characters
    words = []
    word_images = []
    for item in contours:
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
    # The green plate is to be changed to 8
    for k, z in enumerate(word_images):
        plt.subplot(1, i, k + 1)
        plt.imshow(word_images[k], cmap='gray')
    plt.show()
    return word_images


def pre_processing(path):
    """
    Pre-processing before image matching
    :param path: image path
    :return: original, gray and threshold images
    """
    # Show original image
    img_raw = cv.imread(path)
    plt_show_raw(img_raw)
    # Gaussian noise
    img_Gaussian = cv.GaussianBlur(img_raw, (3, 3), 0)
    # Gray scale processing, binary processing
    img_gray = cv.cvtColor(img_Gaussian, cv.COLOR_RGB2GRAY)
    plt_show_grey(img_gray)
    # Adaptive threshold processing
    retval, img_threshold = cv.threshold(img_gray, 0, 255, cv.THRESH_OTSU)
    plt_show_grey(img_threshold)
    return img_raw, img_gray, img_threshold


def read_images(image_name):
    """
    Retrieve all images in a folder with the input parameter - file name
    :param image_name: image name
    :return:
    """
    referImg_list = []  # Storage path
    for fileName in os.listdir(image_name):  # Take out all the files in the folder
        referImg_list.append(image_name + '/' + fileName)  # Store in list
    return referImg_list


# Chinese, English, number Characters templates
char_dict = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
            'X', 'Y', 'Z',
            '藏', '川', '鄂', '甘', '赣', '贵', '桂', '黑', '沪', '吉', '冀', '津', '晋', '京', '辽', '鲁', '蒙', '闽', '宁',
            '青', '琼', '陕', '苏', '皖', '湘', '新', '渝', '豫', '粤', '云', '浙']


def get_chinese_words():
    """
    List of templates (matching the characters and numbers of the license plate)
    :return:
    """
    chin_words = []
    for i in range(34, 64):
        c_word = read_images('../refer_images/' + char_dict[i])
        chin_words.append(c_word)
    return chin_words


def get_english_words():
    """
    English template list (matches only the second character of the license plate)
    :return:
    """
    eng_words = []
    for i in range(10, 34):
        e_word = read_images('../refer_images/' + char_dict[i])
        eng_words.append(e_word)
    return eng_words


def get_eng_num_words():
    """
    List of alphanumeric templates
    :return:
    """
    eng_num_word = []
    for i in range(0, 34):
        word = read_images('../refer_images/' + char_dict[i])
        eng_num_word.append(word)
    return eng_num_word


def char_dict_score(char_dict, image):
    """
    Retrieve a template address to match with an image and return a score
    :param char_dict: characters dictionary  - template
    :param image: character image
    :return: score
    """
    char_dict_img = cv.imdecode(np.fromfile(char_dict, dtype=np.uint8), 1)
    char_dict_img = cv.cvtColor(char_dict_img, cv.COLOR_RGB2GRAY)  # Grayscale processing
    ret, char_dict_img = cv.threshold(
        char_dict_img, 0, 255, cv.THRESH_OTSU)  # Threshold processing
    image_input = image.copy()
    height, width = image_input.shape  # The target image and the template should be the same size
    char_dict_img = cv.resize(char_dict_img, (width, height))  # Change the size of the specified image
    # cv.TM_CCOEFF Calculate the correlation coefficient, the larger the value the more similar it is
    result = cv.matchTemplate(image_input, char_dict_img,
                              cv.TM_CCOEFF)  # Template matching (images, templates, parameters)
    return result[0][0]


def template_matching(word_images):
    """
    Using processed images, matched to sample templates
    :param word_images: character images need to recognition
    :return: Recognition result
    """
    chin_words_list = get_chinese_words()  # Obtain the Chinese sample template image
    eng_words_list = get_english_words()  # Obtain the English sample template image
    eng_num_words_list = get_eng_num_words()  # Obtain English and digital sample template images

    results = []  # License plate result
    for index, word_image in enumerate(word_images):
        if index == 0:  # FirstLetter
            best_score = []  # Maximum match value
            for words in chin_words_list:
                score = []  # Stores all similarity values
                for word in words:  # Perform template traversal matching
                    result = char_dict_score(word, word_image)
                    score.append(result)  # Stores all similarity values
                best_score.append(max(score))  # Get the maximum similarity value
            i = best_score.index(max(best_score))  # Get the subscript of the maximum value
            r = char_dict[34 + i]
            results.append(r)  # Add to results[]
            continue
        if index == 1:  # The second characters
            best_score = []
            for eng_word_list in eng_words_list:
                score = []
                for eng_word in eng_word_list:
                    result = char_dict_score(eng_word, word_image)
                    score.append(result)
                best_score.append(max(score))
            i = best_score.index(max(best_score))
            r = char_dict[10 + i]
            results.append(r)
            continue
        else:  # Matching the other characters
            best_score = []
            for eng_num_word_list in eng_num_words_list:
                score = []
                for eng_num_word in eng_num_word_list:
                    result = char_dict_score(eng_num_word, word_image)
                    score.append(result)
                best_score.append(max(score))
            i = best_score.index(max(best_score))
            r = char_dict[i]
            results.append(r)
            continue
    return results

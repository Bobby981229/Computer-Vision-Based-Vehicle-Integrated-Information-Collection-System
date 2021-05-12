"""
 Image slant correction -- Traditional method
"""
# Importing the required modules
import math
import cv2 as cv
from matplotlib import pyplot as plt
from openCV_functions import gray_gauss, plt_show_raw


def license_image(slant_image):
    """
    Extracting the license plate image
    :param slant_image: a slant vehicle image
    :return: Vehicle contours
    """
    image_gray = gray_gauss(slant_image)  # Gaussian & greyscale noise reduction
    # Sobel operator edge detection (detection in the Y direction)
    Sobel_x = cv.Sobel(image_gray, cv.CV_16S, 1, 0)  # x, y
    absX = cv.convertScaleAbs(Sobel_x)  # to unit 8
    image_edge = absX
    # plt_show_grey(image_edge)
    # Adaptive threshold processing
    ret, image = cv.threshold(image_edge, 0, 255, cv.THRESH_OTSU)
    # plt_show_grey(image)
    kernelX = cv.getStructuringElement(cv.MORPH_OPEN, (16, 6))
    print(kernelX)
    image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernelX, iterations=1)  # Closed, white parts seen as a whole
    # plt_show_grey(image)
    # Removal of some white spots
    kernelX = cv.getStructuringElement(cv.MORPH_RECT, (10, 1))
    kernelY = cv.getStructuringElement(cv.MORPH_RECT, (1, 10))
    # Expansion and corrosion
    image = cv.dilate(image, kernelX)
    image = cv.erode(image, kernelX)
    # Corrosion and expansion
    image = cv.erode(image, kernelY)
    image = cv.dilate(image, kernelY)
    # plt_show_grey(image)
    image = cv.medianBlur(image, 19)  # Median filtering to remove noise
    # plt_show_grey(image)
    contours, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # Contours detection
    image_rawEdge = slant_image.copy()
    cv.drawContours(image_rawEdge, contours, -1, (0, 255), 5)  # Draw the contours
    plt_show_raw(image_rawEdge)
    return image_rawEdge, contours


def boundingRect_draw(slant_image):
    """
    Filter the outline of the location of the license plate
    :param slant_image:
    :return:
    """
    image_rawEdge, contour = license_image(slant_image)
    # Aspect ratio of 3:1 or 4:1 to judgement
    for index, item in enumerate(contour):
        rect = cv.boundingRect(item)  # Using the smallest rectangle, wrap the found shape
        x = rect[0]
        y = rect[1]
        weight = rect[2]
        height = rect[3]
        if (weight > (height * 2.5)) and (weight < (height * 5)):
            print(index)
            image2 = slant_image.copy()
            cv.drawContours(image2, contour, 1, (0, 0, 255), 5)
            plt_show_raw(image2)
            return image, contour


def calculate_slope(image, contour):
    """
    Calculate slope
    :param image: licence plate image
    :param contour:
    :return: slop - k
    """
    cnt = contour[1]  # Coordinate points on the contour
    image3 = image.copy()

    h, w = image3.shape[:2]
    # Returns the direction vector and the coordinate points on the x,y axis
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


def image_rotation(k):
    """
    License Image Skew Correction
    :param k: slop
    :return: vehicle after correction
    """
    a = math.atan(k)
    a = math.degrees(a)
    image4 = original_image.copy()
    # Image rotation
    h, w = image4.shape[:2]
    print(h, w)
    # Centre of rotation, angle of rotation, scaling
    M = cv.getRotationMatrix2D((w / 2, h / 2), a, 0.8)
    correct_image = cv.warpAffine(image4, M, (int(w * 1.1), int(h * 1.1)))
    plt_show_raw(correct_image)

    # Display of corrected license plate images
    adjusted_image = boundingRect_draw(correct_image)
    return adjusted_image


original_image = cv.imread("C:/Users/72946/Desktop/Project/test images/ModelS.png")
image = original_image.copy()
carLicense_image, contour = boundingRect_draw(image)  # Returning the LP and outline of the car after positioning
k = calculate_slope(carLicense_image, contour)
rawImage = image_rotation(k)

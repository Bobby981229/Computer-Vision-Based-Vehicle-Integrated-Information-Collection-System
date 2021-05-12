"""
 Image slant correction -- New method
"""

# Importing the required modules
import cv2
import numpy as np
import tkinter as tk
import tkinter.messagebox


def locate_and_correct(img_org, img_mask):
    """
    This function uses cv2 to detect the edges of img_mask and obtains the edge coordinates of the LP
    area (stored in contours) and the coordinates of the four endpoints of the smallest outer rectangle,
    From the edge coordinates of the license plate, the nearest points to the 4 endpoints of the minimum outer
    rectangle are calculated to be the four endpoints of the parallelogram license plate, thus realising the
    positioning and correction of the number plate.
    :param img_org: Origin image
    :param img_mask: Binarised image with image separation by U_Net
    :return: License plate after slant correction
    """
    # cv2.imshow('img_mask',img_mask)
    # cv2.waitKey(0)
    # ret,thresh = cv2.threshold(img_mask[:,:,0],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # Binarization
    # cv2.imshow('thresh',thresh)
    # cv2.waitKey(0)
    try:
        contours, hierarchy = cv2.findContours(img_mask[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except:  # Preventing opencv version inconsistencies to report errors
        ret, contours, hierarchy = cv2.findContours(img_mask[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not len(contours):  # contours length is 0 means that the plate is not detected
        tk.messagebox.showerror('Error', 'Failed to recognition!')
        print("Failed")
        return [], []
    else:
        corrected_img = []
        img_org_copy = img_org.copy()  # For plotting the outline of a positioned license plate
        for i, cont in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cont)  # Get the smallest outer rectangle
            img_lp_mask = img_mask[y:y + h, x:x + w]  # Extract the labeled lp area

            # Filter by the following criteria
            if np.mean(img_lp_mask) >= 75 and w > 15 and h > 15:
                # Get the minimum outer rectangle, centre point coordinates, width, height and rotation angle
                rect = cv2.minAreaRect(cont)
                box = cv2.boxPoints(rect).astype(np.int32)  # Get the coordinates of the four vertices of the rectangle
                # cv2.drawContours(img_mask, contours, -1, (0, 0, 255), 2)
                # cv2.drawContours(img_mask, [box], 0, (0, 255, 0), 2)
                # cv2.imshow('img_mask',img_mask)
                # cv2.waitKey(0)
                cont = cont.reshape(-1, 2).tolist()

                box = sorted(box, key=lambda xy: xy[0])  # Sort by left and right
                # The first 2 are the coordinates of the left side and the last 2 are the coordinates of the right side
                box_left, box_right = box[:2], box[2:]
                box_left = sorted(box_left, key=lambda x: x[1])  # Then sort by top and bottom
                box_right = sorted(box_right, key=lambda x: x[1])  # Coordinates of the top right and bottom right endpoints
                box = np.array(box_left + box_right)  # [top left, bottom left, top right, bottom right]
                # print(box)
                # The four coordinates here are the four coordinates of the smallest outer rectangle
                x0, y0 = box[0][0], box[0][1]
                x1, y1 = box[1][0], box[1][1]
                x2, y2 = box[2][0], box[2][1]
                x3, y3 = box[3][0], box[3][1]

                def point_to_line_distance(X, Y):
                    if x2 - x0:
                        k_up = (y2 - y0) / (x2 - x0)  # Slope is not infinity
                        d_up = abs(k_up * X - Y + y2 - k_up * x2) / (k_up ** 2 + 1) ** 0.5
                    else:  # Slope infinity
                        d_up = abs(X - x2)
                    if x1 - x3:
                        k_down = (y1 - y3) / (x1 - x3)  # Slope is not infinity
                        d_down = abs(k_down * X - Y + y1 - k_down * x1) / (k_down ** 2 + 1) ** 0.5
                    else:  # Slope infinity
                        d_down = abs(X - x1)
                    return d_up, d_down

                d0, d1, d2, d3 = np.inf, np.inf, np.inf, np.inf
                l0, l1, l2, l3 = (x0, y0), (x1, y1), (x2, y2), (x3, y3)

                """
                Calculate the distances between the coordinates in the cont and the four coordinates of the rectangle 
                and the distances to the top and bottom lines, add the distances and weights, and successfully calculate
                the coordinates of the four vertices of the quadrilateral.
                """
                for each in cont:
                    x, y = each[0], each[1]
                    dis0 = (x - x0) ** 2 + (y - y0) ** 2
                    dis1 = (x - x1) ** 2 + (y - y1) ** 2
                    dis2 = (x - x2) ** 2 + (y - y2) ** 2
                    dis3 = (x - x3) ** 2 + (y - y3) ** 2
                    d_up, d_down = point_to_line_distance(x, y)
                    weight = 0.975
                    if weight * d_up + (1 - weight) * dis0 < d0:  # d0 updated
                        d0 = weight * d_up + (1 - weight) * dis0
                        l0 = (x, y)
                    if weight * d_down + (1 - weight) * dis1 < d1:
                        d1 = weight * d_down + (1 - weight) * dis1
                        l1 = (x, y)
                    if weight * d_up + (1 - weight) * dis2 < d2:
                        d2 = weight * d_up + (1 - weight) * dis2
                        l2 = (x, y)
                    if weight * d_down + (1 - weight) * dis3 < d3:
                        d3 = weight * d_down + (1 - weight) * dis3
                        l3 = (x, y)

                p0 = np.float32([l0, l1, l2, l3])  # Top left, bottom left, top right, bottom right
                p1 = np.float32([(0, 0), (0, 80), (240, 0), (240, 80)])  # Required rectangle  - LP
                transform_mat = cv2.getPerspectiveTransform(p0, p1)  # Composition of the conversion matrix
                lic = cv2.warpPerspective(img_org, transform_mat, (240, 80))  # Carrying out plate correction

                corrected_img.append(lic)
                # Plot the outline of the positioned license plate on the original image
                cv2.drawContours(img_org_copy, [np.array([l0, l1, l3, l2])], -1, (0, 0, 255), 2)
    return img_org_copy, corrected_img


def erode(img_mask):
    """
    morphological operation to mask - img_mask
    :param img_mask: mask img_mask
    :return:
    """
    # print(img_mask.shape)
    gray = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imshow("binary", binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))  # Defining the shape and size of structural elements
    img = cv2.erode(binary, kernel)  # Corrosion operation
    # cv2.imshow("erode_demo", img)
    return img
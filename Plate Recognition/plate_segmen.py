"""
Segmentation of license plate areas using the U-net model
"""
import os
import cv2
import numpy as np
import tkinter as tk
import tkinter.messagebox
from tensorflow import keras


def unet_predict(unet, img_src_path):
    """
    Predict the licence plate by U-Net model
    :param unet: U-Net Model
    :param img_src_path: image path
    :return:
    """
    img_src = cv2.imdecode(np.fromfile(
        img_src_path, dtype=np.uint8), -1)  # reading
    # img_src=cv2.imread(img_src_path)
    if img_src.shape != (512, 512, 3):
        img_src = cv2.resize(img_src, dsize=(512, 512))[:, :, :3]
    img_src = img_src.reshape(1, 512, 512, 3)  # shape image to (1,512,512,3)
    img_mask = unet.predict(img_src)

    img_src = img_src.reshape(512, 512, 3)  # Reshape the original image to 3 dimensions
    img_mask = img_mask.reshape(512, 512, 3)  # Reshape the predicted image to 3 dimensions

    img_mask = img_mask / np.max(img_mask) * 255  # Normalised and multiplied by 255
    img_mask[:, :, 2] = img_mask[:, :, 1] = img_mask[:, :, 0]  # All three channels remain the same
    img_mask = img_mask.astype(np.uint8)  # Convert img_mask type to int type

    return img_src, img_mask


def locate(img_src, img_mask, img_name):
    """
    This function uses cv2 to detect the edges of img_mask and obtains the edge coordinates of the LP
    area (stored in contours) and the coordinates of the four endpoints of the smallest outer rectangle,
    From the edge coordinates of the license plate, the nearest points to the 4 endpoints of the minimum outer
    rectangle are calculated to be the four endpoints of the parallelogram license plate, thus realising the
    positioning and correction of the number plate.
    :param img_src: original image
    :param img_mask: Binarised image with image separation by U_Net
    :return: License plate after slant correction
    """
    contours, hierarchy = cv2.findContours(
        img_mask[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not len(contours):
        tk.messagebox.showerror('Error', 'Failed to recognition!')
        print("Failed")
    else:
        flag = 0  # Default Flag is 0 because there is not necessarily a license plate area
        for i, cont in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cont)  # Get the smallest outer rectangle
            img_cut_mask = img_mask[y:y + h, x:x + w]  # Extract the labeled lp area

            if np.mean(img_cut_mask) >= 75 and w > 15 and h > 15:
                # Get the minimum outer rectangle, centre point coordinates, width, height and rotation angle
                rect = cv2.minAreaRect(cont)
                box = cv2.boxPoints(rect).astype(np.int32)  # Get the coordinates of the four vertices of the rectangle
                cv2.drawContours(img_mask, contours, -1, (0, 0, 255), 2)
                cv2.drawContours(img_mask, [box], 0, (0, 255, 0), 2)

                cont = cont.reshape(-1, 2).tolist()
                # Sort the coordinates of the smallest outer rectangle to end up with [top left, bottom left, top right, bottom right]
                # Sort by left and right first
                box = sorted(box, key=lambda xy: xy[0])
                # The first 2 are the coordinates of the left side and the last 2 are the coordinates of the right side
                box_left, box_right = box[:2], box[2:]
                # Then sort by top and bottom
                box_left = sorted(box_left, key=lambda x: x[1])
                # Coordinates of the top right and bottom right endpoints
                box_right = sorted(box_right, key=lambda x: x[1])
                box = np.array(box_left + box_right)  # [top left, bottom left, top right, bottom right]

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
                    if weight * d_up + (1 - weight) * dis0 < d0:
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

                p0 = np.float32([l0, l1, l2, l3])
                p1 = np.float32([(0, 0), (0, 80), (240, 0), (240, 80)])
                transform_mat = cv2.getPerspectiveTransform(p0, p1)  # Construct the transformation matrix
                lic = cv2.warpPerspective(
                    img_src, transform_mat, (240, 80))  # Correction of license plate

                if len(contours) == 1:  # Only one area can be considered a license plate area
                    flag += 1
                    print('saving ', save_path + name[0:7] + '.jpg')
                    cv2.imencode('.jpg', lic)[1].tofile(
                        save_path + img_name[0:7] + '.jpg')

        if not flag:
            print("Failed to recognition!")
            tk.messagebox.showerror('Error', 'No plate area detected or plate area too small !')


if __name__ == '__main__':
    test_path = '../labelme/labeled_img/'  # Use Baidu API labeled the images
    save_path = '../labelme/license_img/'  # images path
    unet_path = '../models/unet_green10.h5'
    listdir = os.listdir(test_path)
    unet = keras.models.load_model(unet_path)
    # print(listdir)
    for name in listdir:
        print(name)
        img_src_path = test_path + name
        img_src, img_mask = unet_predict(unet, img_src_path)
        locate(img_src, img_mask, name)

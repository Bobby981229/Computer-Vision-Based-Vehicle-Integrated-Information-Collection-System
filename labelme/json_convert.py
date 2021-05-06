import os
import cv2
import numpy as np

path = r'../labelme/json/'  # path为json文件存放的路径
json_file = os.listdir(path)  # /json文件夹下的所有json文件
items_num = len(json_file)  # json文件的个数

# 将json文件label转换为到data文件夹
for i in range(items_num):
    os.system('labelme_json_to_dataset ../labelme/json/%d.json -o ../labelme/data/%d_json' % (i, i))


#  如果文件夹不存在, 则创建一个新的文件夹
image_path = '../labelme/json_image/'
label_path = '../labelme/json_label/'
if not os.path.exists(image_path):
    os.makedirs(image_path)
if not os.path.exists(label_path):
    os.makedirs(label_path)


for i in range(items_num):
    print(i)
    img = cv2.imread('../labelme/data/%d_json/img.png' % i)
    label = cv2.imread('../labelme/data/%d_json/label.png' % i)
    print(img.shape)
    label = label / np.max(label[:, :, 2]) * 255
    label[:, :, 0] = label[:, :, 1] = label[:, :, 2]
    print(np.max(label[:, :, 2]))
    # cv2.imshow('l',label)
    # cv2.waitKey(0)
    print(set(label.ravel()))
    cv2.imwrite(image_path + '%d.png' % i, img)
    cv2.imwrite(label_path + '%d.png' % i, label)


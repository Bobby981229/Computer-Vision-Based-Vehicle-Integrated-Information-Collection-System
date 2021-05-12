from tensorflow.keras import layers, models
import numpy as np
import cv2
import os


def cnn_train():
    """
    Training cnn model
    :return: cnn model - cnn.h5
    """

    # Chinese licence plate (LP) characters dictionary
    char_dict = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9, "苏": 10,
                 "浙": 11, "皖": 12, "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19, "桂": 20,
                 "琼": 21, "川": 22, "贵": 23, "云": 24, "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29, "新": 30,
                 "0": 31, "1": 32, "2": 33, "3": 34, "4": 35, "5": 36, "6": 37, "7": 38, "8": 39, "9": 40,
                 "A": 41, "B": 42, "C": 43, "D": 44, "E": 45, "F": 46, "G": 47, "H": 48, "J": 49, "K": 50,
                 "L": 51, "M": 52, "N": 53, "P": 54, "Q": 55, "R": 56, "S": 57, "T": 58, "U": 59, "V": 60,
                 "W": 61, "X": 62, "Y": 63, "Z": 64}

    # Read data set
    # LP number dataset path (Size: 240 × 80)
    img_path = 'C:/Users/72946/Desktop/Project/License_Plate_Recognition_Project/labelme/cnn_dataset/'
    img_name = sorted(os.listdir(img_path))  # LP number
    len_name = len(img_name)  # the length of LP number - 7
    X_train, y_train = [], []  # images training set and training set label
    # traverse the characters from each LP
    for i in range(len_name):
        print("Reading the image No.%d " % i)
        img = cv2.imdecode(np.fromfile(img_path + img_name[i], dtype=np.uint8), -1)  # training images
        label = [char_dict[name] for name in img_name[i][0:7]]  # The first 7 chars of the image name as training label
        X_train.append(img)  # add PL images into X_train list
        y_train.append(label)  # add label (image name) into y_train list
    X_train = np.array(X_train)
    y_train = [np.array(y_train)[:, i] for i in range(7)]  # y_train is a list of length 7

    # cnn model
    Input = layers.Input(80, 240, 3)  # LP image shape (80,240,3)
    conv = Input
    conv = layers.Conv2D(kernel_size=(3, 3), filters=16, padding='same', strides=1, activation='relu')(conv)
    conv = layers.MaxPool2D(pool_size=(2, 2), padding='same', strides=2)(conv)
    for i in range(3):
        conv = layers.Conv2D(kernel_size=(3, 3), filters=32 * 2 ** i, padding='valid', activation='relu')(conv)
        conv = layers.Conv2D(kernel_size=(3, 3), filters=32 * 2 ** i, padding='valid', activation='relu')(conv)
        conv = layers.MaxPool2D(pool_size=(2, 2), padding='same', strides=2)(conv)
        conv = layers.Dropout(0.5)(conv)
    conv = layers.Flatten()(conv)
    conv = layers.Dropout(0.3)(conv)
    # The 7 outputs correspond to 7 characters of LP, and each output has 65 category probabilities
    Output = [layers.Dense(65, activation='softmax', name='c%d' % (i + 1))(conv) for i in range(7)]
    model = models.Model(inputs=Input, outputs=Output)
    model.summary()
    model.compile(optimizer='adam',  # Adaptive Moment Estimation
                  loss='sparse_categorical_crossentropy',  # y_train is not one-hot encoded
                  metrics=['accuracy'])

    # Model training
    print("Start training CNN model...")
    model.fit(X_train, y_train, epochs=35)  # The total loss is the sum of 7 losses,
    model.save('../models/cnn.h5')
    print('cnn.h5 Saved successfully!!!')


# License Plate Recognition
def cnn_predict(cnn_model, plate_image):
    """
    Input cnn model, and LP image
    :param cnn_model: cnn model, cnn.h5
    :param plate_image: waited recognition LP image
    :return: LP image and recognition results
    """
    char_dict = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫",
                  "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2",
                  "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M",
                  "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    Lic_pred = []
    for lic in plate_image:
        lic_pred = cnn_model.predict(lic.reshape(1, 80, 240, 3))  # The predicted shape should be (1, 80, 240, 3)
        lic_pred = np.array(lic_pred).reshape(7, 65)  # Convert list to array
        # Count the number of those with a predicted probability value greater than 80%
        if len(lic_pred[lic_pred >= 0.8]) >= 4:  # >= 4 considered high recognition rate
            chars = ''
            for arg in np.argmax(lic_pred, axis=1):  # Take the value with the highest probability in each row
                chars += char_dict[arg]
            chars = chars[0:2] + '·' + chars[2:]
            Lic_pred.append((lic, chars))  # Store the LP and the recognition result together in Lic_pred
    return Lic_pred


def main():
    """Set the parameters and call the functions"""
    cnn_train()  # After training, get cnn.h5, which is used for LPR


if __name__ == "__main__":
    """Execute the current module"""
    main()

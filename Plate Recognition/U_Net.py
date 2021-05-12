import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def u_net_train():
    """
    Training a U-Net model for predicting licence plate position
    :return: U-Net Model
    """
    height = 512  # image size is 512 × 512
    width = 512
    path = '../labelme/'
    items_names = os.listdir(path + 'train_image_green')
    items_numbers = len(items_names)
    print(items_numbers)
    X_train, y_train = [], []
    for i in range(items_numbers):
        print("Reading the No. %d image" % i)
        img = cv2.imread(path + 'train_image_green/%d.png' % i)
        label = cv2.imread(path + 'train_label_green/%d.png' % i)
        X_train.append(img)
        y_train.append(label)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    #  Shrink path module to under-sampling
    def ContractingPathBlock(x, nb_filter, kernel_size, strides=(1, 1), padding='same'):
        x = layers.Conv2D(nb_filter, kernel_size, strides=strides, padding=padding)(x)
        x = layers.BatchNormalization(axis=3)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        return x

    #  Expand path module to up-sampling
    def ExpansivePathBlock(x, filters, kernel_size, strides=(2, 2), padding='same'):
        x = layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)
        x = layers.BatchNormalization(axis=3)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        return x

    # Iterative Shrinkage Module
    input = layers.Input(shape=(height, width, 3))

    conv1 = ContractingPathBlock(input, 8, (3, 3))
    conv1 = ContractingPathBlock(conv1, 8, (3, 3))
    pool1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)

    conv2 = ContractingPathBlock(pool1, 16, (3, 3))
    conv2 = ContractingPathBlock(conv2, 16, (3, 3))
    pool2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

    conv3 = ContractingPathBlock(pool2, 32, (3, 3))
    conv3 = ContractingPathBlock(conv3, 32, (3, 3))
    pool3 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)

    conv4 = ContractingPathBlock(pool3, 64, (3, 3))
    conv4 = ContractingPathBlock(conv4, 64, (3, 3))
    pool4 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4)

    conv5 = ContractingPathBlock(pool4, 128, (3, 3))
    conv5 = layers.Dropout(0.5)(conv5)
    conv5 = ContractingPathBlock(conv5, 128, (3, 3))
    conv5 = layers.Dropout(0.5)(conv5)

    # Up-sampled and concatenate
    convt1 = ExpansivePathBlock(conv5, 64, (3, 3))
    concat1 = layers.concatenate([conv4, convt1], axis=3)
    concat1 = layers.Dropout(0.5)(concat1)
    conv6 = ContractingPathBlock(concat1, 64, (3, 3))
    conv6 = ContractingPathBlock(conv6, 64, (3, 3))

    convt2 = ExpansivePathBlock(conv6, 32, (3, 3))
    concat2 = layers.concatenate([conv3, convt2], axis=3)
    concat2 = layers.Dropout(0.5)(concat2)
    conv7 = ContractingPathBlock(concat2, 32, (3, 3))
    conv7 = ContractingPathBlock(conv7, 32, (3, 3))

    convt3 = ExpansivePathBlock(conv7, 16, (3, 3))
    concat3 = layers.concatenate([conv2, convt3], axis=3)
    concat3 = layers.Dropout(0.5)(concat3)
    conv8 = ContractingPathBlock(concat3, 16, (3, 3))
    conv8 = ContractingPathBlock(conv8, 16, (3, 3))

    convt4 = ExpansivePathBlock(conv8, 8, (3, 3))
    concat4 = layers.concatenate([conv1, convt4], axis=3)
    concat4 = layers.Dropout(0.5)(concat4)
    conv9 = ContractingPathBlock(concat4, 8, (3, 3))
    conv9 = ContractingPathBlock(conv9, 8, (3, 3))

    conv9 = layers.Dropout(0.5)(conv9)
    outpt = layers.Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv9)

    model = models.Model(input, outpt)
    sgd = tf.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    model.summary()

    print("Start training U-Net model...")
    model.fit(X_train, y_train, epochs=20, batch_size=16)  # Parameter settings
    model.save('../models/unet_green.h5')
    print('unet_greenModel10.h5 Saved successfully!!!')


def u_net_predict(u_net, img_path):
    img_origin = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)  #
    # img_src=cv2.imread(img_src_path)
    if img_origin.shape != (512, 512, 3):
        img_origin = cv2.resize(img_origin, dsize=(512, 512), interpolation=cv2.INTER_AREA)[:, :, :3]
    img_origin = img_origin.reshape(1, 512, 512, 3)  # The predicted image shape is (1,512,512,3)
    img_mask = u_net.predict(img_origin)  # normalization, divide by 255 to make a prediction
    img_origin = img_origin.reshape(512, 512, 3)  # Reshape the original image to 3 dimensions
    img_mask = img_mask.reshape(512, 512, 3)  # Reshape the predicted image to 3 dimensions
    img_mask = img_mask / np.max(img_mask) * 255  # Normalised and multiplied by 255
    img_mask[:, :, 2] = img_mask[:, :, 1] = img_mask[:, :, 0]  # All three channels remain the same
    img_mask = img_mask.astype(np.uint8)  # Convert img_mask type to int type

    return img_origin, img_mask


def main():
    """Set the parameters and call the functions"""
    u_net_train()  # trained to get unet.h5, which is used for license plate positioning


if __name__ == "__main__":
    """Execute the current module"""
    main()



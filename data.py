DATA_DIR = '/home/misiu/src/deep-learning/faces/data'

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

IMG_H = 96
IMG_W = 96


def load_train_data():
    the_csv = pd.read_csv(os.path.join(DATA_DIR, 'training.csv'))
    imgs = decode_images_from_csv(the_csv)
    ys = the_csv.iloc[:, :30].values
    return imgs, ys


def convert_to_img(string_list):
    return np.array(map(float, string_list.split(' '))).reshape(IMG_W, IMG_H)


def decode_images_from_csv(df):
    imgs = np.zeros(shape=(len(df), 1, IMG_H, IMG_W))

    raw_img_data = list(df.Image)

    for idx, data in enumerate(raw_img_data):
        img = convert_to_img(data)
        imgs[idx, 0] = img
        # plt.imshow(img, cmap=plt.get_cmap('gray'))
        # plt.show()

    return imgs


if __name__ == '__main__':
    imgs, ys = load_train_data()



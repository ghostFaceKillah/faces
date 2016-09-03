DATA_DIR = '/home/misiu/src/deep-learning/faces/data'

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools as it


IMG_H = 96
IMG_W = 96


REDUCED = [
    y + x for x,y in it.product(
        ['_x', '_y'], ['left_eye_center', 'right_eye_center', 'nose_tip', 'mouth_center_bottom_lip']
    )
]


REDUCED_POINTNAMES = [
    [x + '_x', x + '_y'] for x in
         ['left_eye_center', 'right_eye_center', 'nose_tip', 'mouth_center_bottom_lip']
]


def load_train_data(drop_na=True, reduced_dataset=True):
    the_csv = pd.read_csv(os.path.join(DATA_DIR, 'training.csv'))
    imgs = decode_images_from_csv(the_csv)
    ys = the_csv.iloc[:, :30]

    if reduced_dataset:
        ys = ys[REDUCED]
        # The whole point of the reduced dataset is to drop less nulls

    if drop_na:
        ys = ys[ys.notnull().all(axis=1)]
        imgs = imgs[list(ys.index)]

    for i in xrange(imgs.shape[0]):
        pass


    ys = ys.values
    return imgs, ys


def load_column_names(reduced=True):
    the_csv = pd.read_csv(os.path.join(DATA_DIR, 'training.csv'))

    ys = the_csv.iloc[:, :30]
    if reduced:
        ys = ys[REDUCED]

    return ys.columns


def load_test_data():
    the_csv = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    imgs = decode_images_from_csv(the_csv)

    return imgs


def convert_to_img(string_list):
    return np.array(map(float, string_list.split(' '))).reshape(IMG_W, IMG_H)


def decode_images_from_csv(df):
    imgs = np.zeros(shape=(len(df), 1, IMG_H, IMG_W))

    raw_img_data = list(df.Image)

    for idx, data in enumerate(raw_img_data):
        img = convert_to_img(data)
        imgs[idx, 0] = img

    return imgs


def plot_img(img, ys=None):
    plt.imshow(img[0], cmap=plt.get_cmap('gray'))

    if ys is not None:
        points = [list(ys[name_pair]) for name_pair in REDUCED_POINTNAMES]

        for x, y in points:
            plt.plot(x, y, marker='o', color='magenta')

    plt.show()


if __name__ == '__main__':
    imgs, ys = load_train_data()


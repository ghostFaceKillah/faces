from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

import tensorflow as tf

from data import IMG_H, IMG_W
import data


def build_network():
    inputs = Input((1, IMG_H, IMG_W))

    conv1 = Convolution2D(
        32, 3, 3, activation='relu', border_mode='same'
    )(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(
        64, 3, 3, activation='relu', border_mode='same'
    )(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(
        64, 3, 3, activation='relu', border_mode='same'
    )(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(
        128, 3, 3, activation='relu', border_mode='same'
    )(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(
        256, 3, 3, activation='relu', border_mode='same'
    )(pool4)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    flat = Flatten()(pool5)

    dense1 = Dense(2048, activation='relu')(flat)
    drop1 = Dropout(0.5)(dense1)

    dense2 = Dense(2048, activation='relu')(drop1)
    drop2 = Dropout(0.5)(dense2)

    dense3 = Dense(30, activation='relu')(drop2)

    model = Model(input=inputs, output=dense3)

    model.compile(
        optimizer=Adam(),
        loss='mse'
    )
    return model


def train():
    imgs, ys = data.load_train_data()
    model = build_network()

    model.fit(
        imgs, ys,
        validation_split=0.2
    )


if __name__ == '__main__':
    train()


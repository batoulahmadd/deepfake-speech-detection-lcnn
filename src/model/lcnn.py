from typing import List
import tensorflow as tf
from keras.layers import BatchNormalization, Dense, Dropout, Flatten, Input, MaxPool2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import regularizers
from keras.models import Model

from .layers import Maxout


def MaxOutConv2D(input: tf.Tensor, dim: int, kernel_size: int, strides: int, padding: str = "same") -> tf.Tensor:
    conv_out = Conv2D(
        dim,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        kernel_regularizer=regularizers.l2(1e-4)  # 🔧 L2 regularization
    )(input)
    mfm_out = Maxout(int(dim / 2))(conv_out)
    return mfm_out


def MaxOutDense(x: tf.Tensor, dim: int) -> tf.Tensor:
    dense_out = Dense(dim, kernel_regularizer=regularizers.l2(1e-4))(x)
    mfm_out = Maxout(int(dim / 2))(dense_out)
    return mfm_out


def build_lcnn(shape: List[int], n_label: int = 2) -> tf.keras.Model:
    input = Input(shape=shape)

    conv2d_1 = MaxOutConv2D(input, 64, kernel_size=5, strides=1, padding="same")
    maxpool_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv2d_1)

    conv_2d_2 = MaxOutConv2D(maxpool_1, 64, kernel_size=1, strides=1, padding="same")
    batch_norm_2 = BatchNormalization()(conv_2d_2)

    conv2d_3 = MaxOutConv2D(batch_norm_2, 96, kernel_size=3, strides=1, padding="same")
    maxpool_3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv2d_3)
    batch_norm_3 = BatchNormalization()(maxpool_3)

    conv_2d_4 = MaxOutConv2D(batch_norm_3, 96, kernel_size=1, strides=1, padding="same")
    batch_norm_4 = BatchNormalization()(conv_2d_4)

    conv2d_5 = MaxOutConv2D(batch_norm_4, 128, kernel_size=3, strides=1, padding="same")
    maxpool_5 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv2d_5)

    conv_2d_6 = MaxOutConv2D(maxpool_5, 128, kernel_size=1, strides=1, padding="same")
    batch_norm_6 = BatchNormalization()(conv_2d_6)

    conv_2d_7 = MaxOutConv2D(batch_norm_6, 64, kernel_size=3, strides=1, padding="same")
    batch_norm_7 = BatchNormalization()(conv_2d_7)

    conv_2d_8 = MaxOutConv2D(batch_norm_7, 64, kernel_size=1, strides=1, padding="same")
    batch_norm_8 = BatchNormalization()(conv_2d_8)

    conv_2d_9 = MaxOutConv2D(batch_norm_8, 64, kernel_size=3, strides=1, padding="same")
    maxpool_9 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv_2d_9)
    flatten = Flatten()(maxpool_9)

    # Add dropout before dense layers to reduce overfitting
    dropout_pre_dense = Dropout(0.5)(flatten)

    dense_10 = MaxOutDense(dropout_pre_dense, 160)
    batch_norm_10 = BatchNormalization()(dense_10)
    dropout_10 = Dropout(0.75)(batch_norm_10)

    output = Dense(n_label, activation="softmax")(dropout_10)

    return Model(inputs=input, outputs=output)

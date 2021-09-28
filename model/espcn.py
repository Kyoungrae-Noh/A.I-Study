from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda
from tensorflow.python.keras.models import Model

from model.common import normalize, denormalize, pixel_shuffle

import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
import time
import os


def espcn(scale, num_filters=32):
    x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize)(x_in)

    x = Conv2D(filters=num_filters*2, kernel_size=5, padding='same', activation='relu')(x)
    x = Conv2D(filters=num_filters, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(filters=3*scale*scale, kernel_size=3, padding='same')(x)

    x = upsample(x, scale)

    x = Lambda(denormalize)(x)
    # x = tf.keras.activations.tanh(x)

    return Model(x_in, x, name="espcn")

def upsample(x, scale):
    def upsample_1(x, factor, **kwargs):
        # x = Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
        x = Conv2D(3 * (factor ** 2), 3, padding='same', **kwargs)(x)
        return Lambda(pixel_shuffle(scale=factor))(x)

    if scale == 2:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
    elif scale == 3:
        x = upsample_1(x, 3, name='conv2d_1_scale_3')
    elif scale == 4:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
        x = upsample_1(x, 2, name='conv2d_2_scale_2')

    return x


# def edsr(scale, num_filters=64, num_res_blocks=8, res_block_scaling=None):
#     x_in = Input(shape=(None, None, 3))
#     x = Lambda(normalize)(x_in)
#
#     x = b = Conv2D(num_filters, 3, padding='same')(x)
#     for i in range(num_res_blocks):
#         b = res_block(b, num_filters, res_block_scaling)
#     b = Conv2D(num_filters, 3, padding='same')(b)
#     x = Add()([x, b])
#
#     x = upsample(x, scale, num_filters)
#     x = Conv2D(3, 3, padding='same')(x)
#
#     x = Lambda(denormalize)(x)
#     return Model(x_in, x, name="edsr")
#
#
# def res_block(x_in, filters, scaling):
#     x = Conv2D(filters, 3, padding='same', activation='relu')(x_in)
#     x = Conv2D(filters, 3, padding='same')(x)
#     if scaling:
#         x = Lambda(lambda t: t * scaling)(x)
#     x = Add()([x_in, x])
#     return x




# Basic Code is taken from https://github.com/ckmarkoh/GAN-tensorflow

import tensorflow as tf
import numpy as np
from scipy.misc import imsave
import os
import shutil
from PIL import Image
import time
import random

from layers import general_conv2d, general_deconv2d

"""
For image size 256*256
img_height = 256
img_width = 256
img_layer = 3
img_size = img_height * img_width


batch_size = 1
pool_size = 50
generator_first_layer_filters = 32
discriminator_first_layer_filters = 64
"""

# update the variables
img_height = 32
img_width = 32
img_layer = 3
img_size = img_height * img_width

batch_size = 1
generator_first_layer_filters = 8
discriminator_first_layer_filters = 16


def build_resnet_block(inputres, dim, name="resnet"):
    with tf.variable_scope(name):

        out_res = tf.pad(inputres, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")

        out_res = general_conv2d(out_res, dim,
                                 kernel=3, stride=1, stddev=0.02,
                                 padding="VALID", name="c1")

        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")

        out_res = general_conv2d(out_res, dim,
                                 kernel=3, stride=1, stddev=0.02,
                                 padding="VALID", name="c2", do_relu=False)

        return tf.nn.relu(out_res + inputres)

def build_generator_resnet_1block(inputgen, name="generator"):
    with tf.variable_scope(name):
        f_stride = 5
        ks = 2
        stride = 1

        # This pads the inputgen with 0
        # inputgen is 4D, so the padding arguement needs 4 dimensions
        # inputgen = [batch_size, img_width, img_height, img_layerA]
        # [0, 0] : pad this dimension of input with no padding before, or after
        # [ks, ks]: pad this dimesnion of input with ks values before and after
        pad_input = tf.pad(inputgen, [[0, 0], [ks, ks], [ks, ks], [0, 0]],
                           "REFLECT")

        o_c1 = general_conv2d(pad_input, generator_first_layer_filters,
                              kernel=f_stride, stride=1, stddev=0.02,
                              name="c1")

        o_c2 = general_conv2d(o_c1, generator_first_layer_filters*2, kernel=ks,
                              stride=stride, stddev=0.02,
                              padding="SAME", name="c2")

        o_r1 = build_resnet_block(o_c2, generator_first_layer_filters*2, "r1")

        o_c4 = general_deconv2d(o_r1, generator_first_layer_filters, kernel=ks,
                                stride=stride, stddev=0.02, padding="SAME",
                                name="c4")

        o_c4_pad = tf.pad(o_c4, [[0, 0], [ks, ks],
                          [ks, ks], [0, 0]], "REFLECT")

        o_c6 = general_conv2d(o_c4_pad, img_layer, kernel=f_stride,
                              stride=1, stddev=0.02, padding="VALID",
                              name="c6", do_relu=False)

        # Adding the tanh layer

        out_gen = tf.nn.tanh(o_c6, "t1")

        return out_gen


# This builds a generator with 2 resnet blocks
def build_generator_resnet_2blocks(inputgen, name="generator"):
    with tf.variable_scope(name):
        f_stride = 5
        ks = 2
        stride = 1

        # This pads the inputgen with 0
        # inputgen is 4D, so the padding arguement needs 4 dimensions
        # inputgen = [batch_size, img_width, img_height, img_layerA]
        # [0, 0] : pad this dimension of input with no padding before, or after
        # [ks, ks]: pad this dimesnion of input with ks values before and after
        pad_input = tf.pad(inputgen, [[0, 0], [ks, ks], [ks, ks], [0, 0]],
                           "REFLECT")

        # Encoding
        o_c1 = general_conv2d(pad_input, generator_first_layer_filters,
                              kernel=f_stride, stride=1, stddev=0.02,
                              name="c1")
        # o_c1.shape = (28, 28, 8)

        o_c2 = general_conv2d(o_c1, generator_first_layer_filters*2, kernel=ks,
                              stride=stride, stddev=0.02,
                              padding="SAME", name="c2")
        # o_c2.shape = (28, 28, 16)

        # Transformation
        o_r1 = build_resnet_block(o_c2, generator_first_layer_filters*2, "r1")
        o_r2 = build_resnet_block(o_r1, generator_first_layer_filters*2, "r2")
        # o_r2.shape = (28, 28, 16)

        # Decoding
        o_c4 = general_deconv2d(o_r2, generator_first_layer_filters, kernel=ks,
                                stride=stride, stddev=0.02, padding="SAME",
                                name="c4")

        o_c4_pad = tf.pad(o_c4, [[0, 0], [ks, ks],
                          [ks, ks], [0, 0]], "REFLECT")

        o_c6 = general_conv2d(o_c4_pad, img_layer, kernel=f_stride,
                              stride=1, stddev=0.02, padding="VALID",
                              name="c6", do_relu=False)

        # Adding the tanh layer

        out_gen = tf.nn.tanh(o_c6, "t1")

        return out_gen


def build_gen_discriminator(inputdisc, name="discriminator"):

    with tf.variable_scope(name):
        f = 4

        o_c1 = general_conv2d(inputdisc, discriminator_first_layer_filters,
                              kernel=f, stride=2, stddev=0.02,
                              padding="SAME", name="c1",
                              do_norm=False, lrelu_slope=0.2)

        o_c2 = general_conv2d(o_c1, discriminator_first_layer_filters*2,
                              kernel=f, stride=2, stddev=0.02,
                              padding="SAME", name="c2", lrelu_slope=0.2)

        o_c3 = general_conv2d(o_c2, discriminator_first_layer_filters*4,
                              kernel=f, stride=2, stddev=0.02,
                              padding="SAME", name="c3", lrelu_slope=0.2)

        # o_c4 = general_conv2d(o_c3, discriminator_first_layer_filters*8,
        #                       kernel=f, stride=1, stddev=0.02,
        #                       padding="SAME", name="c4",
        #                       lrelu_slope=0.2)

        o_c5 = general_conv2d(o_c3, 1, kernel=f, stride=1, stddev=0.02,
                              padding="SAME", name="c5",
                              do_norm=False, do_relu=False)

        return o_c5


'''
def patch_discriminator(inputdisc, name="discriminator"):

    with tf.variable_scope(name):
        f= 4

        patch_input = tf.random_crop(inputdisc,[1,70,70,3])
        o_c1 = general_conv2d(patch_input, discriminator_first_layer_filters, f, f, 2, 2, 0.02, "SAME", "c1", do_norm="False", lrelu_slope=0.2)
        o_c2 = general_conv2d(o_c1, discriminator_first_layer_filters*2, f, f, 2, 2, 0.02, "SAME", "c2", lrelu_slope=0.2)
        o_c3 = general_conv2d(o_c2, discriminator_first_layer_filters*4, f, f, 2, 2, 0.02, "SAME", "c3", lrelu_slope=0.2)
        o_c4 = general_conv2d(o_c3, discriminator_first_layer_filters*8, f, f, 2, 2, 0.02, "SAME", "c4", lrelu_slope=0.2)
        o_c5 = general_conv2d(o_c4, 1, f, f, 1, 1, 0.02, "SAME", "c5",do_norm=False,do_relu=False)

        return o_c5
        '''
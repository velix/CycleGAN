# Basic Code is taken from https://github.com/ckmarkoh/GAN-tensorflow

import tensorflow as tf
import numpy as np
import os
import shutil
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
generator_first_layer_filters = 64
discriminator_first_layer_filters = 64


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

# This builds a generator with 2 resnet blocks
def build_generator_resnet_2blocks(inputgen, name="generator"):
    with tf.variable_scope(name):
        big_kernel = 5
        small_kernel = 3
        stride = 2
        reflection_padding = 2

        print('GENERATOR')
        print("inputgen shape: ", inputgen.get_shape())

        # This pads the inputgen with 0
        # inputgen is 4D, so the padding arguement needs 4 dimensions
        # inputgen = [batch_size, img_width, img_height, img_layerA]
        # [0, 0] : pad this dimension of input with no padding before, or after
        # [small_kernel, small_kernel]: pad this dimesnion of input with small_kernel values before and after
        pad_input = tf.pad(inputgen,
                           [[0, 0], [reflection_padding, reflection_padding],
                           [reflection_padding, reflection_padding], [0, 0]],
                           "REFLECT")

        print("pad_input shape: ", pad_input.get_shape())

        # Encoding
        c7s1_c1 = general_conv2d(pad_input, generator_first_layer_filters,
                              kernel=big_kernel,
                              stride=1, 
                              stddev=0.02,
                              name="c7s1_c1")
        print("c7s1_c1 shape: ", c7s1_c1.get_shape())


        d_c2 = general_conv2d(c7s1_c1, generator_first_layer_filters*2,
                              kernel=small_kernel,
                              stride=stride,
                              stddev=0.02,
                              padding="SAME", name="d_c2")
        print("d_c2 shape: ", d_c2.get_shape())

        # Transformation
        o_r1 = build_resnet_block(d_c2, generator_first_layer_filters*2, "r1")
        print("o_r1 shape: ", o_r1.get_shape())
        o_r2 = build_resnet_block(o_r1, generator_first_layer_filters*2, "r2")
        print("o_r2 shape: ", o_r2.get_shape())

        # Decoding
        u_c4 = general_deconv2d(o_r2, generator_first_layer_filters,
                                kernel=small_kernel,
                                stride=stride, 
                                stddev=0.02,
                                padding="SAME", name="u_c4")
        print("u_c4 shape: ", u_c4.get_shape())

        u_c5 = general_deconv2d(u_c4, generator_first_layer_filters,
                                kernel=small_kernel,
                                stride=stride, 
                                stddev=0.02,
                                padding="SAME", name="u_c5")
        print("u_c5 shape: ", u_c5.get_shape())

        u_c5_pad = tf.pad(u_c5,
                          [[0, 0], [reflection_padding, reflection_padding],
                          [reflection_padding, reflection_padding], [0, 0]],
                          "REFLECT")
        print("u_c5_pad pad shape: ", u_c5_pad.get_shape())

        c7s1_c6 = general_conv2d(u_c5_pad, img_layer,
                              kernel=big_kernel,
                              stride=2,
                              stddev=0.02,
                              padding="VALID", name="c7s1_c6", do_relu=False)
        print("c7s1_c6 shape: ", c7s1_c6.get_shape())
        print()

        # Adding the tanh layer
        out_gen = tf.nn.tanh(c7s1_c6, "t1")

        return out_gen

def build_gen_discriminator(inputdisc, name="discriminator"):

    with tf.variable_scope(name):
        kernel = 4

        print('DISCRIMINATOR')
        print('inputdisc: ', inputdisc.get_shape())

        patch_input = tf.random_crop(inputdisc,[1, 9, 9, 3])
        print('patch input: ', patch_input.get_shape())

        o_c1 = general_conv2d(patch_input, discriminator_first_layer_filters,
                              kernel=kernel, stride=2, stddev=0.02,
                              padding="SAME", name="c1",
                              do_norm=False, lrelu_slope=0.2)
        print('o_c1: ', o_c1.get_shape())


        o_c2 = general_conv2d(o_c1, discriminator_first_layer_filters*2,
                              kernel=kernel, stride=2, stddev=0.02,
                              padding="SAME", name="c2",
                              lrelu_slope=0.2)
        print('o_c2: ', o_c2.get_shape())

        # o_c3 = general_conv2d(o_c2, discriminator_first_layer_filters*4,
        #                       kernel=kernel, stride=2, stddev=0.02,
        #                       padding="SAME", name="c3",
        #                       lrelu_slope=0.2)
        # print('o_c3: ', o_c3.get_shape())

        # o_c4 = general_conv2d(o_c3, discriminator_first_layer_filters*8,
        #                       kernel=kernel, stride=2, stddev=0.02,
        #                       padding="SAME", name="c4",
        #                       lrelu_slope=0.2)
        # print('o_c4: ', o_c4.get_shape())

        o_c5 = general_conv2d(o_c2, 1,
                              kernel=kernel, stride=1, stddev=0.02,
                              padding="SAME", name="c5",
                              do_norm=False, do_relu=False)
        print('o_c5: ', o_c5.get_shape())
        print()
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

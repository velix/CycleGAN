# Basic Code is taken from https://github.com/ckmarkoh/GAN-tensorflow

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import scipy.misc.imsave as imsave
import os
import shutil
from PIL import Image
import time
import random

from layers import *

img_height = 256
img_width = 256
img_layer = 3
img_size = img_height * img_width

batch_size = 1
pool_size = 50
ngf = 32
ndf = 64


def build_resnet_block(inputres, dim, name="resnet"):
    with tf.variable_scope(name):
        out_res = tf.pad(inputres, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = general_conv2d(out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c1")
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = general_conv2d(out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c2", do_relu=False)

        return tf.nn.relu(out_res + inputres)


'''numberOfGEN: the number of Cnn:s in the Encoder
numberOfres: is the number of resNets
numberOfDec: is the number of deconvolutional layers in the decoder
reflection = 'true': sets the reflaction padding as default but you can choose to to set it false
imageDim: is the image dimension which is used on the decoding part'''

def BuildGeneratorAndResNet(inputgen,numberOfGEN, numberOfres,numberOfDec,imageDim,reflection = 'true' , name="generator"):
    with tf.variable_scope(name):
        f = 7
        ks = 3

        '''These variables are used in the docder where dimfromRes is the output
         from the resNet into the decoder
         ordDim is the ordinary dimension'''
        dimFromRes = [batch_size, imageDim/2, imageDim/2, ngf * 2]
        ordDim = [batch_size, imageDim, imageDim, ngf]

        '''Creates Encoder
        This part creates as many cnn:s as the numberOfGEN in the input of the function'''
        pad_input = tf.pad(inputgen, [[0, 0], [ks, ks], [ks, ks], [0, 0]], "REFLECT")
        for i in range(numberOfGEN):
            '''This if statement initializes the first cnn and adds it to the second
             and when i is larger then 1 it multiplies ngf with 2 for each new layer and changes the name'''
            if i == 0:
                outputG = general_conv2d(pad_input, ngf, f, f, 1, 1, 0.02, name="c"+str(i+1))
            else:
                outputG = general_conv2d(outputG, ngf * (i*2), ks, ks, 2, 2, 0.02, "SAME", "c" + str(i+1))

        '''Creates ResNet
        This for loop creates as many resnets as numberOfres'''
        for r in range(1,numberOfres+1):
            ''' This if statement initializes the first resnet and sends it in to
             the second and so on'''
            if r == 1:
                outputR = build_resnet_block(outputG, ngf * 4, "r"+str(r))
            else:
                outputR = build_resnet_block(outputR, ngf * 4, "r"+str(r))

        '''Creates Decoder
        Creates as many deconvolutional layers as the numberOfDec'''
        for k in range(1,numberOfDec+1):
        ''' this if statement initializes the first deconvolutional layer that
             takes the input from the resnets'''
            if k ==1:
                outputD = general_deconv2d(outputR, dimFromRes, ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "c"+str(i+k))
            else:
                 '''The layers between the first and the second last has zero padding(else statement)
                 while the second last has a reflection padding for the code with 6 resnets
                 we can basically choose to set the the reflection padding to false if we don't want one
                 The last deconvolutional layer doesn't have any padding and represents the elif statement'''
                if k == numberOfDec-1 and reflection == 'true':
                    outputD = tf.pad(outputD, [[0, 0], [ks, ks], [ks, ks], [0, 0]], "REFLECT")
                elif k == numberOfDec:
                    outputD = general_conv2d(outputD, img_layer, f, f, 1, 1, 0.02, "VALID", "c"+str(i+k), do_relu=False)
                else:
                    outputD = general_deconv2d(outputD, ordDim, ngf, ks, ks, 2, 2, 0.02, "SAME", "c"+str(i+k))

        # Adding the tanh layer

        outGenerator = tf.nn.tanh(outputD, "t1")

        return outGenerator


''' This function creates the discriminator
    We can choose to use a patch discrimiantor or an ordinary discrimiantor
    by setting patch to false or true respectively'''
def buildDiscriminator(inputdisc,numberOfDisc, patch = 'false',name="discriminator"):
    with tf.variable_scope(name):
        f = 4
        ''' This if statement initializes the first convolutional layer depending on whether
            the patch is true or false, if it is true then we use crop which
            randomly crops a tensor to a given size ([1,70,70,3] (elif statement))
            If patch is false then we use a normal convolutional layer (if statement),
            the ndf for the layer increases with two between the first and the second layer (else)
            The first and the last layer doesn't have relu or norm'''

        for i in range(numberOfDisc):
            if i == 0 and patch == 'false':
                outputdisc = general_conv2d(inputdisc, ndf, f, f, 2, 2, 0.02, "SAME", "c"+str(i+1), do_norm=False, relufactor=0.2)
            elif i == 0 and patch == 'true':
                outputdisc = tf.random_crop(inputdisc,[1,70,70,3])
            elif i == numberOfDisc:
                outputdisc = general_conv2d(outputdisc, 1, f, f, 1, 1, 0.02, "SAME", "c"+str(i+1), do_norm=False, do_relu=False)
            else:
                outputdisc = general_conv2d(outputdisc, ndf *(i*2), f, f, 2, 2, 0.02, "SAME", "c"+str(i+1), relufactor=0.2)

        return outputdisc

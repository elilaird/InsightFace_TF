#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
VGG-19 for ImageNet.
Pre-trained model in this example - VGG19 NPZ and
trainable examples of VGG16/19 in TensorFlow can be found here:
https://github.com/machrisaa/tensorflow-vgg
For simplified CNN layer see "Convolutional layer (Simplified)"
in read the docs website.
"""

import os
import time

import numpy as np
import skimage
import skimage.io
import skimage.transform
import tensorflow as tf
from scipy.misc import imread, imresize
import tensorlayer as tl
from tensorlayer.layers import *
from imagenet_classes import *


DATA_PATH = '/home/aurora/workspaces2/PycharmProjects/tensorflow/tensorlayer/example/data'
VGG_MEAN = [103.939, 116.779, 123.68]


def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    if ((0 <= img).all() and (img <= 1.0).all()) is False:
        raise Exception("image value should be [0, 1]")
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img


def print_prob(prob):
    synset = class_names
    # print prob
    pred = np.argsort(prob)[::-1]
    # Get top1 label
    top1 = synset[pred[0]]
    print("Top1: ", top1, prob[pred[0]])
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print("Top5: ", top5)
    return top1


def _Vgg19(rgb):
    """
    Build the VGG 19 Model
    Parameters
    -----------
    rgb : rgb image placeholder [batch, height, width, 3] values scaled [0, 1]
    """
    start_time = time.time()
    print("build model started")
    rgb_scaled = rgb * 255.0
    # Convert RGB to BGR
    if tf.__version__ <= '0.11':
        red, green, blue = tf.split(3, 3, rgb_scaled)
    else:  # TF 1.0
        print(rgb_scaled)
        red, green, blue = tf.split(rgb_scaled, 3, 3)
    if red.get_shape().as_list()[1:] != [224, 224, 1]:
        raise Exception("image size unmatch")
    if green.get_shape().as_list()[1:] != [224, 224, 1]:
        raise Exception("image size unmatch")
    if blue.get_shape().as_list()[1:] != [224, 224, 1]:
        raise Exception("image size unmatch")
    if tf.__version__ <= '0.11':
        bgr = tf.concat(3, [
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
    else:
        bgr = tf.concat(
            [
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ], axis=3)
    if bgr.get_shape().as_list()[1:] != [224, 224, 3]:
        raise Exception("image size unmatch")
    # input layer
    net_in = InputLayer(bgr, name='input')
    # conv1
    network = Conv2dLayer(net_in, act=tf.nn.relu, shape=[3, 3, 3, 64], strides=[1, 1, 1, 1], padding='SAME', name='conv1_1')
    network = Conv2dLayer(network, act=tf.nn.relu, shape=[3, 3, 64, 64], strides=[1, 1, 1, 1], padding='SAME', name='conv1_2')
    network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool2d, name='pool1')
    # conv2
    network = Conv2dLayer(network, act=tf.nn.relu, shape=[3, 3, 64, 128], strides=[1, 1, 1, 1], padding='SAME', name='conv2_1')
    network = Conv2dLayer(network, act=tf.nn.relu, shape=[3, 3, 128, 128], strides=[1, 1, 1, 1], padding='SAME', name='conv2_2')
    network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool2d, name='pool2')
    # conv3
    network = Conv2dLayer(network, act=tf.nn.relu, shape=[3, 3, 128, 256], strides=[1, 1, 1, 1], padding='SAME', name='conv3_1')
    network = Conv2dLayer(network, act=tf.nn.relu, shape=[3, 3, 256, 256], strides=[1, 1, 1, 1], padding='SAME', name='conv3_2')
    network = Conv2dLayer(network, act=tf.nn.relu, shape=[3, 3, 256, 256], strides=[1, 1, 1, 1], padding='SAME', name='conv3_3')
    network = Conv2dLayer(network, act=tf.nn.relu, shape=[3, 3, 256, 256], strides=[1, 1, 1, 1], padding='SAME', name='conv3_4')
    network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool2d, name='pool3')
    # conv4
    network = Conv2dLayer(network, act=tf.nn.relu, shape=[3, 3, 256, 512], strides=[1, 1, 1, 1], padding='SAME', name='conv4_1')
    network = Conv2dLayer(network, act=tf.nn.relu, shape=[3, 3, 512, 512], strides=[1, 1, 1, 1], padding='SAME', name='conv4_2')
    network = Conv2dLayer(network, act=tf.nn.relu, shape=[3, 3, 512, 512], strides=[1, 1, 1, 1], padding='SAME', name='conv4_3')
    network = Conv2dLayer(network, act=tf.nn.relu, shape=[3, 3, 512, 512], strides=[1, 1, 1, 1], padding='SAME', name='conv4_4')
    network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool2d, name='pool4')
    # conv5
    network = Conv2dLayer(network, act=tf.nn.relu, shape=[3, 3, 512, 512], strides=[1, 1, 1, 1], padding='SAME', name='conv5_1')
    network = Conv2dLayer(network, act=tf.nn.relu, shape=[3, 3, 512, 512], strides=[1, 1, 1, 1], padding='SAME', name='conv5_2')
    network = Conv2dLayer(network, act=tf.nn.relu, shape=[3, 3, 512, 512], strides=[1, 1, 1, 1], padding='SAME', name='conv5_3')
    network = Conv2dLayer(network, act=tf.nn.relu, shape=[3, 3, 512, 512], strides=[1, 1, 1, 1], padding='SAME', name='conv5_4')
    network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool2d, name='pool5')
    # fc 6~8
    network = FlattenLayer(network, name='flatten')
    network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc6')
    network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc7')
    network = DenseLayer(network, n_units=1000, act=tf.identity, name='fc8')
    print("build model finished: %fs" % (time.time() - start_time))
    return network


def _Vgg19_simple_api(rgb):
    """
    Build the VGG 19 Model
    Parameters
    -----------
    rgb : rgb image placeholder [batch, height, width, 3] values scaled [0, 1]
    """
    start_time = time.time()
    print("build model started")
    rgb_scaled = rgb * 255.0
    # Convert RGB to BGR
    if tf.__version__ <= '0.11':
        red, green, blue = tf.split(3, 3, rgb_scaled)
    else:  # TF 1.0
        print(rgb_scaled)
        red, green, blue = tf.split(rgb_scaled, 3, 3)
    if red.get_shape().as_list()[1:] != [224, 224, 1]:
        raise Exception("image size unmatch")
    if green.get_shape().as_list()[1:] != [224, 224, 1]:
        raise Exception("image size unmatch")
    if blue.get_shape().as_list()[1:] != [224, 224, 1]:
        raise Exception("image size unmatch")
    if tf.__version__ <= '0.11':
        bgr = tf.concat(3, [
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
    else:
        bgr = tf.concat(
            [
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ], axis=3)
    if bgr.get_shape().as_list()[1:] != [224, 224, 3]:
        raise Exception("image size unmatch")
    # input layer
    net_in = InputLayer(bgr, name='input')
    # conv1
    network = Conv2d(net_in, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_1')
    network = Conv2d(network, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_2')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')
    # conv2
    network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_1')
    network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_2')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')
    # conv3
    network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_1')
    network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_2')
    network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_3')
    network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_4')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3')
    # conv4
    network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_1')
    network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_2')
    network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_3')
    network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_4')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool4')
    # conv5
    network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_1')
    network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_2')
    network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_3')
    network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_4')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool5')
    # fc 6~8
    network = FlattenLayer(network, name='flatten')
    network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc6')
    network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc7')
    network = DenseLayer(network, n_units=1000, act=tf.identity, name='fc8')
    print("build model finished: %fs" % (time.time() - start_time))
    return network


def get_vgg19(inputs, sess=None, pretrained=True):
    network = _Vgg19(inputs)
    if pretrained:
        vgg19_npy_path = "../model_weights/vgg19.npy"
        npz = np.load(vgg19_npy_path, encoding='latin1').item()
        params = []
        for val in sorted(npz.items()):
            W = np.asarray(val[1][0])
            b = np.asarray(val[1][1])
            print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
            params.extend([W, b])
        print("Restoring model from npz file")
        tl.files.assign_params(sess, params, network)
        return network
    else:
        tl.layers.initialize_global_variables(sess)
        return network


if __name__ == '__main__':
    sess = tf.compat.v1.InteractiveSession()
    x = tf.compat.v1.placeholder("float", [None, 224, 224, 3])
    network = get_vgg19(x, sess)
    y = network.outputs
    probs = tf.nn.softmax(y, name="prob")
    img1 = load_image(os.path.join(DATA_PATH, "tiger.jpeg"))  # test data in github
    img1 = img1.reshape((1, 224, 224, 3))
    start_time = time.time()
    prob = sess.run(probs, feed_dict={x: img1})
    print("End time : %.5ss" % (time.time() - start_time))

    print_prob(prob[0])
"""
Madry model from:
https://github.com/MadryLab/mnist_challenge/blob/master/model.py

Modified to have the signature expected by the tensorflow implementation
of the Carlini's L2 attack

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Model(object):
    def __init__(self):
        self.image_size = 28
        self.num_channels = 1
        self.num_labels = 10

        # first convolutional layer
        self.W_conv1 = self._weight_variable([5,5,1,32])
        self.b_conv1 = self._bias_variable([32])

        # second convolutional layer
        self.W_conv2 = self._weight_variable([5,5,32,64])
        self.b_conv2 = self._bias_variable([64])

        # first fully connected layer
        self.W_fc1 = self._weight_variable([7 * 7 * 64, 1024])
        self.b_fc1 = self._bias_variable([1024])

        # output layer
        self.W_fc2 = self._weight_variable([1024,10])
        self.b_fc2 = self._bias_variable([10])

    def predict(self, input: tf.placeholder):
        h_conv1 = tf.nn.relu(self._conv2d(input, self.W_conv1) + self.b_conv1)
        h_pool1 = self._max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(self._conv2d(h_pool1, self.W_conv2) + self.b_conv2)
        h_pool2 = self._max_pool_2x2(h_conv2)

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W_fc1) + self.b_fc1)

        pre_softmax = tf.matmul(h_fc1, self.W_fc2) + self.b_fc2
        return pre_softmax


    @staticmethod
    def _weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def _bias_variable(shape):
        initial = tf.constant(0.1, shape = shape)
        return tf.Variable(initial)

    @staticmethod
    def _conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

    @staticmethod
    def _max_pool_2x2( x):
        return tf.nn.max_pool(x,
                              ksize = [1,2,2,1],
                              strides=[1,2,2,1],
                              padding='SAME')

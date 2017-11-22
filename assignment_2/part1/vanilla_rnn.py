# MIT License
# 
# Copyright (c) 2017 Tom Runia
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2017-10-19

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

################################################################################

class VanillaRNN(object):

    def __init__(self, input_length, input_dim, num_hidden, num_classes, batch_size):

        self._input_length = input_length
        self._input_dim    = input_dim
        self._num_hidden   = num_hidden
        self._num_classes  = num_classes
        self._batch_size   = batch_size

        initializer_weights = tf.variance_scaling_initializer()
        #initializer_biases  = tf.constant_initializer(0.0)

        # Initialize the stuff you need
        # ...        
        self.W_hx = tf.Variable(initializer_weights(shape = [input_dim, num_hidden]))
        self.W_hh = tf.Variable(initializer_weights(shape = [num_hidden, num_hidden]))
        self.b_h =  tf.Variable(tf.zeros([num_hidden]))
        
        self.W_ho = tf.Variable(initializer_weights(shape = [num_hidden, num_classes]))
        self.b_o = tf.Variable(tf.zeros([num_classes]))

    def _rnn_step(self, h_prev, x):
        # Single step through Vanilla RNN cell ...
        return tf.nn.tanh(tf.matmul(h_prev, self.W_hh) + tf.matmul( x, self.W_hx) + self.b_h)

    def compute_logits(self, x):
        # Implement the logits for predicting the last digit in the palindrome
        h_T = tf.scan(self._rnn_step, tf.transpose(x, [1, 0, 2]),
                      initializer = tf.zeros(shape=[self._batch_size, self._num_hidden]))[-1]
        logits = tf.matmul(h_T, self.W_ho) + self.b_o
        return logits

    def compute_loss(self, logits, labels):
        # Implement the cross-entropy loss for classification of the last digit
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        return loss

    def accuracy(self, logits, labels):
        # Implement the accuracy of predicting the
        # last digit over the current batch ...
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1)), tf.float32))
        return accuracy

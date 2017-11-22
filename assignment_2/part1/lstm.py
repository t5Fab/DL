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

import tensorflow as tf


class LSTM(object):

    def __init__(self, input_length, input_dim, num_hidden, num_classes, batch_size):

        self._input_length = input_length
        self._input_dim    = input_dim
        self._num_hidden   = num_hidden
        self._num_classes  = num_classes
        self._batch_size   = batch_size

        initializer_weights = tf.variance_scaling_initializer()
        initializer_biases  = tf.constant_initializer(0.0)

        # Initialize the stuff you need
        # ...
        #g
        self.W_gx = tf.Variable(initializer_weights(shape = [input_dim, num_hidden]))
        self.W_gh = tf.Variable(initializer_weights(shape = [num_hidden, num_hidden]))
        self.b_g =  tf.Variable(tf.zeros([num_hidden]))
        
        #i
        self.W_ix = tf.Variable(initializer_weights(shape = [input_dim, num_hidden]))
        self.W_ih = tf.Variable(initializer_weights(shape = [num_hidden, num_hidden]))
        self.b_i =  tf.Variable(tf.zeros([num_hidden]))
        #f
        self.W_fx = tf.Variable(initializer_weights(shape = [input_dim, num_hidden]))
        self.W_fh = tf.Variable(initializer_weights(shape = [num_hidden, num_hidden]))
        self.b_f =  tf.Variable(tf.zeros([num_hidden]))
        #o
        self.W_ox = tf.Variable(initializer_weights(shape = [input_dim, num_hidden]))
        self.W_oh = tf.Variable(initializer_weights(shape = [num_hidden, num_hidden]))
        self.b_o =  tf.Variable(tf.zeros([num_hidden]))
        #out
        self.W_out = tf.Variable(initializer_weights(shape = [num_hidden, num_classes]))
        self.b_out = tf.Variable(tf.zeros([num_classes]))

        

    def _lstm_step(self, lstm_state_tuple, x):
        # Single step through LSTM cell ...
        c, h = tf.unstack(lstm_state_tuple)
        
        g = tf.nn.tanh(tf.matmul(h, self.W_gh) + tf.matmul( x, self.W_gx) + self.b_g)
        i = tf.nn.sigmoid(tf.matmul(h, self.W_ih) + tf.matmul( x, self.W_ix) + self.b_i)
        f = tf.nn.sigmoid(tf.matmul(h, self.W_fh) + tf.matmul( x, self.W_fx) + self.b_f)
        o = tf.nn.sigmoid(tf.matmul(h, self.W_oh) + tf.matmul( x, self.W_ox) + self.b_o)
        c = tf.multiply(g, i) + tf.multiply(c, f)
        h = tf.multiply(tf.nn.tanh(c), o)
        return tf.stack([c, h])

    def compute_logits(self, x):
        # Implement the logits for predicting the last digit in the palindrome
        last = tf.scan(self._lstm_step, tf.transpose(x, [1, 0, 2]),
                      initializer = tf.zeros(shape=[2, self._batch_size, self._num_hidden]))[-1]
        _, h_T = tf.unstack(last)
        logits = tf.matmul(h_T, self.W_out) + self.b_out
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
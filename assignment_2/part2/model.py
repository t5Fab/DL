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
import numpy as np

class TextGenerationModel(object):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden, lstm_num_layers):

        self._seq_length = seq_length
        self._lstm_num_hidden = lstm_num_hidden
        self._lstm_num_layers = lstm_num_layers
        self._batch_size = batch_size
        self._vocab_size = vocabulary_size

        # Initialization:
        # ...
        

    def _build_model(self, x, init_states):
        # Implement your model to return the logits per step of shape:
        #   [timesteps, batch_size, vocab_size]

        states = []
        feed = tf.one_hot(indices = x, depth = self._vocab_size)
        for i, prev_state in enumerate(tf.split(init_states, num_or_size_splits = self._lstm_num_layers, axis = 0)):
            c, h = tf.split(prev_state, num_or_size_splits = 2, axis = 1)
            prev_state = tf.nn.rnn_cell.LSTMStateTuple(tf.squeeze(c), tf.squeeze(h))
            feed, state = tf.nn.dynamic_rnn(cell=tf.nn.rnn_cell.LSTMCell(num_units=self._lstm_num_hidden),
											initial_state = prev_state,
                                            dtype=tf.float32, inputs=feed, scope = 'LSTM'+str(i))
            states.append(state[-1])
        logits_per_step = tf.layers.dense(feed, self._vocab_size)
        return logits_per_step, tf.stack(states)


    def _compute_loss(self, logits, labels):
        # Cross-entropy loss, averaged over timestep and batch
        loss = tf.losses.softmax_cross_entropy(labels, logits)
        return loss

    def probabilities(self, logits):
        # Returns the normalized per-step probabilities
        probabilities = tf.nn.softmax(logits)
        return probabilities

    def predictions(self, logits):
        # Returns the per-step predictions
        predictions = tf.argmax(logits, axis = -1)
        return predictions
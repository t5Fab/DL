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
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

import utils
from vanilla_rnn import VanillaRNN
from lstm import LSTM

################################################################################

def train(config):

    assert config.model_type in ('RNN', 'LSTM')

    # Setup the model that we are going to use
    if config.model_type == 'RNN':
        print("Initializing Vanilla RNN model...")
        model = VanillaRNN(
            config.input_length, config.input_dim, config.num_hidden,
            config.num_classes, config.batch_size
        )
    else:
        print("Initializing LSTM model...")
        model = LSTM(
            config.input_length, config.input_dim, config.num_hidden,
            config.num_classes, config.batch_size
        )

    ###########################################################################
    # Implement code here.
    ###########################################################################
    x = tf.placeholder(tf.float32, shape = [config.batch_size, config.input_length - 1, config.input_dim], name='inputs')
    labels = tf.placeholder(tf.float32, shape = [config.batch_size, config.num_classes], name='labels')
    
    global_step = tf.Variable(0., dtype=tf.float32)
    
    # Define the optimizer
    optimizer = tf.train.RMSPropOptimizer(config.learning_rate)
    
    
    ###########################################################################
    # Implement code here.
    ###########################################################################

    ###########################################################################
    # QUESTION: what happens here and why?
    ###########################################################################
    logits = model.compute_logits(x)
    loss = model.compute_loss(logits, labels)
    accuracy = model.accuracy(logits, labels)
    
    grads_and_vars = optimizer.compute_gradients(loss)

    grads, variables = zip(*grads_and_vars)
    grads_clipped, _ = tf.clip_by_global_norm(grads, clip_norm=config.max_norm_gradient)
    apply_gradients_op = optimizer.apply_gradients(zip(grads_clipped, variables), global_step=global_step)
    ############################################################################

    ###########################################################################
    # Implement code here.
    ###########################################################################
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for train_step in range(config.train_steps + 1):
            batch = utils.generate_palindrome_batch(config.batch_size, config.input_length)
            numbers = np.reshape(batch[:,:-1], [config.batch_size, config.input_length - 1, config.input_dim])
            final =  np.eye(config.num_classes)[batch[:,-1]] # make the last one-hot
            
            # Only for time measurement of step through network
            t1 = time.time()
            

            
            sess.run(apply_gradients_op, {x: numbers, labels: final})
                
            # Only for time measurement of step through network
            t2 = time.time()+10**-6
            examples_per_second = config.batch_size/float(t2-t1)
    
            # Print the training progress
            if train_step % config.print_every == 0:
                batch = utils.generate_palindrome_batch(config.batch_size, config.input_length) #test batch
                numbers = np.reshape(batch[:,:-1], [config.batch_size, config.input_length - 1, config.input_dim])
                final =  np.eye(config.num_classes)[batch[:,-1]]

                curr_acc, curr_loss = sess.run([accuracy, loss], {x: numbers, labels: final})
                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, "
                      "Examples/Sec = {:.2f}, Accuracy = {}, Loss = {}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), train_step,
                    config.train_steps, config.batch_size, examples_per_second,
                    curr_acc, curr_loss
                ))


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=5, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=2500, help='Number of training steps')
    parser.add_argument('--max_norm_gradient', type=float, default=10.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')

    config = parser.parse_args()

    # Train the model
    train(config)





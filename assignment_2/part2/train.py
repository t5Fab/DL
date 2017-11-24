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

import time
from datetime import datetime
import argparse

import numpy as np
import tensorflow as tf

from dataset import TextDataset
from model import TextGenerationModel

from scipy import stats

import os
import shutil

def train(config):

    # Initialize the text dataset
    dataset = TextDataset('books/' + config.txt_file)

    # Initialize the model
    model = TextGenerationModel(
        batch_size=config.batch_size,
        seq_length=config.seq_length,
        vocabulary_size=dataset.vocab_size,
        lstm_num_hidden=config.lstm_num_hidden,
        lstm_num_layers=config.lstm_num_layers
    )

    ###########################################################################
    # Implement code here.
    ###########################################################################
    
    x = tf.placeholder(shape=(None, None), dtype=tf.int32, name = 'in_sequence')
    y = tf.placeholder(shape=(None, None), dtype=tf.int32, name = 'out_sequence')

    zero_state = tf.zeros(shape=[config.lstm_num_layers, 2, config.batch_size, config.lstm_num_hidden], dtype=tf.float32)
    initial_state = tf.placeholder_with_default(zero_state, shape=[config.lstm_num_layers, 2, config.batch_size, config.lstm_num_hidden])
    
    labels = tf.one_hot(indices = y, depth = dataset.vocab_size)

    logits, states = model._build_model(x, initial_state)
    loss = model._compute_loss(logits, labels)
    prediction = model.predictions(logits)
    global_step = tf.Variable(0., dtype=tf.float32)
    # Define the optimizer
    optimizer = tf.train.RMSPropOptimizer(config.learning_rate)

    # Compute the gradients for each variable
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step)
    grads, variables = zip(*grads_and_vars)
    grads_clipped, _ = tf.clip_by_global_norm(grads, clip_norm=config.max_norm_gradient)
    apply_gradients_op = optimizer.apply_gradients(zip(grads_clipped, variables), global_step=global_step)

    ###########################################################################
    # Implement code here.
    ###########################################################################
    num_losses = 5
    losses = []

    saver = tf.train.Saver()

    with tf.Session() as sess:
        if config.load:
            print('loading past session---------------------------------------------------')
            saver.restore(sess, "./saved_session/generative_LSTM.ckpt")
        else:
            sess.run(tf.global_variables_initializer())

        for train_step in range(int(global_step.eval()), int(config.train_steps) + 1):

            # Only for time measurement of step through network
            t1 = time.time()

            #######################################################################
            # Implement code here.
            #######################################################################
            sentences, follows = dataset.batch(config.batch_size, config.seq_length)

            sess.run(apply_gradients_op, {x: sentences, y: follows})
            

            # Only for time measurement of step through network
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)

            # Output the training progress
            if train_step % config.print_every == 0:
                sentences, follows = dataset.batch(config.batch_size, config.seq_length)
                curr_loss = sess.run(loss, {x: sentences, y: follows})

                losses.append(curr_loss)

                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, Loss = {}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), train_step+1,
                    int(config.train_steps), config.batch_size, examples_per_second, curr_loss
                ))

            if train_step % config.sample_every == 0:
                sentences, _ = dataset.batch(config.batch_size, config.seq_length)
                for i in range(config.seq_length-1):
                    sentences[:, i+1] = sess.run(prediction, {x: sentences})[:, i]
                for s in sentences[:5]:
                    print(dataset.convert_to_string(s))

                if config.save:
                    saver.save(sess, './saved_session/generative_LSTM.ckpt')#, global_step = global_step)

            if train_step % config.sample_every*10 == 0:
                for filename in [f for f in os.listdir('./saved_session') if 'old' not in f]:
                    shutil.copy('./saved_session/' + filename, './saved_session/old_checkpoint/' + filename)

            # early stopping
            if (len(losses) > 2*num_losses) and (train_step > config.train_steps/100):
                old_loss = np.array(losses[(-2*num_losses):-num_losses])
                new_loss = np.array(losses[-num_losses:])
                if (stats.ttest_ind(old_loss, new_loss, equal_var = False)[1] > .5):
                    print('early stopping at: ' + str(curr_loss))
                    break


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')

    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')
    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm_gradient', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--load', type=bool, default=False, help='Controls if a previous session is loaded')
    parser.add_argument('--save', type=bool, default=True, help='Indicates if the session gets saved')

    parser.add_argument('--gpu_mem_frac', type=float, default=0.5, help='Fraction of GPU memory to allocate')
    parser.add_argument('--log_device_placement', type=bool, default=False, help='Log device placement for debugging')
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    config = parser.parse_args()

    # Train the model
    train(config)
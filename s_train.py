#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 12:58:16 2017

@author: sergigomezpalleja
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import importlib 

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import s00_input_params as input_params
#importlib.reload(input_params)
import s01_input_data as input_data
#importlib.reload(input_data)
import s03_models as models
#importlib.reload(models)
from tensorflow.python.platform import gfile

args = input_params.parser.parse_args()

# We want to see all the logging messages.
tf.logging.set_verbosity(tf.logging.INFO)
  
# Start a new TensorFlow session.
sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
  
model_settings = models.prepare_model_settings(
    len(input_data.prepare_words_list(args.wanted_words.split(','))),
    args.sample_rate, args.clip_duration_ms, args.window_size_ms,
    args.window_stride_ms, args.dct_coefficient_count, args.energy_coef_volume, 
    args.pooling_dims.split(','), args.stride_dims.split(','),
    args.beta1, args.beta2)

audio_processor = input_data.AudioProcessor(
      args.download_data, args.data_url, args.data_dir, args.audio_train_path, args.silence_percentage,
      args.unknown_percentage,
      args.wanted_words.split(','), args.development_percentage,
      args.testing_percentage, model_settings)

fingerprint_size = model_settings['fingerprint_size']
label_count = model_settings['label_count']
time_shift_samples = int((args.time_shift_ms * args.sample_rate) / 1000)
# Figure out the learning rates for each training phase. Since it's often
# effective to have high learning rates at the start of training, followed by
# lower levels towards the end, the number of steps and learning rates can be
# specified as comma-separated lists to define the rate at each stage. For
# example --how_many_training_steps=10000,3000 --learning_rate=0.001,0.0001
# will run 13,000 training loops in total, with a rate of 0.001 for the first
# 10,000, and 0.0001 for the final 3,000.
training_steps_list = list(map(int, args.how_many_training_steps.split(',')))
learning_rates_list = list(map(float, args.learning_rate.split(',')))
if len(training_steps_list) != len(learning_rates_list):
    raise Exception(
        '--how_many_training_steps and --learning_rate must be equal length '
        'lists, but are %d and %d long instead' % (len(training_steps_list),
                                                   len(learning_rates_list)))
fingerprint_input = tf.placeholder(
      tf.float32, [None, fingerprint_size], name='fingerprint_input')

logits, dropout_prob,first_weights,second_weights,final_fc_weights = models.create_model(
      fingerprint_input,
      model_settings,
      args.model_architecture,
      is_training=True)

# Define loss and optimizer
ground_truth_input = tf.placeholder(
      tf.int64, [None], name='groundtruth_input')

# Optionally we can add runtime checks to spot when NaNs or other symptoms of
# numerical errors start occurring during training.
control_dependencies = []
if args.check_nans:
    checks = tf.add_check_numerics_ops()
    control_dependencies = [checks]

# Create the back propagation and training evaluation machinery in the graph.
with tf.name_scope('cross_entropy'):
    cross_entropy_mean = (tf.losses.sparse_softmax_cross_entropy(
            labels=ground_truth_input, logits=logits)
            + args.regul_coef*tf.nn.l2_loss(first_weights) 
            + args.regul_coef*tf.nn.l2_loss(second_weights)
            + args.regul_coef*tf.nn.l2_loss(final_fc_weights))

cross_entropy_summary = tf.summary.scalar('cross_entropy', cross_entropy_mean)  
beta1 = model_settings['beta1']
beta2 = model_settings['beta2']
#with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
with tf.name_scope('train'):
    learning_rate_input = tf.placeholder(
            tf.float32, [], name = 'learning_rate_input')
    if (args.optim_method == 'GDO'):
        train_step = tf.train.GradientDescentOptimizer(
            learning_rate_input).minimize(cross_entropy_mean)
    elif (args.optim_method == 'adam'):
        if beta1 and beta2:
            train_step = tf.train.AdamOptimizer(
                learning_rate_input, beta1, beta2).minimize(cross_entropy_mean)
        else:
            train_step = tf.train.AdamOptimizer(
                learning_rate_input).minimize(cross_entropy_mean)

res_op = tf.identity(logits, name='op_to_restore')
predicted_indices = tf.argmax(logits, 1)
correct_prediction = tf.equal(predicted_indices, ground_truth_input)
confusion_matrix = tf.confusion_matrix(
            ground_truth_input, predicted_indices, num_classes=label_count)
evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
evaluation_step_summary = tf.summary.scalar('accuracy', evaluation_step)    
    
global_step = tf.train.get_or_create_global_step()
increment_global_step = tf.assign(global_step, global_step + 1)    
    
saver = tf.train.Saver(tf.global_variables())
    
# Merge all the summaries and write them out to /tmp/retrain_logs (by default)
merged_summaries = tf.summary.merge([cross_entropy_summary, evaluation_step_summary])
train_writer = tf.summary.FileWriter(args.summaries_dir + 'train/',
                                       sess.graph)
development_writer = tf.summary.FileWriter(args.summaries_dir + 'development/')    
    
tf.global_variables_initializer().run()

start_step = 1

if args.start_checkpoint: 
    models.load_variables_from_checkpoint(sess, args.start_checkpoint)
    start_step = global_step.eval(session=sess)
    
tf.logging.info('Training from step: %d ', start_step)

# Save graph.pbtxt.    
tf.train.write_graph(sess.graph_def, args.train_dir,
                     args.model_architecture + '.pbtxt')

# Save list of words.
with gfile.GFile(
      os.path.join(args.train_dir, args.model_architecture + '_labels.txt'),
      'w') as f:
    f.write('\n'.join(audio_processor.words_list))
    
# Training loop
#training_steps_max = 10
training_steps_max = np.sum(training_steps_list)
for training_step in xrange(start_step, training_steps_max + 1):
    # Figure out what the current learning rate is.
    training_steps_sum = 0
    for i in range(len(training_steps_list)):
        training_steps_sum += training_steps_list[i]
        if training_step <= training_steps_sum:
            learning_rate_value = learning_rates_list[i]
            break
        
    # Pull the audio samples we'll use for training.
    train_fingerprints, train_ground_truth = audio_processor.get_data(
        args.batch_size, 0, model_settings, args.background_frequency,
        args.background_volume, time_shift_samples, 'training', sess)     
    # Run the graph with this batch of training data.
    train_summary, train_accuracy, cross_entropy_value, _, _ = sess.run(
        [
            merged_summaries, evaluation_step, cross_entropy_mean, train_step,
            increment_global_step
        ],
        feed_dict={
            fingerprint_input: train_fingerprints,
            ground_truth_input: train_ground_truth,
            learning_rate_input: learning_rate_value,
            dropout_prob: 0.5
        })
    train_writer.add_summary(train_summary, training_step)
    #tf.logging.info('Step #%d: rate %f, accuracy %.1f%%, cross entropy %f' %
     #               (training_step, learning_rate_value, train_accuracy * 100,
     #                cross_entropy_value))
    is_last_step = (training_step == training_steps_max)
    if (training_step % args.eval_step_interval) == 0 or is_last_step:
        set_size = audio_processor.set_size('development')
        total_accuracy = 0
        total_conf_matrix = None
        for i in xrange(0, set_size, args.batch_size):
            dev_fingerprints, dev_ground_truth = (
                    audio_processor.get_data(args.batch_size, i, model_settings, 0.0,
                                     0.0, 0, 'development', sess))
            # Run a validation step and capture training summaries for TensorBoard
            # with the `merged` op.
            dev_summary, dev_accuracy, conf_matrix = sess.run(
            [merged_summaries, evaluation_step, confusion_matrix],
            feed_dict={
                fingerprint_input: dev_fingerprints,
                ground_truth_input: dev_ground_truth,
                dropout_prob: 1.0
            })
            development_writer.add_summary(dev_summary, training_step)
            batch_size = min(args.batch_size, set_size - i)
            total_accuracy += (dev_accuracy * batch_size) / set_size
            if total_conf_matrix is None:
                total_conf_matrix = conf_matrix
            else:
                    total_conf_matrix += conf_matrix
        tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
        tf.logging.info('Step %d: Validation accuracy = %.1f%% (N=%d)' %
                      (training_step, total_accuracy * 100, set_size))
        # Save the model checkpoint periodically.
        if (training_step % args.save_step_interval == 0 or
                training_step == training_steps_max):
            checkpoint_path = os.path.join(args.train_dir,
                                     args.model_architecture + '_' + args.name_model +'.ckpt')
            tf.logging.info('Saving to "%s-%d"', checkpoint_path, training_step)
            saver.save(sess, checkpoint_path, global_step=training_step)

set_size = audio_processor.set_size('testing')
tf.logging.info('set_size=%d', set_size)
total_accuracy = 0
total_conf_matrix = None
for i in xrange(0, set_size, args.batch_size):
    test_fingerprints, test_ground_truth = audio_processor.get_data(
        args.batch_size, i, model_settings, 0.0, 0.0, 0, 'testing', sess)
    test_accuracy, conf_matrix = sess.run(
        [evaluation_step, confusion_matrix],
        feed_dict={
            fingerprint_input: test_fingerprints,
            ground_truth_input: test_ground_truth,
            dropout_prob: 1.0
        })
    batch_size = min(args.batch_size, set_size - i)
    total_accuracy += (test_accuracy * batch_size) / set_size
    if total_conf_matrix is None:
      total_conf_matrix = conf_matrix
    else:
      total_conf_matrix += conf_matrix

tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (total_accuracy * 100,
                                                           set_size))  

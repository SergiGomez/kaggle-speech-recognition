#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 12:54:53 2017
@author: sergigomezpalleja
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
Model definitions for simple speech recognition.
"""

import math

import tensorflow as tf

def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms,
                           dct_coefficient_count,energy_coef_volume,
                           pooling_dims, stride_dims, beta1, beta2):
    
    """Calculates common settings needed for all models.
    Args:
    label_count: How many classes are to be recognized.
    sample_rate: Number of audio samples per second.
    clip_duration_ms: Length of each audio clip to be analyzed.
    window_size_ms: Duration of frequency analysis window.
    window_stride_ms: How far to move in time between frequency windows.
    dct_coefficient_count: Number of frequency bins to use for analysis.
    
     Returns:
    Dictionary containing common settings.
    """
    output = dict()
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    length_minus_window = (desired_samples - window_size_samples)
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
        fingerprint_size = dct_coefficient_count * spectrogram_length
    pooling_dims = [int(x) for x in pooling_dims]
    stride_dims = [int(x) for x in stride_dims]    
    beta1 = 1. - math.pow(10.,float(beta1))
    beta2 = 1. - math.pow(10.,float(beta2))
    output['desired_samples'] = desired_samples
    output['window_size_samples'] = window_size_samples
    output['window_stride_samples'] = window_stride_samples
    output['spectrogram_length'] = spectrogram_length
    output['dct_coefficient_count'] = dct_coefficient_count
    output['fingerprint_size'] = fingerprint_size
    output['label_count'] = label_count
    output['sample_rate'] = sample_rate
    output['energy_coef_volume'] = energy_coef_volume
    output['pooling_dims'] = pooling_dims
    output['stride_dims'] = stride_dims
    output['beta1'] = beta1
    output['beta2'] = beta2
    return output    

def create_model(fingerprint_input, model_settings, model_architecture,
                 is_training, runtime_settings=None):
    
    """Builds a model of the requested architecture compatible with the settings.
  There are many possible ways of deriving predictions from a spectrogram
  input, so this function provides an abstract interface for creating different
  kinds of models in a black-box way. You need to pass in a TensorFlow node as
  the 'fingerprint' input, and this should output a batch of 1D features that
  describe the audio. Typically this will be derived from a spectrogram that's
  been run through an MFCC, but in theory it can be any feature vector of the
  size specified in model_settings['fingerprint_size'].
  The function will build the graph it needs in the current TensorFlow graph,
  and return the tensorflow output that will contain the 'logits' input to the
  softmax prediction process. If training flag is on, it will also return a
  placeholder node that can be used to control the dropout amount.
  See the implementations below for the possible model architectures that can be
  requested.
  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    model_architecture: String specifying which kind of model to create.
    is_training: Whether the model is going to be used for training.
    runtime_settings: Dictionary of information about the runtime.
  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
  Raises:
    Exception: If the architecture type isn't recognized.
  """
    if model_architecture == 'single_fc':
        return create_single_fc_model(fingerprint_input, model_settings,
                                  is_training)
    elif model_architecture == 'conv':
        return create_conv_model(fingerprint_input, model_settings, is_training)
    elif model_architecture == 'low_latency_conv':
        return create_low_latency_conv_model(fingerprint_input, model_settings,
                                             is_training)
    elif model_architecture == 'low_latency_svdf':
        return create_low_latency_svdf_model(fingerprint_input, model_settings,
                                         is_training, runtime_settings)
    elif model_architecture == 'cnn_tpool2':
        return create_tpool2_model(fingerprint_input, model_settings, is_training)
    elif model_architecture == 'lstm':
        return create_rnn_model(fingerprint_input, model_settings, is_training)
    else:
        raise Exception('model_architecture argument "' + model_architecture +
                    '" not recognized, should be one of "single_fc", "conv",' +
                    ' "low_latency_conv, or "low_latency_svdf"')
        
    
def load_variables_from_checkpoint(sess, start_checkpoint):
    """Utility function to centralize checkpoint restoration.
    Args:
    sess: TensorFlow session.
    start_checkpoint: Path to saved checkpoint on disk.
    """
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, start_checkpoint)
    
    
def create_single_fc_model(fingerprint_input, model_settings, is_training):
      
    """Builds a model with a single hidden fully-connected layer.
    This is a very simple model with just one matmul and bias layer. As you'd
    expect, it doesn't produce very accurate results, but it is very fast and
    simple, so it's useful for sanity testing.
    Here's the layout of the graph:
    (fingerprint_input)
          v
    [MatMul]<-(weights)
          v
    [BiasAdd]<-(bias)
          v
    Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.
    Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
    """
    if is_training:    
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    fingerprint_size = model_settings['fingerprint_size']
    label_count = model_settings['label_count']
    weights = tf.Variable(
            tf.truncated_normal([fingerprint_size, label_count], stddev=0.001))
    bias = tf.Variable(tf.zeros([label_count]))
    logits = tf.matmul(fingerprint_input, weights) + bias
    if is_training:
        return logits, dropout_prob
    else:
        return logits        

def create_conv_model(fingerprint_input, model_settings, is_training):
  """Builds a standard convolutional model.
  This is roughly the network labeled as 'cnn-trad-fpool3' in the
  'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
  http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf
  Here's the layout of the graph:
  (fingerprint_input)
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MaxPool]
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MaxPool]
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
  This produces fairly good quality results, but can involve a large number of
  weight parameters and computations. For a cheaper alternative from the same
  paper with slightly less accuracy, see 'low_latency_conv' below.
  During training, dropout nodes are introduced after each relu, controlled by a
  placeholder.
  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.
  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
  """
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
  first_filter_width = 8
  first_filter_height = 20
  first_filter_count = 64
  first_weights = tf.Variable(
      tf.truncated_normal(
          [first_filter_height, first_filter_width, 1, first_filter_count],
          stddev=0.01))
  first_bias = tf.Variable(tf.zeros([first_filter_count]))
  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 1, 1, 1],
                            'SAME') + first_bias
  first_relu = tf.nn.relu(first_conv)
  if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  else:
    first_dropout = first_relu
  max_pool = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
  second_filter_width = 4
  second_filter_height = 10
  second_filter_count = 64
  second_weights = tf.Variable(
      tf.truncated_normal(
          [
              second_filter_height, second_filter_width, first_filter_count,
              second_filter_count
          ],
          stddev=0.01))
  second_bias = tf.Variable(tf.zeros([second_filter_count]))
  second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 1, 1, 1], 'SAME') + second_bias
  second_relu = tf.nn.relu(second_conv)
  if is_training:
    second_dropout = tf.nn.dropout(second_relu, dropout_prob)
  else:
    second_dropout = second_relu
  second_conv_shape = second_dropout.get_shape()
  second_conv_output_width = second_conv_shape[2]
  second_conv_output_height = second_conv_shape[1]
  second_conv_element_count = int(
      second_conv_output_width * second_conv_output_height *
      second_filter_count)
  flattened_second_conv = tf.reshape(second_dropout,
                                     [-1, second_conv_element_count])
  label_count = model_settings['label_count']
  final_fc_weights = tf.Variable(
      tf.truncated_normal(
          [second_conv_element_count, label_count], stddev=0.01))
  final_fc_bias = tf.Variable(tf.zeros([label_count]))
  final_fc = tf.matmul(flattened_second_conv, final_fc_weights) + final_fc_bias
  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc


def create_tpool2_model(fingerprint_input, model_settings, is_training):
  """Builds a standard convolutional model.
  This is roughly the network labeled as 'cnn-trad-fpool3' in the
  'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
  http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf
  Here's the layout of the graph:
  (fingerprint_input)
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MaxPool]
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MaxPool]
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
  This produces fairly good quality results, but can involve a large number of
  weight parameters and computations. For a cheaper alternative from the same
  paper with slightly less accuracy, see 'low_latency_conv' below.
  During training, dropout nodes are introduced after each relu, controlled by a
  placeholder.
  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    is_training: Whether the model is going to be used for training.
  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.
  """
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  pooling_time_size, pooling_freq_size  = model_settings['pooling_dims']
  stride_time_size, stride_freq_size  = model_settings['stride_dims']
  fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size, 1])
  first_filter_width = 8
  first_filter_height = 21
  first_filter_count = 94
  first_weights = tf.Variable(
      tf.truncated_normal(
          [first_filter_height, first_filter_width, 1, first_filter_count],
          stddev=0.01))
  #Sergi 2018-01-14:También tengo que cambiar a formato NCHW aquí
#  first_weights = tf.Variable(
#      tf.truncated_normal(
#          [1,first_filter_height, first_filter_width, first_filter_count],
#          stddev=0.01))
  first_bias = tf.Variable(tf.zeros([first_filter_count]))
  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 1, 1, 1],'SAME', use_cudnn_on_gpu = True) + first_bias
  first_relu = tf.nn.relu(first_conv)
  if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  else:
    first_dropout = first_relu
  max_pool = tf.nn.max_pool(first_dropout, [1, pooling_time_size, pooling_freq_size, 1], 
                            [1, stride_time_size, stride_freq_size, 1], 'SAME')
  second_filter_width = 4
  second_filter_height = 10
  second_filter_count = 94
  second_weights = tf.Variable(
      tf.truncated_normal(
          [
              second_filter_height, second_filter_width, first_filter_count,
              second_filter_count
          ],
          stddev=0.01))
  second_bias = tf.Variable(tf.zeros([second_filter_count]))
#  second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 1, 1, 1],
#                             'SAME', data_format='NCHW') + second_bias
  
  second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 1, 1, 1], 'SAME') + second_bias
  second_relu = tf.nn.relu(second_conv)
  if is_training:
    second_dropout = tf.nn.dropout(second_relu, dropout_prob)
  else:
    second_dropout = second_relu
  second_conv_shape = second_dropout.get_shape()
  second_conv_output_width = second_conv_shape[2]
  second_conv_output_height = second_conv_shape[1]
  second_conv_element_count = int(
      second_conv_output_width * second_conv_output_height *
      second_filter_count)
  flattened_second_conv = tf.reshape(second_dropout,
                                     [-1, second_conv_element_count])
  label_count = model_settings['label_count']
  final_fc_weights = tf.Variable(
      tf.truncated_normal(
          [second_conv_element_count, label_count], stddev=0.01))
  final_fc_bias = tf.Variable(tf.zeros([label_count]))
  final_fc = tf.matmul(flattened_second_conv, final_fc_weights) + final_fc_bias
  if is_training:
    return final_fc, dropout_prob,first_weights,second_weights,final_fc_weights  
  else:
    return final_fc
        

def LSTMStep(my_input, previous_output, memory, my_shape):
    real_input = tf.concat([previous_output, my_input], 1)
    sigmoid_output1 = tf.contrib.layers.fully_connected(real_input, my_shape, activation_fn = tf.nn.sigmoid) # OJO INDEX
    sigmoid_output2 = tf.contrib.layers.fully_connected(real_input, my_shape, activation_fn = tf.nn.sigmoid)
    tanh_output1 = tf.contrib.layers.fully_connected(real_input, my_shape, activation_fn = tf.nn.tanh)
    sigmoid_output3 = tf.contrib.layers.fully_connected(real_input, my_shape, activation_fn = tf.nn.sigmoid)
    sigtanh_output = sigmoid_output2*tanh_output1
    memory_out = memory*sigmoid_output1 + sigtanh_output
    out = tf.nn.tanh(memory_out)*sigmoid_output3
    return out, memory_out

                
def create_rnn_model(fingerprint_input, model_settings, is_training):
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                              [-1, input_time_size, input_frequency_size])

    #out = tf.zeros([-1, 1, input_frequency_size, 1]) # DICCIONARIO en caso de no funcionar con clave step_idx
    x_out = tf.placeholder(tf.float32, shape=[None, input_frequency_size], name='x_out')
    out = dict()
    out[0] = tf.zeros_like(x_out)
    #out = tf.zeros_like(x_out)

    x_memory = tf.placeholder(tf.float32, shape=[None, input_frequency_size], name='x_memory')
    memory = dict()
    memory[0] = tf.zeros_like(x_memory)
    #memory = tf.zeros_like(x_memory)
    #memory = tf.zeros([-1, 1, input_frequency_size, 1])
    for step_idx in range(0, input_time_size):
        out[step_idx + 1], memory[step_idx + 1] = LSTMStep(
                    fingerprint_4d[:,step_idx,:],
                    out[step_idx],
                    memory[step_idx],
                    model_settings['dct_coefficient_count']
                )
    final_out = tf.contrib.layers.fully_connected(out[input_time_size], 12, activation_fn = tf.nn.softmax)
    if is_training:
        return final_out, dropout_prob
    else:
        return final_out

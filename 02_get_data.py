#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 21:18:37 2017

@author: sergigomezpalleja
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os.path import join
from glob import glob

import re
import hashlib
import math
import os.path
import random
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
RANDOM_SEED = 59185

sample_rate = 16000
clip_duration_ms = 1000
desired_samples = int(sample_rate * clip_duration_ms / 1000)

class AudioProcessor(object):
  """Handles loading, partitioning, and preparing audio training data."""

  def __init__(self, data_url, data_dir, silence_percentage, unknown_percentage,
               wanted_words, validation_percentage, testing_percentage,
               model_settings):
    self.data_dir = data_dir
    self.maybe_download_and_extract_dataset(data_url, data_dir)
    self.prepare_data_index(silence_percentage, unknown_percentage,
                            wanted_words, validation_percentage,
                            testing_percentage)
    self.prepare_background_data()
    self.prepare_processing_graph(model_settings)
    
     def prepare_processing_graph(self, model_settings):
    """Builds a TensorFlow graph to apply the input distortions.
    Creates a graph that loads a WAVE file, decodes it, scales the volume,
    shifts it in time, adds in background noise, calculates a spectrogram, and
    then builds an MFCC fingerprint from that.
    This must be called with an active TensorFlow session running, and it
    creates multiple placeholder inputs, and one output:
      - wav_filename_placeholder_: Filename of the WAV to load.
      - foreground_volume_placeholder_: How loud the main clip should be.
      - time_shift_padding_placeholder_: Where to pad the clip.
      - time_shift_offset_placeholder_: How much to move the clip in time.
      - background_data_placeholder_: PCM sample data for background noise.
      - background_volume_placeholder_: Loudness of mixed-in background.
      - mfcc_: Output 2D fingerprint of processed audio.
    Args:
      model_settings: Information about the current model being trained.
    """
    desired_samples = model_settings['desired_samples']
    wav_filename_placeholder_ = tf.placeholder(tf.string, [])
    wav_loader = io_ops.read_file(wav_filename_placeholder_)
    wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1, desired_samples=desired_samples)
    # Allow the audio sample's volume to be adjusted.
    foreground_volume_placeholder_ = tf.placeholder(tf.float32, [])
    scaled_foreground = tf.multiply(wav_decoder.audio,foreground_volume_placeholder_)
    # Shift the sample's start position, and pad any gaps with zeros.
    time_shift_padding_placeholder_ = tf.placeholder(tf.int32, [2, 2])
    time_shift_offset_placeholder_ = tf.placeholder(tf.int32, [2])
    padded_foreground = tf.pad(
        scaled_foreground,
        time_shift_padding_placeholder_,
        mode='CONSTANT')
    sliced_foreground = tf.slice(padded_foreground,
                                 time_shift_offset_placeholder_,
                                 [desired_samples, -1])
    # Mix in background noise.
    self.background_data_placeholder_ = tf.placeholder(tf.float32,
                                                       [desired_samples, 1])
    self.background_volume_placeholder_ = tf.placeholder(tf.float32, [])
    background_mul = tf.multiply(self.background_data_placeholder_,
                                 self.background_volume_placeholder_)
    background_add = tf.add(background_mul, sliced_foreground)
    background_clamp = tf.clip_by_value(background_add, -1.0, 1.0)
    # Run the spectrogram and MFCC ops to get a 2D 'fingerprint' of the audio.
    spectrogram = contrib_audio.audio_spectrogram(
        background_clamp,
        window_size=model_settings['window_size_samples'],
        stride=model_settings['window_stride_samples'],
        magnitude_squared=True)
    self.mfcc_ = contrib_audio.mfcc(
        spectrogram,
        wav_decoder.sample_rate,
        dct_coefficient_count=model_settings['dct_coefficient_count'])

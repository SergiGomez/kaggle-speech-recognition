#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 13:02:06 2017

@author: sergigomezpalleja
"""
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
'--name_model',
  type=str,
  # pylint: disable=line-too-long
  default='cnn_tpool2_adam_',
  # pylint: enable=line-too-long
  help='The name of the model that will be used to save the ckpt file')
parser.add_argument(
'--data_url',
  type=str,
  # pylint: disable=line-too-long
  default='http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz',
  # pylint: enable=line-too-long
  help='Location of speech training data archive on the web.')
parser.add_argument(
  '--data_dir',
  type=str,
  default='data/',
  help="""\
  Where to download the speech training data to.
  """)
parser.add_argument(
  '--optim_method',
  type=str,
  default='adam', #adam o GDO
  help="""\
  Where to download the speech training data to.
  """)
parser.add_argument(
  '--energy_coef_volume',
  type=bool,
  default=False,
  help="""\
  If we want to add the coef of the energies (data / background) where we add the volume to the signal.
  """)
parser.add_argument(
  '--background_volume',
  type=float,
  #Sergi 2018-01-07: I try to increase volume and see if we reduce overfitting
  default=0.2,
  help="""\
  How loud the background noise should be, between 0 and 1.
  """)
parser.add_argument(
  '--background_frequency',
  type=float,
  default=0.8,
  help="""\
  How many of the training samples have background noise mixed in.
  """)
parser.add_argument(
  '--silence_percentage',
  type=float,
  default=10.0,
  help="""\
  How much of the training data should be silence.
  """)
parser.add_argument(
  '--unknown_percentage',
  type=float,
  default=10.0,
  help="""\
  How much of the training data should be unknown words.
  """)
parser.add_argument(
  '--time_shift_ms',
  type=float,
  default=100.0,
  help="""\
  Range to randomly shift the training audio by in time.
  """)
parser.add_argument(
  '--testing_percentage',
  type=int,
  default=10,
  help='What percentage of wavs to use as a test set.')
parser.add_argument(
  '--development_percentage',
  type=int,
  default=10,
  help='What percentage of wavs to use as a dev set.')
parser.add_argument(
  '--sample_rate',
  type=int,
  default=16000,
  help='Expected sample rate of the wavs',)
parser.add_argument(
  '--clip_duration_ms',
  type=int,
  default=1000,
  help='Expected duration in milliseconds of the wavs',)
parser.add_argument(
  '--window_size_ms',
  type=float,
  default=30.0,
  help='How long each spectrogram timeslice is',)
parser.add_argument(
  '--window_stride_ms',
  type=float,
  default=10.0,
  help='How long each spectrogram timeslice is',)
parser.add_argument(
  '--dct_coefficient_count',
  type=int,
  default=40,
  help='How many bins to use for the MFCC fingerprint',)
parser.add_argument(
  '--how_many_training_steps',
  type=str,
  default='8000,2000',
  help='How many training loops to run',)
parser.add_argument(
  '--eval_step_interval',
  type=int,
  default=400,
  help='How often to evaluate the training results.')
parser.add_argument(
  '--learning_rate',
  type=str,
  default='0.00032,0.000032',
  help='How large a learning rate to use when training.')
parser.add_argument(
  '--batch_size',
  type=int,
  default=100,
  help='How many items to train with at once',)
parser.add_argument(
  '--summaries_dir',
  type=str,
  default='retrain_logs/',
  help='Where to save summary logs for TensorBoard.')
parser.add_argument(
  '--wanted_words',
  type=str,
  default='yes,no,up,down,left,right,on,off,stop,go',
  help='Words to use (others will be added to an unknown label)',)
parser.add_argument(
  '--train_dir',
  type=str,
  default='speech_commands_train/',
  help='Directory to write event logs and checkpoint.')
parser.add_argument(
  '--save_step_interval',
  type=int,
  default=100,
  help='Save model checkpoint every save_steps.')
parser.add_argument(
  '--start_checkpoint',
  type=str,
  default='',
  help='If specified, restore this pretrained model before any training.')
parser.add_argument(
  '--model_architecture',
  type=str,
  default= 'cnn_tpool2', #'single_fc'
  help='What model architecture to use')
parser.add_argument(
  '--check_nans',
  type=bool,
  default=True,
  help='Whether to check for invalid numbers during processing')
parser.add_argument(
  '--download_data',
  type=bool,
  default=False,
  help='Whether to import data from url')
parser.add_argument(
  '--audio_train_path',
  type=str,
  default= 'data/train/audio/',
  help='Path to the audio for training')
parser.add_argument(
  '--pooling_dims',
  type=str,
  default= '2,2',
  help='Pooling dimensions in time x frequency')
parser.add_argument(
  '--stride_dims',
  type=str,
  default= '2,2',
  help='Stride dimensions in time x frequency')
parser.add_argument(
  '--unknown_threshold',
  type=float,
  default= -1,
  help='Min val of (unknown - max logi)/max logit')
parser.add_argument(
  '--beta1',
  type=str,
  default= '-1',
  help='log(1-x) of beta1 param of Adam optimizer')
parser.add_argument(
  '--beta2',
  type=str,
  default= '-3',
  help='log(1-x) of beta1 param of Adam optimizer')
parser.add_argument(
  '--regul_coef',
  type=float,
  default= 0,
  help='Use Regularization term in the loss function')

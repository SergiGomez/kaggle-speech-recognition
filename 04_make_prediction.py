from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os.path import join
from glob import glob
import tensorflow as tf
import importlib
import numpy as np
import pandas as pd 
import re

import s00_input_params as input_params
#importlib.reload(input_params)
import s01_input_data as input_data
#importlib.reload(input_data)
import s03_models as models
#importlib.reload(models)

#path to test wav files
path_test_files = 'data/test/audio/' 
#path where weights have been saved
path_checkpoint = 'speech_commands_train/'

args = input_params.parser.parse_args()

# We want to see all the logging messages.
tf.logging.set_verbosity(tf.logging.INFO)

model_settings = models.prepare_model_settings(
    len(input_data.prepare_words_list(args.wanted_words.split(','))),
    args.sample_rate, args.clip_duration_ms, args.window_size_ms,
    args.window_stride_ms, args.dct_coefficient_count, args.energy_coef_volume, args.pooling_dims.split(','), args.stride_dims.split(','),
    args.beta1, args.beta2)

audio_processor_pred = input_data.AudioProcessor_Prediction(path_test_files, model_settings)

fingerprint_size = model_settings['fingerprint_size']     
label_count = model_settings['label_count']
time_shift_samples = int((args.time_shift_ms * args.sample_rate) / 1000)

sess = tf.Session()

checkpoint_file=tf.train.latest_checkpoint(path_checkpoint)
#checkpoint_file = 'speech_commands_train/cnn_tpool2_cnn_tpool2_lr4.ckpt-10000'
#First let's load meta graph and restore weights
#with import_meta_graph we create the network
saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
#we still need to load the value of the parameters that we had trained on this graph
saver.restore(sess, checkpoint_file)

graph = tf.get_default_graph()
fingerprint_input = graph.get_tensor_by_name("fingerprint_input:0")
dropout_prob = graph.get_tensor_by_name("dropout_prob:0")

op_to_restore = graph.get_tensor_by_name("op_to_restore:0")

# All testing files 

all_test_files = glob(os.path.join(path_test_files,'*wav'))
#there is one sample with less data, 
#we remove it and assign unknown to it after the for loop
all_test_files.remove('data/test/audio/clip_127f6c3c6.wav')
print(len(all_test_files))
sample_count = len(all_test_files)    
# Our submission data frame
names_files_submission = []
prediction_value_submission = []

for i in range(0, sample_count):  
    sample_file = all_test_files[i]
    pred_fingerprint = audio_processor_pred.get_one_sample_prediction(
        sample_file, 0, model_settings, 0, 'prediction', sess)
    pred_feed = np.reshape(pred_fingerprint,(1,len(pred_fingerprint)))
    if (args.model_architecture == 'single_fc'):
        feed_dict = {fingerprint_input: pred_feed}
    else: 
        feed_dict = {fingerprint_input: pred_feed,
                      dropout_prob: 1.0}
    #prediction = sess.run(op_to_restore, feed_dict)[0]
	#-------------------------------------------------------
    logits = sess.run(op_to_restore, feed_dict)[0]
    #max_logits = tf.reduce_max(logits, 1, name = "max_logits")
    #unknowns_dif = tf.add(logits[:,1], -max_logits, name = "unknowns_dif")
    #unknowns_impact = tf.div(unknowns_dif,max_logits, name = "unknowns_impact")
    #prediction = tf.where(tf.less(unknowns_impact, args.unknown_threshold),
    #                             tf.ones_like(tf.argmax(logits, 1)), tf.argmax(logits, 1))
								 
    # if array
    max_logits = np.max(logits, axis = 0)
    unknowns_impact = np.abs((logits[1]-max_logits)/max_logits)
    logits[unknowns_impact < args.unknown_threshold, 1] = 1
    prediction = np.argmax(logits)
    #-------------------------------------------------------
    sample_file_short = os.path.basename(sample_file)
    names_files_submission.append(sample_file_short)
    prediction_value_submission.append(prediction)
    if (i % 1000) == 0:
        tf.logging.info('Prediction number: %d ', i)

names_files_submission.append('clip_127f6c3c6.wav')
prediction_value_submission.append(1)
    
submission  = pd.DataFrame(columns=['fname', 'label'])
submission['fname'] = names_files_submission
submission['label'] = prediction_value_submission
submission.loc[:,'label']= submission.loc[:,'label'].replace([0],'silence')
submission.loc[:,'label']= submission.loc[:,'label'].replace([1],'unknown')
submission.loc[:,'label']= submission.loc[:,'label'].replace([2],'yes')
submission.loc[:,'label']= submission.loc[:,'label'].replace([3],'no')
submission.loc[:,'label']= submission.loc[:,'label'].replace([4],'up')
submission.loc[:,'label']= submission.loc[:,'label'].replace([5],'down')
submission.loc[:,'label']= submission.loc[:,'label'].replace([6],'left')
submission.loc[:,'label']= submission.loc[:,'label'].replace([7],'right')
submission.loc[:,'label']= submission.loc[:,'label'].replace([8],'on')
submission.loc[:,'label']= submission.loc[:,'label'].replace([9],'off')
submission.loc[:,'label']= submission.loc[:,'label'].replace([10],'stop')
submission.loc[:,'label']= submission.loc[:,'label'].replace([11],'go')    

submission.to_csv("submission_sergi.csv", index=False)

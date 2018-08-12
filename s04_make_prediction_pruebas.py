from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os.path import join
from glob import glob
import tensorflow as tf
import importlib
import numpy as np
#import pandas as pd 

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
    args.window_stride_ms, args.dct_coefficient_count)

audio_processor_pred = input_data.AudioProcessor_Prediction(path_test_files, model_settings)

fingerprint_size = model_settings['fingerprint_size']     
label_count = model_settings['label_count']
time_shift_samples = int((args.time_shift_ms * args.sample_rate) / 1000)

sess = tf.Session()
#sess.run(tf.global_variables_initializer())

#with tf.name_scope('prediction'):
#with graph.as_default():
#fingerprint_input = tf.placeholder(
#  tf.float32, [None, fingerprint_size], name='fingerprint_input_1')

checkpoint_file=tf.train.latest_checkpoint(path_checkpoint)
#First let's load meta graph and restore weights
#with import_meta_graph we create the network
saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
#we still need to load the value of the parameters that we had trained on this graph
saver.restore(sess, checkpoint_file)

graph = tf.get_default_graph()
fingerprint_input = graph.get_tensor_by_name("fingerprint_input:0")
#
## we restore the input placeholder
#logits, dropout_prob = models.create_model(
#        fingerprint_input,
#        model_settings,
#        args.model_architecture,
#        is_training=False)
#predicted_indices = tf.argmax(logits, 1)
op_to_restore = graph.get_tensor_by_name("op_to_restore:0")

# All testing files 
all_test_files = glob(os.path.join(path_test_files,'*wav'))
sample_count = len(all_test_files)    
for i in range(0, sample_count):  
    sample_file = all_test_files[i]
    pred_fingerprint = audio_processor_pred.get_one_sample_prediction(
        sample_file, 0, model_settings, 0, 'prediction', sess)
    pred_feed = np.reshape(pred_fingerprint,(1,len(pred_fingerprint)))
    feed_dict = {fingerprint_input: pred_feed}
    prediction = sess.run(op_to_restore, feed_dict)
    print(prediction)
    
    #output = tf.get_collection("placeholder")[0]
    input = graph.get_operation_by_name("evaluation_step").outputs[0]
    prediction=graph.get_operation_by_name("prediction").outputs[0]
    tf.logging.info('Model %m restored',checkpoint_file)

# All testing files 
all_test_files = glob(os.path.join(path_test_files,'*wav'))
sample_count = len(all_test_files)    

pred_labels = pd.DataFrame()
index_to_word = input_data.prepare_words_list(args.wanted_words.split(','))

for i in xrange(0, sample_count):  
    sample_file = all_test_files[i]
    pred_fingerprint = audio_processor_pred.get_one_sample_prediction(
        sample_file, 0, model_settings, 0, 'prediction', sess)
    
    feed_dict = {fingerprint_input: pred_fingerprint}
    prediction = sess.run(predicted_indices, feed_dict)
    
    prediction = sess.run(output, feed_dict={x: pred_fingerprint})
    predicted_ind = tf.argmax(prediction, 1)
    pred_labels
    df = df.append(pd.Series(['a', 'b'], index=['col1','col2']), ignore_index=True)
#with graph.as_default():
#     session_conf = tf.ConfigProto(allow_safe_placement=True, log_device_placement =False)
#     sess = tf.Session(config = session_conf)
#     with sess.as_default():
#          saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
#          saver.restore(sess,checkpoint_file)
#          input = graph.get_operation_by_name("input").outputs[0]
#          prediction=graph.get_operation_by_name("prediction").outputs[0]
#          
#          tf.logging.info('set_size=%d', set_size)
#
#          print sess.run(prediction,feed_dict={input:newdata})
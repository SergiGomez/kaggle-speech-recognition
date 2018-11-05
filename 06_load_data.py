# Import this file to use in your code
# It includes the following functions:
#   load_data
#   load_audio
#
# Read all the files in data/train directory and return a list of labels and file paths
# To use this code:
#   from load_data import load_data, load_audio
#   data_path = 'data/'  (relative path to data is ok)
#   labels, uids, paths = load_train_data(data_path)
#   audios = load_audio(paths)
#
# all audios loaded ~ 2GB RAM 

import os
from os.path import join
from glob import glob
import re
import random
import pandas as pd
import numpy as np

from scipy import signal
from scipy.io import wavfile

# Set the seed
random.seed(42)
np.random.seed(42)

## GLOBAL VARIABLES
sample_rate = 16000

known_labels = 'yes no up down left right on off stop go'.split()
unknown_labels = 'bed bird cat dog one two three four five six seven eight nine happy house marvin sheila tree wow zero'.split()
silence_label = ['silence']

possible_labels = known_labels + ['silence', 'unknown']
id2name = {i: name for i, name in enumerate(possible_labels)}
name2id = {name: i for i, name in id2name.items()}
name2id.update({name: 11 for name in unknown_labels})

## Functions
def load_train_data(data_path, max_examples = -1):
    '''
    Returns 3 lists:
      - labels: Label associated to each audio file
      - uids: id of the user that recorded each audio
      - paths: paths to each audio file
    '''
    # Count how many files of each class we have
    counter = {label : 0 for label in known_labels + unknown_labels + silence_label}

    # Just a simple regexp for paths with three groups:
    # prefix, label, user_id
    pattern = re.compile('(.+[\/\\\\])?(\w+)[\/\\\\]([^_]+)_.+wav')
    files_path = join(data_path, 'train','audio','*','*wav')
    all_files = glob(files_path)
    random.shuffle(all_files)
    possible = set(possible_labels)
    labels, uids, paths = [], [], []
    for entry in all_files:
        r = re.match(pattern, entry)
        if r:
            label, uid = r.group(2), r.group(3)
            if label == '_background_noise_':
                label = 'silence'
            if max_examples < 0 or (max_examples > 0 and counter[label] < max_examples):
                counter[label] = counter[label] + 1
                #if label not in possible:
                #    label = 'unknown'

                labels.append(label)
                uids.append(uid)
                paths.append(entry)

    print('There are {} train samples'.format(len(paths)))
    return labels, uids, paths

def load_audio(path, fill=False, resample=False):
    '''
    Returns a numpy array corresponding to decoded wav file from path
    If path is a list of paths, returns a list of decoded files
    '''

    if (isinstance(path, str)):
        # convert to list so it is iterable
        path = [path]

    samples_list = []
    for file_path in path:
        sample_rate, samples = wavfile.read(str(file_path))
        if (fill and len(samples) < 16000):
            # TODO: fill with background noise instead of zeros
            samples.resize(16000, refcheck=False)

        if resample:
            new_sample_rate = sample_rate / 2
            samples = signal.resample(samples, int(new_sample_rate / sample_rate * samples.shape[0]))

        samples_list.append(samples)

    # If the input is only one path,dont return a list of arrays, return only the array
    return samples_list if len(samples_list) > 1 else samples_list[0]

def get_silence(raw_data_df, n, resample=False):
    ''' Returns `n` silence audio files of 1 second length.
    '''
    output_df_list = []
    silence_df = raw_data_df.loc[raw_data_df['label'] == 'silence']

    num_silences = silence_df.shape[0]

    for i in range(num_silences):
        output_audios = []
        sub_silence_df = silence_df.iloc[i]
        path = sub_silence_df['path']
        audio = load_audio(path, resample)
        uid = sub_silence_df['uid']
        audio_length = len(audio)

        for j in range (int(n / num_silences)):
            starting_time = random.randint(0, audio_length - 16000)
            output_audios.append(audio[starting_time:(starting_time + 16000)])

        output_df_list.append(pd.DataFrame({'label' : 'silence',
                                            'uid'   : uid,
                                            'path'  : path,
                                            'wav'   : output_audios}))

    return pd.concat(output_df_list)

def get_balanced_unknown(raw_data_df, n):
    num_unknown = len(unknown_labels)
    num_unknown_samples = int(n / num_unknown)

    unknown_df_list = []

    for label in unknown_labels:
        aux_df = raw_data_df.loc[raw_data_df['label'] == label]
        n_samples = aux_df.shape[0]

        r = False
        if num_unknown_samples > n_samples:
            r = True

        sub_df = aux_df.iloc[np.random.choice(range(n_samples), num_unknown_samples, replace=r)]
        unknown_df_list.append(sub_df)

    return pd.concat(unknown_df_list)

def split_train(df, train_size = 0.8, dev_size = 0.1, test_size = 0.1):
    if (train_size + dev_size + test_size != 1):
        print('Weights must sum 1!')
        return

    df['set'] = np.nan

    df_list = []
    for label in possible_labels:
        sub_df = df.loc[df['label'] == label].copy()
        n = sub_df.shape[0]

        idx_sample = np.random.choice(range(n), n, replace=False)
        idx_train = idx_sample[:int(n * train_size)]
        idx_dev   = idx_sample[int(n * train_size):int(n * (train_size + dev_size))]
        idx_test  = idx_sample[int(n * (train_size + dev_size)):]

        sub_df.iloc[idx_train, 4] = 'train'
        sub_df.iloc[idx_dev, 4] = 'dev'
        sub_df.iloc[idx_test, 4] = 'test'

        df_list.append(sub_df)

    return pd.concat(df_list)

def prepare_data(data_path, augment_data = None, unknown_rate = 1, silence_rate = 1, train_size = 0.8, dev_size = 0.1,
                test_size = 0.1, random_state=None, resample = False, max_examples = -1):
    '''
    Prepare the dataset:
     1. Balances the classes (there will be n * `unknown_rate` unknown samples
        and n * `silence_rate` silence samples, where n is the mean number of
        samples of known labels.
     2. Split the dataframe in train, dev and test according to `train_size`,
        `dev_size` and `test_size` proportions.
    Returns:
     - A pandas dataframe with columns:
      * label: label of the audio
      * path: relative path to the audio file
      * uid: user of the audio
      * wav: raw decoded wav file (list)
      * set: set in which the audio belongs: train, dev or test
    '''
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    labels, uids, paths = load_train_data(data_path, max_examples)
    raw_data_df = pd.DataFrame({'label' : labels, 'uid' : uids, 'path' : paths})

    mean_samples = np.mean(raw_data_df.loc[raw_data_df['label'].isin(known_labels)]
                       .groupby(['label'])
                       .agg({'path' : 'count'})['path'])

    # Get 3 dataframes, one for known labels, another for unknown labels and one for silence audios
    known_df = raw_data_df.loc[raw_data_df['label'].isin(known_labels)].copy()
    unknown_df = get_balanced_unknown(raw_data_df, mean_samples * unknown_rate)
    silence_df = get_silence(raw_data_df, mean_samples * silence_rate, resample)

    known_df['wav'] = load_audio(known_df['path'].values, fill=True, resample=resample)
    unknown_df['wav'] = load_audio(unknown_df['path'].values, fill=True, resample=resample)

    arranged_data_df = pd.concat([known_df, unknown_df, silence_df])
    arranged_data_df.loc[arranged_data_df['label'].isin(unknown_labels), 'label'] = 'unknown'

    split_data_df = split_train(arranged_data_df, train_size, dev_size, test_size)
    split_data_df.reset_index(drop=True, inplace=True)

    return split_data_df

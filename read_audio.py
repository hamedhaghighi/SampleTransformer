import fnmatch
import os
import random
import re
import threading

import librosa
import numpy as np
import tensorflow as tf

FILE_PATTERN = r'p([0-9]+)_([0-9]+)\.wav'


def get_category_cardinality(files):
    id_reg_expression = re.compile(FILE_PATTERN)
    min_id = None
    max_id = None
    for filename in files:
        matches = id_reg_expression.findall(filename)[0]
        id, recording_id = [int(id_) for id_ in matches]
        if min_id is None or id < min_id:
            min_id = id
        if max_id is None or id > max_id:
            max_id = id

    return min_id, max_id


def randomize_files(files):
    for file in files:
        file_index = random.randint(0, (len(files) - 1))
        yield files[file_index]


def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def load_generic_audio(directory, mode='train'):
    path = os.path.join(directory, 'music_{}.npy'.format(mode))
    audio = np.load(path)
    return audio

def trim_silence(audio, threshold, frame_length=2048):
    '''Removes silence at the beginning and end of a sample.'''
    if audio.size < frame_length:
        frame_length = audio.size
    energy = librosa.feature.rms(audio, frame_length=frame_length)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array, if the whole audio was silence.
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]


def not_all_have_id(files):
    ''' Return true iff any of the filenames does not conform to the pattern
        we require for determining the category id.'''
    id_reg_exp = re.compile(FILE_PATTERN)
    for file in files:
        ids = id_reg_exp.findall(file)
        if not ids:
            return True
    return False


class AudioReader(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 audio_dir,
                 sample_rate,
                 batch_size,
                 gc_enabled,
                 receptive_field,
                 sample_size=None,
                 silence_threshold=None,
                 queue_size=32):
        self.batch_size = batch_size
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.sample_size = sample_size
        self.receptive_field = receptive_field
        self.silence_threshold = silence_threshold
        self.gc_enabled = gc_enabled
        self.threads = []
        self.place_holder_len = self.sample_size + self.receptive_field + 1
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.place_holder_len, 1))
        self.begin_placeholder = tf.placeholder(dtype=tf.bool, shape=())
        self.train_data = load_generic_audio(self.audio_dir)
        self.val_data = load_generic_audio(self.audio_dir, 'valid')
        if self.gc_enabled:
            _, self.gc_category_cardinality = get_category_cardinality(files)
            self.gc_category_cardinality += 1
            print("Detected --gc_cardinality={}".format(
                  self.gc_category_cardinality))
        else:
            self.gc_category_cardinality = None

    def get_len_dataset(self):
        def compute_len(data_loader):
            return (data_loader.shape[0]//self.batch_size)*(data_loader.shape[1]//self.sample_size)
        return compute_len(self.train_data), compute_len(self.val_data)

    def get_input_placeholder(self):
        return (self.sample_placeholder, self.begin_placeholder)

    def get_data_iterator(self , mode):
        total_audio = self.train_data if mode == 'train' else self.val_data
        np.random.shuffle(total_audio)
        while True:
            for i in range(len(total_audio)// self.batch_size):
                audio = total_audio[i*self.batch_size: (i+1)*self.batch_size]
                if self.silence_threshold is not None:
                    # Remove silence
                    audio = trim_silence(audio[:, 0], self.silence_threshold)
                    audio = audio.reshape(-1, 1)
                    if audio.size == 0:
                        print("Warning: {} was ignored as it contains only "
                                "silence. Consider decreasing trim_silence "
                                "threshold, or adjust volume of the audio."
                                .format(filename))

                audio = np.pad(audio, [[0, 0], [self.receptive_field + 1, 0]],
                                'constant') # plus one since we calc loss from second input
                bg_flag = True
                if self.sample_size:
                    while audio.shape[1] > self.place_holder_len:
                        piece = np.expand_dims(audio[:,:self.place_holder_len] , axis=-1)
                        yield (piece, bg_flag)
                        audio = audio[:, self.sample_size:]
                        bg_flag = False
                else:
                    yield (audio, bg_flag)

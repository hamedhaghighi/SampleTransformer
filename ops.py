from __future__ import division

import tensorflow.compat.v1 as tf


def create_adam_optimizer(learning_rate, momentum):
    return tf.train.AdamOptimizer(learning_rate=learning_rate,
                                  epsilon=1e-4)


def create_sgd_optimizer(learning_rate, momentum):
    return tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                      momentum=momentum)


def create_rmsprop_optimizer(learning_rate, momentum):
    return tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                     momentum=momentum,
                                     epsilon=1e-5)


optimizer_factory = {'adam': create_adam_optimizer,
                     'sgd': create_sgd_optimizer,
                     'rmsprop': create_rmsprop_optimizer}


def time_to_batch(value, dilation, name=None):
    with tf.name_scope('time_to_batch'):
        shape = value.shape
        pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
        padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
        reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
        transposed = tf.transpose(reshaped, perm=[1, 0, 2])
        return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])


def batch_to_time(value, dilation, name=None):
    with tf.name_scope('batch_to_time'):
        shape = value.shape
        prepared = tf.reshape(value, [dilation, -1, shape[2]])
        transposed = tf.transpose(prepared, perm=[1, 0, 2])
        return tf.reshape(transposed,
                          [tf.div(shape[0], dilation), -1, shape[2]])


def causal_conv(value, filter_, dilation, name='causal_conv'):
    with tf.name_scope(name):
        filter_width = filter_.shape[0]
        if dilation > 1:
            transformed = time_to_batch(value, dilation)
            conv = tf.nn.conv1d(transformed, filter_, stride=1,
                                padding='VALID')
            restored = batch_to_time(conv, dilation)
        else:
            restored = tf.nn.conv1d(value, filter_, stride=1, padding='VALID')
        # Remove excess elements at the end.
        out_width = value.shape[1] - (filter_width - 1) * dilation
        result = tf.slice(restored,
                          [0, 0, 0],
                          [-1, out_width, -1])
        return result


def mu_law_encode(audio, quantization_channels):
    '''Quantizes waveform amplitudes.'''
    with tf.name_scope('encode'):
        mu = tf.to_float(quantization_channels - 1)
        # Perform mu-law companding transformation (ITU-T, 1988).
        # Minimum operation is here to deal with rare large amplitudes caused
        # by resampling.
        safe_audio_abs = tf.minimum(tf.abs(audio), 1.0)
        magnitude = tf.log1p(mu * safe_audio_abs) / tf.log1p(mu)
        signal = tf.sign(audio) * magnitude
        # Quantize signal to the specified number of levels.
        return tf.cast((signal + 1) / 2 * mu + 0.5 , dtype=tf.int32)

def __linear_quantize(data, q_levels):
    """
    floats in (0, 1) to ints in [0, q_levels-1]
    scales normalized across axis 1
    """
    eps = tf.constant(1e-5, dtype=tf.float64)
    data *= (q_levels - eps)
    data += eps/2
    data =tf.cast(data , dtype=tf.int32)
    return data

def _one_hot(input_batch , q_levels):
        '''One-hot encodes the waveform amplitudes.

        This allows the definition of the network as a categorical distribution
        over a finite set of possible amplitudes.
        '''
        B, _ , _ = input_batch.shape
        with tf.name_scope('one_hot_encode'):
            encoded = tf.one_hot(
                input_batch,
                depth=q_levels,
                dtype=tf.float32)
            shape = [B, -1, q_levels]
            encoded = tf.reshape(encoded, shape)
        return encoded

def quantize(data, q_levels, q_type):
    """
    One of 'linear', 'a-law', 'mu-law' for q_type.
    """
    data = tf.cast(data, tf.float64)
    data = data - tf.reduce_min(data, axis = 1, keepdims=True)
    data = data / tf.reduce_max(data, axis = 1, keepdims=True)
    if q_type == 'linear':
        return __linear_quantize(data, q_levels)
    if q_type == 'a-law':
        return __a_law_quantize(data)
    if q_type == 'mu-law':
        return mu_law_encode(data)
    raise NotImplementedError

def mu_law_decode(output, quantization_channels):
    '''Recovers waveform from quantized values.'''
    with tf.name_scope('decode'):
        mu = quantization_channels - 1
        # Map values back to [-1, 1].
        signal = 2 * (tf.to_float(output) / mu) - 1
        # Perform inverse of mu-law transformation.
        magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
        return tf.sign(signal) * magnitude

class Conv1d(object):
    def __init__(self, out_channel, kernel_size, name, dilation=1, padding = 'VALID'):
        self.kernel_size = kernel_size
        self.out_channel = out_channel
        self.dilation = dilation
        self.name = name
        self.padding = padding
    
    def time_to_batch(self, value, dilation, name=None):
        with tf.name_scope('time_to_batch'):
            shape = tf.shape(value)
            pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
            padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
            reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
            transposed = tf.transpose(reshaped, perm=[1, 0, 2])
            return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])


    def batch_to_time(self, value, dilation, name=None):
        with tf.name_scope('batch_to_time'):
            shape = tf.shape(value)
            prepared = tf.reshape(value, [dilation, -1, shape[2]])
            transposed = tf.transpose(prepared, perm=[1, 0, 2])
            return tf.reshape(transposed,
                          [tf.div(shape[0], dilation), -1, shape[2]])

    def __call__(self, value):
        trainable = True
        if len(tf.get_variable_scope().name.split('/')) > 1:
            block_scope = tf.get_variable_scope().name.split('/')[1]
            if block_scope == 'waveblock_last' and self.name=='post_conv':
                trainable = False
        W = tf.get_variable(self.name, shape=(self.kernel_size, value.shape[2].value, self.out_channel), dtype= tf.float32, trainable=trainable)
        if self.dilation > 1:
            transformed = self.time_to_batch(value, self.dilation)
            conv = tf.nn.conv1d(transformed, W, stride=1, padding=self.padding)
            restored = self.batch_to_time(conv, self.dilation)
        else:
            restored = tf.nn.conv1d(value, W, stride=1, padding=self.padding)
        # Remove excess elements at the end.
        
        out_width = value.shape[1].value - (self.kernel_size - 1) * self.dilation
        result = tf.slice(restored,
                          [0, 0, 0],
                          [-1, out_width, -1])
        return result



from __future__ import print_function

import json
import os
import sys
import time
from datetime import datetime
import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.python import debug as tf_debug
from read_audio import AudioReader
from ops import optimizer_factory
from hierarchy_tf import SampleTransformer
import numpy as np
import blocksparse as bs
import shutil

STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
EPSILON = 0.001
bs.set_entropy()

def save(saver, sess, logdir, step, best_val_loss , load_type):
    model_name = load_type.join('.ckpt')
    checkpoint_path = os.path.join(logdir, model_name)
    
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step, latest_filename=load_type)
    if load_type == 'best':
        np.save(os.path.join(logdir, 'best_val_loss.npy'), np.array([best_val_loss]))
    print('{} checkpoint saved to {} ...\n'.format(load_type, logdir), end="")


def load(saver, sess, logdir, load_type):
    print("Trying to restore saved checkpoints from {} ...\n".format(logdir),
          end="")
    if load_type is not None and os.path.exists(os.path.join(logdir, load_type)):
                ckpt = tf.train.get_checkpoint_state(
                    logdir, latest_filename=load_type)
                if ckpt and ckpt.model_checkpoint_path:
                    print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
                    print('model restored')
                    print("  Global step was: {}".format(global_step))
                    print("  Restoring...", end="")
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    if os.path.exists(os.path.join(logdir, 'best_val_loss.npy')):
                        best_val_loss = np.load(os.path.join(logdir, 'best_val_loss.npy'))[0]
                    print(" Done.")
                    return global_step , best_val_loss
    elif os.path.exists(os.path.join(logdir)):
        user_input = input('should we start from scratch?(y/n) (Warning: Yes will remove all checkpoints)')
        if not user_input.startswith('y'):
            exit(0)
        else:
            shutil.rmtree(logdir)
            # shutil.rmtree(vaegan_checkpoint_dir)
            return None, None
    else:
        print("No checkpoint found.\n")
        return None, None


def get_default_logdir(logdir_root):
    logdir = os.path.join(logdir_root, 'train', STARTED_DATESTRING)
    return logdir



class Train():
    def __init__(self, args):
        self.down_sampling_rates = [16, 16]
        self.kernel_size = 2
        self.args = args
        self.dilation_rates = [2**i for i in range(args.wl + 1)]*2
        self.receptive_field = self.calc_pad(self.args)
        try:
            directories = self.validate_directories(self.args)
        except ValueError as e:
            print("Some arguments are wrong:")
            print(str(e))
            return

        self.logdir = directories['logdir']

        # Even if we restored the model, we will treat it as new training
        # if the trained model is written into an arbitrary location.

        ### modifying samle size to become square complete
        self.args.sample_size = self.args.sample_size - self.receptive_field//2
        # Create network.
        self.net_train = SampleTransformer(self.down_sampling_rates, self.dilation_rates, self.kernel_size, self.receptive_field, self.args)
        # self.net_val = SampleTransformer(self.down_sampling_rates, self.dilation_rates, self.kernel_size, self.receptive_field, self.args)
        # Load raw waveform from VCTK corpus.
        
        with tf.name_scope('create_inputs'):
            # Allow silence trimming to be skipped by specifying a threshold near
            # zero.
            silence_threshold = self.args.silence_threshold if self.args.silence_threshold > \
                                                        EPSILON else None
            gc_enabled = self.args.gc_channels is not None
            self.reader = AudioReader(
                args.data_dir,
                sample_rate=0,
                batch_size=self.args.batch_size,
                gc_enabled=gc_enabled,
                receptive_field= self.receptive_field, # TODO: change receiptive field
                sample_size=self.args.sample_size,
                silence_threshold=silence_threshold)
            
            self.audio_batch, self.begin = self.reader.get_input_placeholder()

        self.trainData_iter = self.reader.get_data_iterator('train')
        self.valData_iter = self.reader.get_data_iterator('val')

        if args.l2_regularization_strength == 0:
            args.l2_regularization_strength = None
        
        self.g_step = tf.placeholder(dtype=tf.int32 , shape=None , name = 'step')
        self.loss_train = self.net_train.loss(self.audio_batch,
                        self.begin,
                        self.g_step,
                        True,
                        l2_regularization_strength=args.l2_regularization_strength)
        
        bs.clear_bst_constants()
        params = tf.trainable_variables()
        grads  = bs.gradients(self.loss_train, params)
        self.global_norm, self.norm_scale = bs.clip_by_global_norm(grads, grad_scale=1.0, clip_norm=1.0)
        adam = bs.AdamOptimizer(learning_rate=self.args.learning_rate, norm_scale=self.norm_scale, grad_scale=1.0, fp16=False)
        self.train_op = adam.apply_gradients(zip(grads, params))
        self.loss_val = self.net_train.loss(self.audio_batch,
                        self.begin,
                        self.g_step,
                        False,
                        l2_regularization_strength=args.l2_regularization_strength)
        # Restoring ...
        with tf.variable_scope('memroy' , reuse=True):
            memory = tf.get_variable('mem')
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
        var_list = tf.trainable_variables() + [memory]
        self.saver = tf.train.Saver(var_list=var_list, max_to_keep=args.max_checkpoints)
        try:
            self.saved_global_step , self.best_val_loss = load(self.saver, self.sess, self.logdir, self.args.load_type)
            if self.saved_global_step is None:
                # The first training step will be saved_global_step + 1,
                # therefore we put -1 here for new or overwritten trainings.
                self.saved_global_step = 0
                self.best_val_loss = np.inf
        except:
            print("Something went wrong while restoring checkpoint. "
                "We will terminate training to avoid accidentally overwriting "
                "the previous model.")
            raise
        self.summary_writer = tf.summary.FileWriter(os.path.join(self.logdir, STARTED_DATESTRING))
       
        
    # Set up session
    def calc_pad(self, args):
        wavenet_pad = sum(self.dilation_rates)*(self.kernel_size - 1)
        if self.args.scalar_input:
            return (wavenet_pad + self.args.init_kernel_size - 1)*2
        return (wavenet_pad + self.kernel_size - 1)*2


    def validate_directories(self, args):
        """Validate and arrange directory related arguments."""
        logdir_root = self.args.logdir_root

        logdir = self.args.logdir
        if logdir is None:
            logdir = get_default_logdir(logdir_root)
            
        else:
            logdir = os.path.join(logdir_root, logdir)
        print('Using default logdir: {}'.format(logdir))
        return {
            'logdir': logdir,
            'logdir_root': logdir_root,
        }

    def get_training_steps(self):
        return self.reader.get_len_dataset() if not self.args.fast else (3 , 2)
    
    def get_global_step(self):
        return self.saved_global_step

    def summarize(self, tag, value, step):
        summary_str = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value), ])
        self.summary_writer.add_summary(summary_str, step)

    def train(self, n_steps, is_train):
        
        # Saver for storing checkpoints of the model.
        step = None
        data_iter = self.trainData_iter if is_train else self.valData_iter
        total_loss = np.array([])
        step = self.saved_global_step
        for _ in range(n_steps):
            data_batch , bg = next(data_iter)
            # import pdb; pdb.set_trace()
            feed_dict={self.audio_batch:data_batch, self.begin: bg}
            if is_train:
                loss_value, _ = self.sess.run([self.loss_train, self.train_op], feed_dict=feed_dict)
            else:
                loss_value = self.sess.run(self.loss_val, feed_dict=feed_dict)
            total_loss = np.append(total_loss, loss_value)
            print_every = 1  if self.args.fast else self.args.print_every
            if step%(print_every)==0 and is_train:
                print('Train: step {:d} - loss = {:.3f}'.format(step, total_loss.mean()))
                self.summarize('train' , total_loss.mean(), step)
            step = (step + 1) if is_train else step
        if not is_train:
            print('Validation: step {:d} - loss = {:.3f}'.format(step, total_loss.mean()))
            self.summarize('validation', total_loss.mean(), step)  
        ## save model ...        
        if not is_train: 
            if total_loss.mean() < self.best_val_loss:
                self.best_val_loss = total_loss.mean()
                save(self.saver, self.sess, self.logdir, step, self.best_val_loss, 'best')
            save(self.saver, self.sess, self.logdir, step, self.best_val_loss, 'last')
        self.saved_global_step = step
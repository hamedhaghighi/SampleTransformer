
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
from options import *
import blocksparse as bs

STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
bs.set_entropy()

def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None


def get_default_logdir(logdir_root):
    logdir = os.path.join(logdir_root, 'train', STARTED_DATESTRING)
    return logdir


def validate_directories(args):
    """Validate and arrange directory related arguments."""

    # Validation
    if args.logdir and args.logdir_root:
        raise ValueError("--logdir and --logdir_root cannot be "
                         "specified at the same time.")

    if args.logdir and args.restore_from:
        raise ValueError(
            "--logdir and --restore_from cannot be specified at the same "
            "time. This is to keep your previous model from unexpected "
            "overwrites.\n"
            "Use --logdir_root to specify the root of the directory which "
            "will be automatically created with current date and time, or use "
            "only --logdir to just continue the training from the last "
            "checkpoint.")

    # Arrangement
    logdir_root = args.logdir_root
    if logdir_root is None:
        logdir_root = LOGDIR_ROOT

    logdir = args.logdir
    if logdir is None:
        logdir = get_default_logdir(logdir_root)
        print('Using default logdir: {}'.format(logdir))

    restore_from = args.restore_from
    if restore_from is None:
        # args.logdir and args.restore_from are exclusive,
        # so it is guaranteed the logdir here is newly created.
        restore_from = logdir

    return {
        'logdir': logdir,
        'logdir_root': args.logdir_root,
        'restore_from': restore_from
    }

def calc_pad(args):
    wavenet_pad = sum(dilation_rates)*(kernel_size - 1)
    if args.scalar_input:
        return (wavenet_pad + args.init_kernel_size - 1)*2
    return (wavenet_pad + kernel_size - 1)*2

def main():
    args = get_arguments()
    dilation_rates = [2**i for i in range(args.wl + 1)]*2
    receptive_field = calc_pad(args)
    try:
        directories = validate_directories(args)
    except ValueError as e:
        print("Some arguments are wrong:")
        print(str(e))
        return

    logdir = directories['logdir']
    restore_from = directories['restore_from']

    # Even if we restored the model, we will treat it as new training
    # if the trained model is written into an arbitrary location.
    is_overwritten_training = logdir != restore_from

    with open(args.wavenet_params, 'r') as f:
        wavenet_params = json.load(f)

    ### modifying samle size to become square complete
    args.sample_size = args.sample_size - receptive_field//2
    # Create network.
    net = SampleTransformer(down_sampling_rates, dilation_rates, kernel_size, receptive_field, args)
    # Load raw waveform from VCTK corpus.
    with tf.name_scope('create_inputs'):
        # Allow silence trimming to be skipped by specifying a threshold near
        # zero.
        silence_threshold = args.silence_threshold if args.silence_threshold > \
                                                      EPSILON else None
        gc_enabled = args.gc_channels is not None
        reader = AudioReader(
            args.data_dir,
            sample_rate=wavenet_params['sample_rate'],
            batch_size=args.batch_size,
            gc_enabled=gc_enabled,
            receptive_field= receptive_field, # TODO: change receiptive field
            sample_size=args.sample_size,
            silence_threshold=silence_threshold)
        
        audio_batch, begin = reader.get_input_placeholder()


    if args.l2_regularization_strength == 0:
        args.l2_regularization_strength = None
    
    g_step = tf.placeholder(dtype=tf.int32 , shape=None , name = 'step')

    loss = net.loss(audio_batch,
                    begin,
                    g_step,
                    l2_regularization_strength=args.l2_regularization_strength)
    bs.clear_bst_constants()
    import pdb; pdb.set_trace()
    params = tf.trainable_variables()
    grads  = bs.gradients(loss, params)
    global_norm, norm_scale = bs.clip_by_global_norm(grads, grad_scale=1.0, clip_norm=1.0)
    adam = bs.AdamOptimizer(learning_rate=args.learning_rate, norm_scale=norm_scale, grad_scale=1.0, fp16=False)
    train_op = adam.apply_gradients(zip(grads, params))
    # # Set up logging for TensorBoard.
    # writer = tf.summary.FileWriter(logdir)
    # writer.add_graph(tf.get_default_graph())
    # run_metadata = tf.RunMetadata()
    # summaries = tf.summary.merge_all()

    # Set up session
   
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    init = tf.global_variables_initializer()
    sess.run(init)

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=args.max_checkpoints)

    try:
        saved_global_step = load(saver, sess, restore_from)
        if is_overwritten_training or saved_global_step is None:
            # The first training step will be saved_global_step + 1,
            # therefore we put -1 here for new or overwritten trainings.
            saved_global_step = -1

    except:
        print("Something went wrong while restoring checkpoint. "
              "We will terminate training to avoid accidentally overwriting "
              "the previous model.")
        raise

    step = None
    last_saved_step = saved_global_step
    trainData_iter = reader.get_data_iterator('train')
    valData_iter = reader.get_data_iterator('val')
    try:
        for step in range(saved_global_step + 1, args.num_steps):
            start_time = time.time()
            # if args.store_metadata and step % 50 == 0:
            #     # Slow run that stores extra information for debugging.
            #     print('Storing metadata')
            #     run_options = tf.RunOptions(
            #         trace_level=tf.RunOptions.FULL_TRACE)
            #     # summary, loss_value, _ = sess.run(
            #     #     [summaries, loss, train_op],
            #     #     options=run_options,
            #     #     run_metadata=run_metadata)
            #     # writer.add_summary(summary, step)
            #     # writer.add_run_metadata(run_metadata,
            #                             # 'step_{:04d}'.format(step))
            #     tl = timeline.Timeline(run_metadata.step_stats)
            #     timeline_path = os.path.join(logdir, 'timeline.trace')
            #     with open(timeline_path, 'w') as f:
            #         f.write(tl.generate_chrome_trace_format(show_memory=True))
            # else:
            data_batch , bg = next(trainData_iter)
            loss_value, _, gn, ns = sess.run([loss, train_op, global_norm, norm_scale], feed_dict={audio_batch:data_batch, begin: bg})
            import pdb; pdb.set_trace()
                
                # writer.add_summary(summary, step)

            duration = time.time() - start_time
            # print('step {:d} - loss = {:.3f}, ({:.3f} sec/step)'
            #       .format(step, loss_value, duration))

            if step % args.checkpoint_every == 0:
                save(saver, sess, logdir, step)
                last_saved_step = step

    except KeyboardInterrupt:
        # Introduce a line break after ^C is displayed so save message
        # is on its own line.
        print()
    finally:
        if step > last_saved_step:
            save(saver, sess, logdir, step)


if __name__ == '__main__':
    main()

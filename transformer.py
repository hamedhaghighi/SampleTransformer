#!/usr/bin/env python

'''
Example of the blocksparse transformer on enwik8.

To download data:

wget http://mattmahoney.net/dc/enwik8.zip
unzip enwik8.zip -d /tmp
'''

import argparse
import numpy       as np
import tensorflow  as tf
import blocksparse as bs
# from mpi4py import MPI
from attention import blocksparse_attention_impl
from attention import attention_impl

def layernorm(x, scope, epsilon=1e-5, relu=False):
    """
    normalize state vector to be zero mean / unit variance + learned scale/shift
    """
    n_state = x.shape[-1].value
    with tf.variable_scope(scope):
        gain = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1.0))
        bias = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0.0))
        return bs.layer_norm(x, gain, bias, axis=-1, epsilon=epsilon, relu=relu)


def conv1d(x, scope, nf, std=0.02, relu=False, fast_gelu=False):
    with tf.variable_scope(scope):
        nx    = x.shape[-1].value
        ndims = x.shape.ndims

        # Note: param initializers are not particularly well tuned in this code
        w = tf.get_variable("w", [nx, nf], initializer=tf.random_normal_initializer(stddev=std))
        b = tf.get_variable("b", [    nf], initializer=tf.constant_initializer(0.0))

        # if hps.float16:
        #     # We delay weight casting till just before use to minimize memory footprint.
        #     # In recompute mode these casts are released just after use on forward pass,
        #     # then remade on the recompute pass.
        #     with tf.control_dependencies([x.op]):
        #         # By setting dx_dtype to float16 we prevent useless casting back to fp32 in the backwards pass.
        #         # Our all-reduce and fused optimizers can accept fp16 natively.
        #         w = bs.float_cast(w, dtype=tf.float16, dx_dtype=tf.float16)

        # merge context and batch dims for more efficient matmul
        if ndims > 2:
            y_shape = tf.concat([tf.shape(x)[: ndims - 1], [nf]], axis=0)
            x = tf.reshape(x, [-1, nx])

        y = tf.matmul(x, w)

        # avoid atomics in bias grad, but be careful as tf handles temp memory badly in the presense of async ops like all-reduce
        y = bs.bias_relu(y, b, relu=relu, fast_gelu=fast_gelu, atomics=False)

        if ndims > 2:
            y = tf.reshape(y, y_shape)

        return y

# Fine sparse structure
# Within each block this mask is applied to force the softmax output to zero where the mask is zero
# This is defined as a callback to avoid having to instantiate the full mask in memory at one time.
# The callback value is immediately converted to a bit mask internally.
def causal_subblock_mask(blk_shape, head_idx, query_idx, key_idx, blk_idx):
    """Prohibit positions in sub-blocks from attending to indices in the future.
    Note: query_idx and key_idx are absolute indices rather than relative to
    each block.
    """
    mask = np.ones(blk_shape, dtype=np.bool)
    if query_idx == key_idx:
        for q, k in np.ndindex(blk_shape):
            if k > q:
                mask[q, k] = 0
    return mask

# Coarse sparse structure
# Only layout[q,k] == 1 blocks are computed and materialized in memory
# Block sizes of 8, 16, 32 and 64 are supported on volta fp16 tensorcores (64 being most appropriate for dense attention)
# Only blocksize 32 currently supported in fp32 on other gpus (sm >= 3.5).
def get_blocksparse_transformer(n_timesteps, n_heads):
    blocksize = 64 if hps.float16 else 32
    n_time_blocks = n_timesteps // blocksize
    # The block layout can also include a head dimension if you don't want the same layout shared by all heads.
    # Each head just has to have the same number of active blocks (but you can always mask them away).
    layout = np.ones([n_time_blocks, n_time_blocks], dtype=np.bool)
    # No query blocks may attend to key blocks in the future.
    # Much more elaborate structures can be defined here aside from the usual lower triangular.
    for q_idx, k_idx in np.ndindex(n_time_blocks, n_time_blocks):
        if k_idx > q_idx:
            layout[q_idx, k_idx] = 0
    bst = bs.BlocksparseTransformer(layout, block_size=blocksize, mask_callback=causal_subblock_mask, heads=n_heads)
    return bst

# very simple to use recompute decorator.  Be sure to pair with bs.gradients() for it to work
@bs.recomputable
def transformer_block(x, memory,  scope, mode , dp , mlp_ratio, dropout_cache,train=True):
    """
    core component of transformer
    performs attention + residual mlp + layer normalization
    """
    
    T = x.shape[1].value
    if np.floor(np.sqrt(T)) == np.sqrt(T):
        local_attn_ctx = np.sqrt(T).astype(np.int)
    elif np.floor(np.sqrt(T*2)) == np.sqrt(T*2):
        local_attn_ctx = np.sqrt(T*2).astype(np.int)
    else:
        raise ValueError('SIK TIR')
    if memory != None:
        B , _ , T, C = memory.shape
        memory = tf.reshape(memory , shape=(B, T*3, C))
        x = tf.concat([x, memory], axis=1)
    n_state = x.shape[-1].value

    with tf.variable_scope(scope):

        h = layernorm(x, "norm_a")

        q = conv1d(h[:, :T], 'proj_q', n_state)
        k = conv1d(h, 'proj_k', n_state)
        v = conv1d(h, 'proj_v', n_state)

        # only need to create one bst per config
        # we could pass this in as an external param but I like to keep the code more local
        a = blocksparse_attention_impl(q, k, v, heads=4, attn_mode=mode, local_attn_ctx=local_attn_ctx, blocksize=32, recompute=True)
        a = conv1d(a, 'proj_a', n_state, std=0.02/6)

        if train and dp > 0.0:
            # preserve the dropout mask through recompute
            key = scope + "_dropout_a"
            a, dropout_cache[key] = bs.dropout(a, keep_prob=1.0 - dp, mask=dropout_cache.get(key))

        # many basic tf ops are about half as fast as they should be in fp16
        x = bs.add(x[:,:T], a)

        m = layernorm(x, "norm_m")

        # fast_gelu: x * sigmoid(1.702 * x)
        m = conv1d(m, 'proj_m1', n_state * mlp_ratio, fast_gelu=True)
        m = conv1d(m, 'proj_m2', n_state)

        if train and dp > 0.0:
            # preserve the dropout mask through recompute
            key = scope + "_dropout_m"
            m, dropout_cache[key] = bs.dropout(m, keep_prob=1.0 - dp, mask=dropout_cache.get(key))

        return bs.add(x, m)


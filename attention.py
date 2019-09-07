import sys
import numpy as np
import tensorflow as tf
import blocksparse as bs
from blocksparse import BlocksparseTransformer
from sa_utils import shape_list, recomputable

global bst_dict 
bst_dict= {}
global dropout_cache
dropout_cache = dict()

def get_attn_mask(n, attn_mode, local_attn_ctx=None):
    if attn_mode == 'all':
        b = tf.matrix_band_part(tf.ones([n, n]), -1, 0)
    elif attn_mode == 'local':
        bandwidth = local_attn_ctx
        ctx = tf.minimum(n - 1, bandwidth - 1)
        b = tf.matrix_band_part(tf.ones([n, n]), ctx, 0)
    elif attn_mode == 'strided':
        stride = local_attn_ctx
        x = tf.reshape(tf.range(n, dtype=tf.int32), [n, 1])
        y = tf.transpose(x)
        z = tf.zeros([n, n], dtype=tf.int32)
        q = z + x
        k = z + y
        c1 = q >= k
        c2 = tf.equal(tf.floormod(q - k, stride), 0)
        c3 = tf.logical_and(c1, c2)
        b = tf.cast(c3, tf.float32)
    else:
        raise ValueError('Not yet implemented')
    b = tf.reshape(b, [1, 1, n, n])
    return b


def strided_transpose(x, n_ctx, local_attn_ctx, blocksize):
    bT_ctx = n_ctx // local_attn_ctx
    assert bT_ctx % blocksize == 0, f'{bT_ctx}, {blocksize}'
    n, t, embd = shape_list(x)
    x = tf.reshape(x, [n, bT_ctx, local_attn_ctx, embd])
    x = tf.transpose(x, [0, 2, 1, 3])
    x = tf.reshape(x, [n, t, embd])
    return x


def split_heads(x, n):
    return tf.transpose(split_states(x, n), [0, 2, 1, 3])


def merge_heads(x):
    return merge_states(tf.transpose(x, [0, 2, 1, 3]))


def split_states(x, n):
    """
    reshape (batch, pixel, state) -> (batch, pixel, head, head_state)
    """
    x_shape = shape_list(x)
    m = x_shape[-1]
    new_x_shape = x_shape[:-1] + [n, m // n]
    return tf.reshape(x, new_x_shape)


def merge_states(x):
    """
    reshape (batch, pixel, head, head_state) -> (batch, pixel, state)
    """
    x_shape = shape_list(x)
    new_x_shape = x_shape[:-2] + [np.prod(x_shape[-2:])]
    return tf.reshape(x, new_x_shape)


@recomputable('attention_impl')
def attention_impl(q, k, v, heads, attn_mode, local_attn_ctx=None):
    q = split_heads(q, heads)
    k = split_heads(k, heads)
    v = split_heads(v, heads)
    n_timesteps = shape_list(k)[2]
    mask = tf.to_float(get_attn_mask(n_timesteps, attn_mode, local_attn_ctx))
    w = tf.matmul(q, k, transpose_b=True)
    scale_amount = 1.0 / np.sqrt(shape_list(q)[-1])
    orig_dtype = q.dtype
    if orig_dtype == tf.float16:
        w = tf.cast(w, tf.float32)
    w = w * scale_amount
    w = w * mask + -1e9 * (1 - mask)
    w = tf.nn.softmax(w)
    w = tf.cast(w, orig_dtype)
    a = tf.matmul(w, v)
    a = merge_heads(a)
    return a


#@recomputable('blocksparse_attention_impl')
def blocksparse_attention_impl(q, k, v, heads, attn_mode, local_attn_ctx=None,
                               blocksize=32, num_verts=None, vertsize=None):
    global bst_dict
    n_ctx = shape_list(q)[1]
    assert shape_list(v)[1]%n_ctx == 0
    if attn_mode == 'strided':
        # Strided attention is implemented on the transposed matrix to provide greater block sparsity
        q = strided_transpose(q, n_ctx, local_attn_ctx, blocksize)
        k = strided_transpose(k, n_ctx, local_attn_ctx, blocksize)
        v = strided_transpose(v, n_ctx, local_attn_ctx, blocksize)
    n_state = shape_list(q)[-1] // heads
    key = f'{local_attn_ctx}' + f'{n_ctx}' + attn_mode
    if key not in bst_dict:
        bst_dict[key]= get_blocksparse_obj(n_ctx, heads, attn_mode, blocksize, local_attn_ctx, num_verts, vertsize, shape_list(v)[1]//n_ctx - 1)
    bst = bst_dict[key]
    scale_amount = tf.cast(1.0 / np.sqrt(n_state), tf.float32)
    w = bst.query_key_op(q, k)
    w = bst.masked_softmax(w, scale=scale_amount)
    a = bst.weight_value_op(w, v)
    if attn_mode == 'strided':
        n, t, embd = shape_list(a)
        bT_ctx = n_ctx // local_attn_ctx
        a = tf.reshape(a, [n, local_attn_ctx, bT_ctx, embd])
        a = tf.transpose(a, [0, 2, 1, 3])
        a = tf.reshape(a, [n, t, embd])
    return a


def get_blocksparse_obj(n_ctx, n_heads, attn_mode, blocksize=32, local_attn_ctx=None, num_verts=4, vertsize=1, n_memory=0):
    '''Defines the block-level sparsity pattern in the attention matrix. Enabled blocks
    will have the callback called on them in order to define a positionwise sparsity mask.'''
    n_bctx = n_ctx // blocksize
    layout = np.ones([n_bctx, n_bctx], dtype=np.bool)
    extra_diagonals = None
    block_chunks = None

    if attn_mode in ['all', 'fixed']:
        pass
    elif attn_mode == 'local':
        assert local_attn_ctx % blocksize == 0
        extra_diagonals = local_attn_ctx // blocksize
    elif attn_mode == 'strided':
        bT_ctx = n_ctx // local_attn_ctx
        assert bT_ctx % blocksize == 0
        block_chunks = bT_ctx // blocksize
    else:
        raise ValueError(f'attn mode {attn_mode} invalid')

    if attn_mode == 'fixed':
        assert n_heads % num_verts == 0
        lctx = local_attn_ctx
        stride = lctx // blocksize
        assert vertsize <= stride
        assert stride % vertsize == 0
        indices = [i for i in range(stride - 1, -1, -1)]
        indices = np.array(indices).reshape([-1, vertsize])
        
        if num_verts == 1:
            layout = np.zeros([n_bctx, n_bctx], dtype=np.bool)
            for idx in indices[0]:
                layout[:, idx::stride] = 1
            for q_idx in range(n_bctx):
                # Each thing can attend to its local block
                row = q_idx // stride
                layout[q_idx, row * stride:(row + 1) * stride] = 1
                # Any query cannot attend to keys above it
                layout[q_idx, q_idx + 1:] = 0
        else:
            layouts = []
            indices = indices[:num_verts]
            for h in range(n_heads):
                layout = np.zeros([n_bctx, n_bctx], dtype=np.bool)
                subindices = indices[h % num_verts]
                for idx in subindices:
                    layout[:, idx::stride] = 1
                for q_idx in range(n_bctx):
                    # Each position can attend to its local block
                    row = q_idx // stride
                    layout[q_idx, row * stride:(row + 1) * stride] = 1
                    # Any query cannot attend to keys above it
                    layout[q_idx, q_idx + 1:] = 0
                layouts.append(layout)
            layout = np.array(layouts)
    else:
        for q_idx, k_idx in np.ndindex(n_bctx, n_bctx):
            if k_idx > q_idx:
                layout[q_idx, k_idx] = 0
            if extra_diagonals and k_idx + extra_diagonals < q_idx:
                layout[q_idx, k_idx] = 0
            if block_chunks is not None:
                layout[q_idx, k_idx] = 0
                offset = q_idx % block_chunks
                if k_idx + offset >= q_idx and k_idx <= q_idx:
                    layout[q_idx, k_idx] = 1
    if attn_mode == 'all' and n_memory > 0:
        layout = np.concatenate([layout, np.ones((n_bctx , n_memory*n_bctx), dtype=np.bool)], axis = 1)
    bst = BlocksparseTransformer(layout, block_size=blocksize,
                                 mask_callback=get_callback(attn_mode, local_attn_ctx),
                                 heads=n_heads)
    return bst


def get_callback(attn_mode, local_attn_ctx=None):
    '''Defines a function which returns the positionwise sparsity pattern for every block
    that is enabled in the blocksparse object
    '''
    def cb(blk_shape, head_idx, qry_idx, key_idx, blk_idx):
        mask = np.ones(blk_shape, dtype=np.bool)

        # on the diagonal blocks mask out the upper diagonal
        if qry_idx == key_idx:
            for q, k in np.ndindex(blk_shape):
                if k > q:
                    mask[q, k] = 0
        if attn_mode in ['all', 'strided', 'fixed']:
            return mask
        if attn_mode == 'local':
            bandwidth = local_attn_ctx
            # convert group indices to absolute indices and mask
            # according to that
            q_pos = blk_shape[0] * qry_idx
            k_pos = blk_shape[1] * key_idx
            for q, k in np.ndindex(blk_shape):
                q_ = q + q_pos
                k_ = k + k_pos
                if k_ > q_ or k_ + bandwidth <= q_:
                    mask[q, k] = 0
            return mask
        raise ValueError
    return cb


def layernorm(x, scope, epsilon=1e-5, relu=False):
    """
    normalize state vector to be zero mean / unit variance + learned scale/shift
    """
    n_state = x.shape[-1].value
    with tf.variable_scope(scope):
        gain = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1.0))
        bias = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0.0))
        return bs.layer_norm(x, gain, bias, axis=-1, epsilon=epsilon, relu=relu, use_tf=True)


def conv1d(x, scope, nf, std=0.02, relu=False, fast_gelu=False):
    with tf.variable_scope(scope):
        nx    = x.shape[-1].value
        ndims = x.shape.ndims

        # Note: param initializers are not particularly well tuned in this code
        w = tf.get_variable("w", [nx, nf], initializer=tf.random_normal_initializer(stddev=std), dtype=tf.float32)
        b = tf.get_variable("b", [    nf], initializer=tf.constant_initializer(0.0), dtype=tf.float32)

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
            y_shape = tf.concat([x.shape[: ndims - 1], [nf]], axis=0)
            x = tf.reshape(x, [-1, nx])

        y = tf.matmul(x, w)

        # avoid atomics in bias grad, but be careful as tf handles temp memory badly in the presense of async ops like all-reduce
        y = bs.bias_relu(y, b, relu=relu, fast_gelu=fast_gelu, atomics=False)

        if ndims > 2:
            y = tf.reshape(y, y_shape)

        return y




@bs.recomputable
def transformer_block(x, memory,  scope, mode , dp , mlp_ratio, train=True):
    """
    core component of transformer
    performs attention + residual mlp + layer normalization
    """
    global dropout_cache
    T = x.shape[1].value
    if np.floor(np.sqrt(T)) == np.sqrt(T):
        local_attn_ctx = np.sqrt(T).astype(np.int)
    elif np.floor(np.sqrt(T*2)) == np.sqrt(T*2):
        local_attn_ctx = np.sqrt(T*2).astype(np.int)
    else:
        raise ValueError('SIK TIR')
    if memory != None:
        B , N , T, C = memory.shape
        memory = tf.reshape(memory , shape=(B, T*N, C))
        x = tf.concat([x, memory], axis=1)
    n_state = x.shape[-1].value

    with tf.variable_scope(scope):

        h = layernorm(x, "norm_a")
        # h = x
        q = conv1d(h[:, :T], 'proj_q', n_state)
        k = conv1d(h, 'proj_k', n_state)
        v = conv1d(h, 'proj_v', n_state)

        # only need to create one bst per config
        # we could pass this in as an external param but I like to keep the code more local
        a = blocksparse_attention_impl(q, k, v, heads=4, attn_mode=mode, local_attn_ctx=local_attn_ctx, blocksize=32)
        a = conv1d(a, 'proj_a', n_state, std=0.02/6) # TODO: correct num layers

        if train and dp > 0.0:
            # preserve the dropout mask through recompute
            key = scope + "_dropout_a"
            a, dropout_cache[key] = bs.dropout(a, keep_prob=1.0 - dp, mask=dropout_cache.get(key))

        # many basic tf ops are about half as fast as they should be in fp16
        x = bs.add(x[:,:T], a)
        m = layernorm(x, "norm_m")
        # m=x
        # fast_gelu: x * sigmoid(1.702 * x)
        m = conv1d(m, 'proj_m1', n_state * mlp_ratio, fast_gelu=True)
        m = conv1d(m, 'proj_m2', n_state)

        if train and dp > 0.0:
            # preserve the dropout mask through recompute
            key = scope + "_dropout_m"
            m, dropout_cache[key] = bs.dropout(m, keep_prob=1.0 - dp, mask=dropout_cache.get(key))

        return bs.add(x, m)

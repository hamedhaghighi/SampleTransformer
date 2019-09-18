import pdb
import tensorflow as tf
import numpy as np
from attention import transformer_block
# import tensorflow.keras.layers as layers 
from ops import quantize, _one_hot, Conv1d

def mpad(tensor, dilation_rate, kernel_size):
    return tf.pad(tensor, [[0, 0], [dilation_rate * (kernel_size - 1), 0],[0,0]])
    
class WaveNetBlock():
    def __init__(self, in_channels, intermediate_channels, kernel_size, dilation_rate , scope):
        self.scope = scope
        self.dilation_rate = dilation_rate
        self.kernel_size = kernel_size
        self.conv = Conv1d(intermediate_channels*2, kernel_size=self.kernel_size, name='pre_conv', dilation=dilation_rate)
        self.post_linear = Conv1d(in_channels, kernel_size=1, name='post_conv')
        self.skip_conv = Conv1d(in_channels, kernel_size=1, name='skip_conv')
        

    def GLU(self, x):
        _, _, C = x.shape
        g, f = tf.split(x, [C.value//2, C.value//2], axis = 2)
        return tf.sigmoid(g) * tf.tanh(f)

    def forward(self, x):
        with tf.variable_scope(self.scope):
            skip_out = self.GLU(self.conv(x))
            input_cut = x.shape[1].value - skip_out.shape[1].value
            y = tf.slice(x, [0, input_cut, 0], [-1, -1, -1])
            return y + self.post_linear(skip_out), self.skip_conv(skip_out)


class WaveNet():
    def __init__(self, in_channels, intermediate_channels, kernel_size, init_kernel_size, output_width, scope, dilation_rates=None):
        self.scope = scope
        self.output_width = output_width
        self.kernel_size = kernel_size
        self.pre_net = Conv1d(in_channels, kernel_size=init_kernel_size, name='pre_net')
        self.wave_blocks = []
        for (i, d) in enumerate(dilation_rates):
            scope = f'waveblock_{i}' if i < len(dilation_rates) - 1 else 'waveblock_last'
            self.wave_blocks.append(WaveNetBlock(in_channels, intermediate_channels, kernel_size, d, scope))
        self.post_net1 = Conv1d(in_channels, kernel_size=1, name='post_net1')
        self.post_net2 = Conv1d(in_channels, kernel_size=1, name='post_net2')

    def forward(self, x):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            h = self.pre_net(x)
            skips = []
            for wb in self.wave_blocks:
                h, s = wb.forward(h)
                skip_cut = s.shape[1].value - self.output_width
                skips.append(tf.slice(s, [0, skip_cut, 0], [-1, -1, -1]))
            h = tf.reduce_sum(tf.stack(skips, axis=0), axis=0)
            h = self.post_net1(tf.nn.relu(h))
            return self.post_net2(tf.nn.relu(h))



class MultiHeadSelfAttention():
    def __init__(self, in_channel, n_dim, block_size , name, mlp_ratio, dp, layers, n_attention_layers):
        # make it deep
        self.layers = layers
        self.dp = dp
        self.mlp_ratio = mlp_ratio
        self.block_size = block_size
        self.scope = name
        self.n_attention_layers = n_attention_layers
    def forward(self, x, is_train, memory = None):
        # reshape to blocks before performing attention    
        if self.block_size != -1: 
            remainder = x.shape[1] % self.block_size
            if remainder != 0:
                x = tf.concat([x, tf.zeros((x.shape[0], (self.block_size - remainder), x.shape[2]))], axis=1)
            B, T, C = x.shape
            x = tf.concat(
                [x[:, i * self.block_size:(i + 1) * self.block_size] for i in range(x.shape[1] // self.block_size)], axis=0)
        for k in range(self.layers):
            if self.scope != 'FA':
                mode = 'local' if k%2 == 0 else 'strided'
            else:
                mode = 'all'
            l_name = self.scope + '_' + str(k)
            # main attention block
            x = transformer_block(x , memory, scope = l_name , mode= mode ,dp =self.dp, mlp_ratio = self.mlp_ratio, n_attention_layers=self.n_attention_layers, train=is_train, recompute=is_train)
        if self.block_size != -1:
            x = tf.concat([x[i * B:(i + 1) * B] for i in range(x.shape[0] // B)], axis=1)
            if remainder != 0:
                x = x[:,:-(self.block_size - remainder.value)]
        return x


class SampleTransformer():
    def __init__(self, down_sampling_rates, dilation_rates, kernel_size, receptive_field, args):
        self.init_kernel_size = kernel_size if not args.scalar_input else args.init_kernel_size
        self.args = args
        self.lbsa = args.lbsa
        self.lsa = args.lsa
        self.lfa = args.lfa
        self.batch_size = args.batch_size
        self.channel_size = args.channel_size
        self.sample_size = args.sample_size
        self.dilation_rates= dilation_rates
        self.down_sampling_rates = down_sampling_rates
        self.kernel_size = kernel_size
        self.receptive_field = receptive_field
        self.scalar_input = args.scalar_input
        self.output_width1 = self.sample_size + self.receptive_field//2
        self.output_width2 = self.sample_size
        n_attention_layers = self.lfa + (self.lsa + self.lbsa)*2
        with tf.variable_scope('memroy' , reuse=tf.AUTO_REUSE):
            self.memory = tf.get_variable('mem', shape = (args.batch_size , args.n_memory , self.output_width1//np.prod(self.down_sampling_rates) , self.channel_size), initializer=tf.zeros_initializer(), trainable=False)
        self.initial_wavenet = WaveNet(self.channel_size, self.channel_size, self.kernel_size, self.init_kernel_size, self.output_width1, 'wavenet1',  self.dilation_rates)
        self.depth = len(down_sampling_rates)
        self.down_path = [MultiHeadSelfAttention(self.channel_size, self.channel_size, block_size=1024 \
            if i==0 else -1 , name = 'BSA' if i==0 else 'SA', mlp_ratio = 4, dp = 0.05, layers = self.lbsa if i==0 else self.lsa, n_attention_layers=n_attention_layers ) for (i, dsr) in enumerate(down_sampling_rates)]
        self.down_sampling = [tf.keras.layers.AveragePooling1D(dsr) for dsr in self.down_sampling_rates]
        self.middle_attention = MultiHeadSelfAttention(self.channel_size, self.channel_size, block_size=-1 , name = 'FA', mlp_ratio = 4, dp = 0.05, layers = self.lfa, n_attention_layers=n_attention_layers)
        self.up_sampling = [tf.keras.layers.UpSampling1D(dsr) for dsr in self.down_sampling_rates[::-1]]
        self.up_path = [MultiHeadSelfAttention(self.channel_size, self.channel_size, block_size=1024 \
            if i==1 else -1 , name = 'UBSA' if i==1 else 'USA', mlp_ratio = 4, dp = 0.05, layers = self.lbsa if i==1 else self.lsa, n_attention_layers=n_attention_layers ) for (i, dsr) in enumerate(down_sampling_rates[::-1])]
        self.final_wavenet = WaveNet(self.channel_size, self.channel_size, self.init_kernel_size, self.kernel_size, self.output_width2, 'wavenet2', self.dilation_rates)
        self.post_wavenet = Conv1d(self.args.q_levels , kernel_size=1, name='post_wavenet')
    
    def forward(self, x, begin , g_step, is_train):
        _ = tf.cond(begin, lambda: self.memory.assign(tf.zeros_like(self.memory)) , lambda: 0.0)
        h = self.initial_wavenet.forward(x)
        inputs = []
        for d in range(self.depth):
            h = self.down_path[d].forward(h, is_train)
            inputs.append(h)
            h = self.down_sampling[d](h)
        h_pre = h
        h = self.middle_attention.forward(h, is_train, self.memory)
        self.memory[:, g_step%(self.memory.shape[1])].assign(h_pre)
        inputs = inputs[::-1]
        for d in range(self.depth):
            h = self.up_sampling[d](h)
            B , T , C = inputs[d].shape
            p_s = np.prod(self.down_sampling_rates[-(d + 1):]) - 1
            h = tf.concat([tf.zeros((B, p_s, C)), h[:, :-p_s]], axis = 1) + inputs[d]
            h = self.up_path[d].forward(h, is_train)
        h = self.final_wavenet.forward(h)
        with tf.variable_scope('post' , reuse=tf.AUTO_REUSE):
            return self.post_wavenet(h)
        

    def loss(self,
             input_batch,
             begin,
             g_step,
             is_train,
             l2_regularization_strength=None,
             name='wavenet'):
    
        encoded_input = quantize(input_batch,
                                        self.args.q_levels, self.args.q_type)

        # gc_embedding = self._embed_gc(global_condition_batch)
        encoded = _one_hot(encoded_input, self.args.q_levels)
        if self.scalar_input:
            network_input = tf.reshape(
                tf.cast(input_batch, tf.float32),
                [self.batch_size, -1, 1])
        else:
            network_input = encoded

        # Cut off the last sample of network input to preserve causality.
        network_input_width = network_input.shape[1] - 1
        network_input = tf.slice(network_input, [0, 0, 0],
                                    [-1, network_input_width, -1])
        raw_output = self.forward(network_input, begin, g_step, is_train)
        with tf.name_scope('loss'):
            # Cut off the samples corresponding to the receptive field
            # for the first predicted sample.
            target_output = tf.slice(
                tf.reshape(
                    encoded,
                    [self.batch_size, -1, self.args.q_levels]),
                [0, self.receptive_field+1, 0],
                [-1, -1, -1])
            target_output = tf.reshape(target_output,
                                        [-1, self.args.q_levels])
            prediction = tf.reshape(raw_output,
                                    [-1, self.args.q_levels])
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=prediction,
                labels=target_output)
            reduced_loss = tf.reduce_mean(loss)

            # tf.summary.scalar('loss', reduced_loss)

            if l2_regularization_strength is None:
                return reduced_loss
            else:
                # L2 regularization for all trainable parameters
                l2_loss = tf.add_n([tf.nn.l2_loss(v)
                                    for v in tf.trainable_variables()
                                    if not('bias' in v.name)])

                # Add the regularization term to the loss
                total_loss = (reduced_loss +
                                l2_regularization_strength * l2_loss)

                # tf.summary.scalar('l2_loss', l2_loss)
                # tf.summary.scalar('total_loss', total_loss)

                return total_loss




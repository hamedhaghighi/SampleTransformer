import pdb
import tensorflow as tf
import numpy as np
from transformer import transformer_block
# import tensorflow.keras.layers as layers 
from ops import quantize, _one_hot

def mpad(tensor, dilation_rate, kernel_size):
    return tf.pad(tensor, [[0, 0], [dilation_rate * (kernel_size - 1), 0],[0,0]])
    
class WaveNetBlock():
    def __init__(self, in_channels, intermediate_channels, kernel_size, dilation_rate):
        self.dilation_rate = dilation_rate
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.Conv1D(intermediate_channels*2, kernel_size=self.kernel_size, dilation_rate= dilation_rate)
        self.post_linear = tf.keras.layers.Conv1D(in_channels, kernel_size=1)
        self.skip_conv = tf.keras.layers.Conv1D(in_channels, kernel_size=1)
        

    def GLU(self, x):
        B, T, C = x.shape
        g, f = tf.split(x, [C.value//2, C.value//2], axis = 2)
        return tf.sigmoid(g) * tf.tanh(f)

    def forward(self, x):
        skip_out = self.GLU(self.conv(x))
        input_cut = x.shape[1].value - tf.shape(skip_out)[1]
        y = tf.slice(x, [0, input_cut, 0], [-1, -1, -1])
        return y + self.post_linear(skip_out), self.skip_conv(skip_out)


class WaveNet():
    def __init__(self, in_channels, intermediate_channels, kernel_size, init_kernel_size, output_width, dilation_rates=None):
        self.output_width = output_width
        self.kernel_size = kernel_size
        self.pre_net = tf.keras.layers.Conv1D(in_channels, kernel_size=init_kernel_size)
        self.wave_blocks = []
        for d in dilation_rates:
            self.wave_blocks.append(WaveNetBlock(in_channels, intermediate_channels, kernel_size, d))
        self.post_net = tf.keras.Sequential([
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv1D(in_channels, kernel_size=1),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv1D(in_channels, kernel_size=1)
        ])

    def forward(self, x):
        h = self.pre_net(x)
        skips = []
        for wb in self.wave_blocks:
            h, s = wb.forward(h)
            skip_cut = s.shape[1].value - self.output_width
            skips.append(tf.slice(s, [0, skip_cut, 0], [-1, -1, -1]))
        h = tf.reduce_sum(tf.stack(skips, axis=0), axis=0)
        return self.post_net(h)



class MultiHeadSelfAttention():
    def __init__(self, in_channel, n_dim, block_size , name, mlp_ratio, dp, layers):
        # make it deep
        self.layers = layers
        self.dp = dp
        self.mlp_ratio = mlp_ratio
        self.block_size = block_size
        self.scope = name
    def forward(self, x, dropout_cache, memory = None):
        # x is B T C
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
            x = transformer_block(x , memory, scope = l_name, mode= mode ,dp =self.dp, mlp_ratio = self.mlp_ratio, dropout_cache = dropout_cache)
        if self.block_size != -1:
            x = tf.concat([x[i * B:(i + 1) * B] for i in range(x.shape[0] // B)], axis=1)
            if remainder != 0:
                x = x[:,:-(self.block_size - remainder.value)]
        return x


class SampleTransformer():
    def __init__(self, down_sampling_rates, dilation_rates, kernel_size, receptive_field, args):
        self.init_kernel_size = kernel_size if not args.scalar_input else args.init_kernel_size
        self.dropout_cache = dict()
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
        self.memory = tf.get_variable('memory', shape = (args.batch_size , 3 , self.output_width1//256 , self.channel_size), initializer=tf.zeros_initializer(), trainable=False)
        self.initial_wavenet = WaveNet(self.channel_size, self.channel_size, self.kernel_size, self.init_kernel_size, self.output_width1, self.dilation_rates)
        self.depth = len(down_sampling_rates)
        self.down_path = [MultiHeadSelfAttention(self.channel_size, self.channel_size, block_size=1024 if i==0 else -1 , name = 'BSA' if i==0 else 'SA', mlp_ratio = 4, dp = 0.05, layers = self.lbsa if i==0 else self.lsa ) for (i, dsr) in enumerate(down_sampling_rates)]
        self.down_sampling = [tf.keras.layers.AveragePooling1D(dsr) for dsr in self.down_sampling_rates]
        self.middle_attention = MultiHeadSelfAttention(self.channel_size, self.channel_size, block_size=-1 , name = 'FA', mlp_ratio = 4, dp = 0.05, layers = self.lfa)
        self.up_sampling = [tf.keras.layers.UpSampling1D(dsr) for dsr in self.down_sampling_rates[::-1]]
        self.up_path = [MultiHeadSelfAttention(self.channel_size, self.channel_size, block_size=1024 if i==1 else -1 , name = 'UBSA' if i==1 else 'USA', mlp_ratio = 4, dp = 0.05, layers = self.lbsa if i==1 else self.lsa ) for (i, dsr) in enumerate(down_sampling_rates[::-1])]
        self.final_wavenet = WaveNet(self.channel_size, self.channel_size, self.init_kernel_size, self.kernel_size, self.output_width2, self.dilation_rates)
        self.post_wavenet = tf.keras.layers.Conv1D(self.args.q_levels , kernel_size=1)
    
    def loss(self,
             input_batch,
             begin,
             g_step,
             global_condition_batch=None,
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
        network_input_width = tf.shape(network_input)[1] - 1
        network_input = tf.slice(network_input, [0, 0, 0],
                                    [-1, network_input_width, -1])
        raw_output = self.forward(network_input, begin, g_step)
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
            loss = tf.nn.softmax_cross_entropy_with_logits(
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

    def forward(self, x, begin , g_step):
        # x is T, B, C which is self attention friendly, wavenet and pooling layers get B, C, T
        if begin == True:
            self.memory.assign(tf.zeros_like(self.memory))
        h = self.initial_wavenet.forward(x)
        inputs = []
        for d in range(self.depth):
            h = self.down_path[d].forward(h, self.dropout_cache)
            inputs.append(h)
            h = self.down_sampling[d](h)
        h_pre = h
        h = self.middle_attention.forward(h, self.dropout_cache, self.memory)
        self.memory[:, g_step%3].assign(h_pre)
        inputs = inputs[::-1]
        for d in range(self.depth):
            h = self.up_sampling[d](h)
            B , T , C = inputs[d].shape
            x = tf.concat([tf.zeros((B, np.prod(self.down_sampling_rates[-(d + 1):]) - 1, C)), inputs[d][:, np.prod(self.down_sampling_rates[-(d + 1):]) - 1:]], axis = 1)
            h = h[:, :x.shape[1]]
            h = self.up_path[d].forward(h + x, self.dropout_cache)
        h = self.final_wavenet.forward(h)
        return self.post_wavenet(h)



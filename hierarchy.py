import pdb
import torch
import numpy as np
import torch.nn as nn
from pytorch_transformers.modeling_transfo_xl import RelLearnableDecoderLayer


def sa2conv(tensor):
    return tensor.permute(1, 2, 0)


def conv2sa(tensor):
    return tensor.permute(2, 0, 1)


class WaveNetBlock(nn.Module):
    def __init__(self, in_channels, intermediate_channels, kernel_size, dilation_rate):
        super().__init__()
        self.padder = nn.ConstantPad1d((dilation_rate * (kernel_size - 1), 0), 0)
        self.conv = nn.Conv1d(in_channels, intermediate_channels * 2, kernel_size, 1, padding=0, dilation=dilation_rate)
        self.act = nn.GLU(dim=1)
        self.post_linear = nn.Conv1d(intermediate_channels, in_channels, kernel_size=1)

    def forward(self, x):
        skip_out = self.act(self.conv(self.padder(x)))
        return x + self.post_linear(skip_out), skip_out


class WaveNet(nn.Module):
    def __init__(self, audio_channels, in_channels, intermediate_channels, kernel_size, dilation_rates=None):
        super().__init__()
        if dilation_rates is None:
            dilation_rates = [2 ** i for i in range(10)] * 4
        self.pre_pad = nn.ConstantPad1d((2, 0), 0)
        self.pre_net = nn.Conv1d(audio_channels, in_channels, kernel_size=3, padding=0)
        self.wave_blocks = nn.ModuleList()
        for d in dilation_rates:
            self.wave_blocks.append(WaveNetBlock(in_channels, intermediate_channels, kernel_size, d))
        self.post_net = nn.Sequential(nn.ReLU(True), nn.Linear(intermediate_channels, intermediate_channels),
                                      nn.ReLU(True), nn.Linear(intermediate_channels, in_channels))

    def forward(self, x):
        h = self.pre_net(self.pre_pad(x))
        skips = []
        for wb in self.wave_blocks:
            h, s = wb(h)
            skips.append(s)
        h = conv2sa(torch.sum(torch.stack(skips, dim=0), dim=0))
        return self.post_net(h)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, dropatt=0, tgt_len=None, ext_len=None,
                 mem_len=None, pre_lnorm=False, r_r_bias=None, r_w_bias=None, block_size=-1):
        super().__init__()
        # make it deep
        self.self_attention = RelLearnableDecoderLayer(n_head, d_model, d_head, d_inner, dropout, dropatt=dropatt,
                                                       tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                                                       pre_lnorm=pre_lnorm,
                                                       r_r_bias=r_r_bias, r_w_bias=r_w_bias, output_attentions=False)
        self.block_size = block_size

    def forward(self, x, mask=None):
        # x is TBC
        if self.block_size != -1:  # BS, B*T//BS, C
            remainder = x.size(0) % self.block_size
            if remainder != 0:
                x = torch.cat([x, torch.zeros((self.block_size - remainder), x.size(1), x.size(2)).to(x)], dim=0)
            T, B, C = x.size()
            x = torch.cat(
                [x[i * self.block_size:(i + 1) * self.block_size] for i in range(x.size(0) // self.block_size)], dim=1)
        # do your thing
        if self.block_size != -1:
            x = torch.cat([x[:, i * B:(i + 1) * B] for i in range(x.size(1) // B)], dim=0)
            if remainder != 0:
                x = x[:-(self.block_size - remainder)]
        return x

    def __repr__(self):
        return super().__repr__()


class SampleTransformer(nn.Module):
    def __init__(self, audio_channels, model_channels, self_attention_kwargs, down_sampling_rates):
        super().__init__()
        self.initial_wavenet = WaveNet(audio_channels, model_channels, model_channels * 2, 3, [1, 2, 4, 8, 1, 2, 4, 8])
        self.depth = len(down_sampling_rates)
        self.down_sampling_rates = np.array(down_sampling_rates)
        self.down_path = nn.ModuleList(
            [MultiHeadSelfAttention(**self_attention_kwargs, block_size=dsr) for dsr in down_sampling_rates])
        self.down_sampling = nn.ModuleList([nn.AvgPool1d(dsr) for dsr in down_sampling_rates])
        self.middle_attention = MultiHeadSelfAttention(**self_attention_kwargs)
        self.up_sampling = nn.ModuleList([nn.Upsample(scale_factor=dsr) for dsr in down_sampling_rates[::-1]])
        self.up_path = nn.ModuleList(
            [MultiHeadSelfAttention(**self_attention_kwargs, block_size=dsr) for dsr in down_sampling_rates[::-1]])
        self.final_wavenet = WaveNet(model_channels, model_channels, model_channels * 2, 3, [1, 2, 4, 8, 1, 2, 4, 8])

    def forward(self, x, mem=None):
        # x is T, B, C which is self attention friendly, wavenet and pooling layers get B, C, T
        h = self.initial_wavenet(sa2conv(x))
        inputs = []
        for d in range(self.depth):
            h = self.down_path[d](h)
            inputs.append(h)
            h = sa2conv(h)
            h = self.down_sampling[d](h)
            h = conv2sa(h)
        h = self.middle_attention(h, mem)
        inputs = inputs[::-1]
        for d in range(self.depth):
            h = sa2conv(h)
            h = self.up_sampling[d](h)
            h = conv2sa(h)
            x = inputs[d][np.cumprod(self.down_sampling_rates[-(d + 1):])[-1] - 1:]
            h = h[:x.size(0)]
            h = self.up_path[d](h + x)
        h = sa2conv(h)
        return self.final_wavenet(h)


def test():
    self_attention_kwargs = dict(n_head=1, d_model=1, d_head=1, d_inner=1, dropout=0)
    for i, dsr in enumerate([[1], [1, 1], [2], [2, 2], [2, 4, 8]]):
        a = SampleTransformer(2, 16, self_attention_kwargs, dsr)
        seq_len = 1024
        batch_size = 3
        x = torch.randn(seq_len, batch_size, 2)  # TBC
        assert a(x).shape == (seq_len - np.cumprod(np.array(dsr))[-1] + 1, batch_size, 16), a(x).shape


if __name__ == '__main__':
    test()

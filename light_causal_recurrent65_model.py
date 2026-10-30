"""Causal recurrent spatial U-Net with 65 frequency groups and full-resolution skips."""

import torch
from torch import nn

from light_causal_fast_model import CausalUNetSpatial, CausalFrequencyConv
from light_causal_global_model import GlobalTemporalLayer
from light_causal_recurrent_model import RecurrentSpatialLayer


RECURRENT65_CANDIDATE = 'causal_recurrent_unet65_v1'


class CausalRecurrentUNet65(CausalUNetSpatial):
    def __init__(self, M=8, *, hidden=96, ffn=192, layers=4, global_hidden=128, context=128, **kwargs):
        nn.Module.__init__(self)
        if M != 8 or kwargs.get('is_causal', True) is False or hidden % 8 or ffn % 8 or layers < 1:
            raise ValueError('Use eight microphones, causal mode and compatible dimensions')
        self.M, self.context = M, context
        self.stem = CausalFrequencyConv(16, 32, frequency_kernel=5)
        widths = (32, 48, hidden)
        self.encoders = nn.ModuleList([CausalFrequencyConv(a, b, 2, 5) for a, b in zip(widths[:-1], widths[1:])])
        self.layers = nn.ModuleList()
        full = None
        for _ in range(layers):
            layer = RecurrentSpatialLayer(hidden, ffn, 65, context, full=full)
            self.layers.append(layer)
            full = layer.full
        with torch.random.fork_rng(devices=[]):
            self.layers.append(GlobalTemporalLayer(65, hidden, global_hidden))
        reverse = tuple(reversed(widths))
        self.upsamplers = nn.ModuleList([nn.ConvTranspose2d(a, b, (1, 4), stride=(1, 2), padding=(0, 1))
            for a, b in zip(reverse[:-1], reverse[1:])])
        self.decoders = nn.ModuleList([CausalFrequencyConv(2 * b, b) for b in reverse[1:]])
        self.beamformer = nn.Conv2d(32, 2 * M, 1)
        nn.init.zeros_(self.beamformer.weight)
        nn.init.zeros_(self.beamformer.bias)
        self.use_bfloat16_training = True

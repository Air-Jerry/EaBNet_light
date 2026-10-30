"""Full-frequency causal spatial encoder with recurrent temporal modeling.

Frequency modules reuse the licensed SpatialNet code. Replacing temporal
attention is an experimental adaptation, not an author-confirmed reproduction.
"""

import torch
from torch import nn

from light_causal_fast_model import ActivationPReLU, activation_norms
from light_causal_spatial_model import CausalSpatialEnhancer, CausalSpatialLayer


RECURRENT_CANDIDATE = 'causal_recurrent_spatial_v1'


class RecurrentSpatialLayer(CausalSpatialLayer):
    def __init__(self, hidden, ffn, frequencies, context, full=None):
        super().__init__(hidden, ffn, frequencies, 4, context, full=full)
        self.mhsa = nn.GRU(hidden, hidden, batch_first=True, bidirectional=False)
        self.fconv1[-1] = ActivationPReLU(hidden)
        self.fconv2[-1] = ActivationPReLU(hidden)
        activation_norms(self)

    def attention(self, value, state):
        batch, frequencies, frames, channels = value.shape
        dtype = value.dtype
        with torch.autocast(value.device.type, enabled=False):
            normalized = self.norm_mhsa(value.float()).reshape(batch * frequencies, frames, channels)
            output, hidden = self.mhsa(normalized, None if state is None else state.get('hidden'))
        cached = {'hidden': hidden.detach().clone()} if state is not None else None
        return output.reshape(batch, frequencies, frames, channels).to(dtype), cached


class CausalRecurrentSpatial(CausalSpatialEnhancer):
    def __init__(self, M=8, *, hidden=96, ffn=192, layers=4, context=128, **kwargs):
        nn.Module.__init__(self)
        if M != 8 or kwargs.get('is_causal', True) is False or hidden % 8 or ffn % 8 or layers < 1:
            raise ValueError('Use eight microphones, causal mode and positive compatible dimensions')
        self.M, self.context = M, context
        self.encoder = nn.Conv1d(2 * M, hidden, kernel_size=5, padding=0)
        self.layers = nn.ModuleList()
        full = None
        for _ in range(layers):
            layer = RecurrentSpatialLayer(hidden, ffn, 257, context, full=full)
            self.layers.append(layer)
            full = layer.full
        self.decoder = nn.Linear(hidden, 2)
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)
        self.gradient_checkpointing = False
        self.use_bfloat16_training = True

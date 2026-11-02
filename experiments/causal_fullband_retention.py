"""Full-resolution cross-band/retention backbone with one batch/stream equation.

Frequency modules reuse the existing licensed SpatialNet implementation. The
retention recurrence is the independently implemented canonical sum/count form
already used in this repository, not the author's numerically different
chunkwise and recurrent operators. No pretrained or frozen model is included.
"""

import torch
from torch import nn

from experiments.causal_retention_mapping import CausalRetention
from light_causal_fast_model import ActivationPReLU, activation_norms
from light_causal_spatial_model import CausalSpatialEnhancer, CausalSpatialLayer


class FullbandRetentionLayer(CausalSpatialLayer):
    def __init__(self, hidden, ffn, heads, context, full=None):
        super().__init__(hidden, ffn, 257, heads, context, full=full)
        self.mhsa = CausalRetention(hidden, heads=heads)
        self.mhsa.norm = nn.Identity()
        nn.init.xavier_uniform_(self.mhsa.output.weight, gain=0.5)
        self.fconv1[-1] = ActivationPReLU(hidden)
        self.fconv2[-1] = ActivationPReLU(hidden)
        activation_norms(self)

    def attention(self, value, state):
        normalized = self.norm_mhsa(value).permute(0, 3, 2, 1).float()
        retained, cached = self.mhsa.process(normalized,
            None if state is None else state.get('retention'), cache=state is not None)
        delta = (retained - normalized).permute(0, 3, 2, 1).to(value.dtype)
        return delta, {'retention': cached} if state is not None else None


class CausalFullbandRetention(CausalSpatialEnhancer):
    """Eight full257-bin blocks; explicit constant-size causal matrix memories."""

    def __init__(self, M=8, *, hidden=96, ffn=192, layers=8, heads=4,
                 context=64, **kwargs):
        nn.Module.__init__(self)
        if (M != 8 or kwargs.get('is_causal', True) is False or layers < 1
                or hidden < 8 or ffn < 8 or hidden % 8 or ffn % 8
                or heads < 1 or hidden % heads or context < 2):
            raise ValueError('Use eight microphones and positive compatible causal dimensions')
        self.M, self.context = M, context
        self.encoder = nn.Conv1d(2 * M, hidden, kernel_size=5, padding=0)
        self.layers = nn.ModuleList()
        full = None
        for _ in range(layers):
            layer = FullbandRetentionLayer(hidden, ffn, heads, context, full=full)
            self.layers.append(layer)
            full = layer.full
        self.decoder = nn.Linear(hidden, 2)
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)
        self.gradient_checkpointing = True
        self.use_bfloat16_training = True

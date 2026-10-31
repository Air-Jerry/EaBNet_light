"""Global frequency attention inside one current frame; no attention across time."""

import torch
from torch import nn
import torch.nn.functional as F

from experiments.causal_tcn_mapping import CausalTCNMapping
from light_causal_fast_model import ActivationLayerNorm


class FrequencyAttentionInput(nn.Conv2d):
    def __init__(self, incoming, outgoing, heads=4, head_dim=16):
        super().__init__(incoming, outgoing, 1)
        self.frequency_position = nn.Parameter(torch.randn(1, 1, 257, outgoing) * .01)
        self.frequency_norm = ActivationLayerNorm(outgoing)
        self.frequency_qkv = nn.Linear(outgoing, 3 * heads * head_dim)
        self.frequency_output = nn.Linear(heads * head_dim, outgoing)
        self.heads, self.head_dim = heads, head_dim
        nn.init.zeros_(self.frequency_output.weight)
        nn.init.zeros_(self.frequency_output.bias)

    def forward(self, features):
        if features.shape[-1] != 257:
            raise ValueError('Expected257 STFT frequency bins')
        value = super().forward(features)
        batch, channels, frames, frequencies = value.shape
        current = value.permute(0, 2, 3, 1)
        current = self.frequency_norm(current + self.frequency_position.to(current.dtype))
        # Each time frame is an independent batch item; only frequency is the sequence.
        qkv = self.frequency_qkv(current).reshape(batch * frames, frequencies, 3, self.heads, self.head_dim)
        query, key, content = (part.transpose(1, 2) for part in qkv.unbind(dim=2))
        attended = F.scaled_dot_product_attention(query, key, content, dropout_p=0., is_causal=False)
        attended = attended.transpose(1, 2).reshape(batch, frames, frequencies, self.heads * self.head_dim)
        delta = self.frequency_output(attended).permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
        return value + delta


class CausalFrequencyAttentionTCN(CausalTCNMapping):
    def __init__(self, M=8, **kwargs):
        super().__init__(M=M, **kwargs)
        original = self.mapping_input
        with torch.random.fork_rng(devices=[]):
            changed = FrequencyAttentionInput(original.in_channels, original.out_channels)
        missing, unexpected = changed.load_state_dict(original.state_dict(), strict=False)
        assert len(missing) == 7 and all(name.startswith('frequency_') for name in missing) and not unexpected
        self.mapping_input = changed

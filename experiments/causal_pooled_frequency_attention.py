"""Current-frame frequency attention with257 queries and129 pooled keys/values."""

import torch
from torch import nn
import torch.nn.functional as F

from experiments.causal_frequency_attention import CausalFrequencyAttentionTCN, FrequencyAttentionInput


class PooledFrequencyAttentionInput(FrequencyAttentionInput):
    def forward(self, features):
        if features.shape[-1] != 257:
            raise ValueError('Expected257 STFT frequency bins')
        value = nn.Conv2d.forward(self, features)
        batch, channels, frames, frequencies = value.shape
        current = self.frequency_norm(value.permute(0, 2, 3, 1) + self.frequency_position.to(value.dtype))
        qkv = self.frequency_qkv(current).reshape(batch * frames, frequencies, 3, self.heads, self.head_dim)
        query, key, content = (part.transpose(1, 2) for part in qkv.unbind(2))
        # Pool adjacent frequencies only; the Nyquist bin remains a singleton.
        def pool(part):
            pairs = part[:, :, :256].reshape(batch * frames, self.heads, 128, 2, self.head_dim).mean(-2)
            return torch.cat((pairs, part[:, :, 256:]), dim=2)
        attended = F.scaled_dot_product_attention(query, pool(key), pool(content), dropout_p=0., is_causal=False)
        attended = attended.transpose(1, 2).reshape(batch, frames, frequencies, self.heads * self.head_dim)
        delta = self.frequency_output(attended).permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
        return value + delta


class CausalPooledFrequencyAttentionTCN(CausalFrequencyAttentionTCN):
    def __init__(self, M=8, **kwargs):
        super().__init__(M=M, **kwargs)
        original = self.mapping_input
        with torch.random.fork_rng(devices=[]):
            changed = PooledFrequencyAttentionInput(original.in_channels, original.out_channels)
        changed.load_state_dict(original.state_dict(), strict=True)
        self.mapping_input = changed

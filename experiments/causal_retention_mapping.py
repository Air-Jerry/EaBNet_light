"""Experimental per-frequency matrix memory with matching batch and stream equations.

Inspired by the retention recurrence in arxiv.org/abs/2403.07675, not an author
implementation. No upstream source is incorporated. Quality and GPU cost are unverified.
"""

import torch
from torch import nn
import torch.nn.functional as F

from experiments.causal_frequency_attention import CausalFrequencyAttentionTCN
from light_causal_fast_model import ActivationLayerNorm
from light_causal_recurrent65_model import CausalRecurrentUNet65


class CausalRetention(nn.Module):
    def __init__(self, channels, heads=4, chunk_size=64):
        super().__init__()
        if heads < 1 or channels % heads or channels < heads or chunk_size < 1:
            raise ValueError('Use compatible channel/head counts and a positive chunk size')
        self.channels, self.heads = channels, heads
        self.head_dim, self.chunk_size = channels // heads, chunk_size
        self.norm = ActivationLayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.gate = nn.Linear(channels, channels)
        self.output = nn.Linear(channels, channels)
        powers = torch.tensor([4., 5., 9., 10.]) if heads == 4 else torch.linspace(4, 10, heads).round()
        self.register_buffer('decay', 1 - torch.pow(2., -powers))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def process(self, features, state=None, cache=False):
        batch, channels, frames, frequencies = features.shape
        if channels != self.channels or frames < 1:
            raise ValueError('Expected nonempty (B,C,T,F) retention features')
        value = features.permute(0, 3, 2, 1)
        normalized = self.norm(value)
        projection = self.qkv(normalized).reshape(batch * frequencies, frames, 3, self.heads, self.head_dim)
        query, key, content = (v.permute(0, 2, 1, 3).float() for v in projection.unbind(2))
        gate = F.silu(self.gate(normalized)).reshape(batch * frequencies, frames, channels)
        shape = (batch * frequencies, self.heads, self.head_dim, self.head_dim)
        if state is None:
            matrix = query.new_zeros(shape)
            count = query.new_zeros(self.heads)
        else:
            matrix, count = state['matrix'], state['count']
            if matrix.shape != shape or count.shape != (self.heads,):
                raise ValueError('Retention state differs from the stream shape')
        pieces, weights = [], {}
        # Canonical decayed sums and counts precede RMS normalization in every mode.
        with torch.autocast(features.device.type, enabled=False):
            key = key * self.head_dim ** -.5
            decay = self.decay.float()
            for start in range(0, frames, self.chunk_size):
                length = min(self.chunk_size, frames - start)
                if length not in weights:
                    index = torch.arange(length, device=features.device)
                    distance = index[:, None] - index[None, :]
                    mask = decay[:, None, None].pow(distance.clamp_min(0)) * (distance >= 0)
                    prefix = decay[:, None].pow(index + 1)
                    end = decay.pow(length)
                    tail = decay[:, None].pow(length - 1 - index)
                    count_addition = (1 - prefix) / (1 - decay[:, None])
                    weights[length] = mask, prefix, end, tail, count_addition
                mask, prefix, end, tail, count_addition = weights[length]
                q = query[:, :, start:start + length]
                k = key[:, :, start:start + length]
                v = content[:, :, start:start + length]
                inner = ((q @ k.transpose(-1, -2)) * mask) @ v
                previous = (q @ matrix) * prefix[None, :, :, None]
                counts = count[:, None] * prefix + count_addition
                accumulated = (inner + previous) / counts[None, :, :, None].sqrt()
                accumulated = accumulated * torch.rsqrt(accumulated.square().mean(-1, keepdim=True) + 1e-6)
                pieces.append(accumulated.transpose(1, 2).reshape(batch * frequencies, length, channels))
                matrix = matrix * end[None, :, None, None] + k.transpose(-1, -2) @ (v * tail[None, :, :, None])
                count = counts[:, -1]
            combined = torch.cat(pieces, 1) * gate.float()
        delta = self.output(combined.to(features.dtype)).reshape(batch, frequencies, frames, channels)
        result = features + delta.permute(0, 3, 2, 1).contiguous(memory_format=torch.channels_last)
        updated = dict(matrix=matrix.detach().clone(), count=count.detach().clone()) if cache else None
        return result, updated

    def forward(self, features):
        return self.process(features)[0]


class CausalAttentionRetentionTCN(CausalFrequencyAttentionTCN):
    def __init__(self, M=8, **kwargs):
        super().__init__(M=M, **kwargs)
        with torch.random.fork_rng(devices=[]):
            self.retention = CausalRetention(self.mapping_input.out_channels)

    def process(self, inpt, state=None):
        estimate, updated = CausalRecurrentUNet65.process(self, inpt, state)
        cache = state is not None
        memories = {} if state is None else state
        with torch.autocast(inpt.device.type, enabled=False):
            reference = inpt[..., 0, :].float()
            scale = torch.linalg.vector_norm(reference, dim=-1).mean(-1, keepdim=True).unsqueeze(-1).clamp_min(1e-4)
            features = torch.cat((inpt.float().flatten(-2), estimate.permute(0, 2, 3, 1)), -1) / scale
            features = features.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
        enabled = self.training and self.use_bfloat16_training and inpt.device.type == 'cuda'
        with torch.autocast(inpt.device.type, dtype=torch.bfloat16, enabled=enabled):
            value = self.mapping_input(features)
            value, retention = self.retention.process(value, memories.get('retention'), cache)
            histories = []
            for index, layer in enumerate(self.mapping_blocks):
                history = memories.get('mapping', [None] * len(self.mapping_blocks))[index]
                value, memory = layer.process(value, history, cache)
                if cache:
                    histories.append(memory)
            delta = self.mapping_output(value).float()
        if cache:
            updated.update(retention=retention, mapping=histories)
        return estimate + delta * scale.permute(0, 3, 1, 2), updated

"""Experimental full-frequency causal spectral mapping with a trainable source."""

import torch
from torch import nn
import torch.nn.functional as F

from light_causal_fast_model import ActivationPReLU, FrameChannelNorm
from light_causal_recurrent65_model import CausalRecurrentUNet65


class CausalMappingBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.history_frames = 2 * dilation
        self.norm = FrameChannelNorm(channels)
        self.input = nn.Conv2d(channels, 2 * channels, 1)
        self.depthwise = nn.Conv2d(channels, channels, (3, 3), dilation=(dilation, 1),
                                   padding=(0, 1), groups=channels)
        self.activation = ActivationPReLU(channels)
        self.output = nn.Conv2d(channels, channels, 1)

    def process(self, value, history=None, cache=False):
        left, gate = self.input(self.norm(value)).chunk(2, dim=1)
        temporal = F.silu(left) * torch.sigmoid(gate)
        shape = (value.shape[0], temporal.shape[1], self.history_frames, value.shape[3])
        if history is None: history = temporal.new_zeros(shape)
        if history.shape != shape: raise ValueError('Mapping cache differs from input stream')
        joined = torch.cat((history, temporal), dim=2)
        result = value + self.output(self.activation(self.depthwise(joined)))
        return result, joined[:, :, -self.history_frames:].detach().clone() if cache else None


class CausalTCNMapping(CausalRecurrentUNet65):
    def __init__(self, M=8, mapping_channels=32, **kwargs):
        super().__init__(M=M, **kwargs)
        with torch.random.fork_rng(devices=[]):
            self.mapping_input = nn.Conv2d(2 * M + 2, mapping_channels, 1)
            self.mapping_blocks = nn.ModuleList([CausalMappingBlock(mapping_channels, dilation)
                                                for dilation in (1, 2, 4, 8, 16, 32)])
            self.mapping_output = nn.Conv2d(mapping_channels, 2, 1)
            nn.init.zeros_(self.mapping_output.weight)
            nn.init.zeros_(self.mapping_output.bias)

    def process(self, inpt, state=None):
        estimate, updated = super().process(inpt, state)
        cache = state is not None
        memories = {} if state is None else state
        # One scale per current frame; no utterance statistics or clean target.
        with torch.autocast(inpt.device.type, enabled=False):
            reference = inpt[..., 0, :].float()
            scale = torch.linalg.vector_norm(reference, dim=-1).mean(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(1e-4)
            features = torch.cat((inpt.float().flatten(-2), estimate.permute(0, 2, 3, 1)), dim=-1) / scale
            features = features.permute(0, 3, 1, 2).contiguous()
        enabled = self.training and self.use_bfloat16_training and inpt.device.type == 'cuda'
        with torch.autocast(inpt.device.type, dtype=torch.bfloat16, enabled=enabled):
            value = self.mapping_input(features)
            histories = []
            for index, layer in enumerate(self.mapping_blocks):
                history = memories.get('mapping', [None] * len(self.mapping_blocks))[index]
                value, memory = layer.process(value, history, cache)
                if cache: histories.append(memory)
            delta = self.mapping_output(value).float()
        if cache: updated['mapping'] = histories
        return estimate + delta * scale.permute(0, 3, 1, 2), updated

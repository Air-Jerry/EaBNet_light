"""Independent full-channel frequency/time LSTM backbone for causal enhancement.

All channels interact in every recurrent path. Bidirectional recurrence is
confined to the current frame's frequencies; time recurrence is unidirectional.
No fixed teacher or external pretrained weights are used.
"""
import math

import torch
from torch import nn

from experiments.causal_gated_multichannel_crn import CausalGatedMultichannelCRN, FixedStatisticsAffine
from light_causal_fast_model import FrameChannelNorm


class CurrentChannelNorm(FrameChannelNorm):
    def forward(self, value):
        return super().forward(value).contiguous()


class FullChannelDualPath(nn.Module):
    def __init__(self, channels, frequencies, frequency_hidden, temporal_hidden, residual_scale):
        super().__init__()
        self.channels, self.frequencies, self.temporal_hidden = channels, frequencies, temporal_hidden
        self.residual_scale = residual_scale
        self.frequency_input_norm = nn.LayerNorm(channels)
        self.frequency = nn.LSTM(channels, frequency_hidden, batch_first=True, bidirectional=True)
        self.frequency_output = nn.Linear(2 * frequency_hidden, channels)
        self.frequency_norm = nn.LayerNorm((frequencies, channels))
        self.temporal_input_norm = nn.LayerNorm(channels)
        self.temporal = nn.LSTM(channels, temporal_hidden, batch_first=True, bidirectional=False)
        self.temporal_output = nn.Linear(temporal_hidden, channels)
        self.temporal_norm = nn.LayerNorm((frequencies, channels))

    def process(self, value, state=None, cache=False):
        batch, channels, frames, frequencies = value.shape
        if (channels, frequencies) != (self.channels, self.frequencies):
            raise ValueError('Dual-path frequency/channel shape differs')
        if state is not None:
            shape = (1, batch * frequencies, self.temporal_hidden)
            if (not isinstance(state, tuple) or len(state) != 2
                    or any(item.shape != shape or item.dtype != torch.float32 or item.device != value.device for item in state)):
                raise ValueError('Temporal LSTM state differs from the stream')
        dtype = value.dtype
        with torch.autocast(value.device.type, enabled=False):
            current = value.float().permute(0, 2, 3, 1)
            frequency = self.frequency_input_norm(current).reshape(batch * frames, frequencies, channels)
            frequency = self.frequency(frequency)[0]
            frequency = self.frequency_output(frequency).reshape(batch, frames, frequencies, channels)
            current = current + self.residual_scale * self.frequency_norm(frequency)
            sequence = self.temporal_input_norm(current).permute(0, 2, 1, 3).reshape(batch * frequencies, frames, channels)
            temporal, hidden = self.temporal(sequence, state)
            temporal = self.temporal_output(temporal).reshape(batch, frequencies, frames, channels).permute(0, 2, 1, 3)
            result = current + self.residual_scale * self.temporal_norm(temporal)
        updated = tuple(item.detach().clone() for item in hidden) if cache else None
        return result.permute(0, 3, 1, 2).to(dtype), updated


class ScaledReadout(nn.ConvTranspose2d):
    def forward(self, value):
        return super().forward(value * self.in_channels ** -.5)


class CausalFullChannelDualPathCRN(CausalGatedMultichannelCRN):
    def __init__(self, M=8, *, channels=64, layers=6, frequency_hidden=96, temporal_hidden=128, **kwargs):
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
               for value in (layers, frequency_hidden, temporal_hidden)):
            raise ValueError('Use positive integer recurrent dimensions')
        super().__init__(M=M, channels=channels, recurrent_layers=layers, **kwargs)
        self.recurrent = nn.ModuleList([FullChannelDualPath(channels, 65, frequency_hidden, temporal_hidden,
            1 / math.sqrt(2 * layers)) for _ in range(layers)])

        def normalize(parent):
            for name, child in list(parent.named_children()):
                if isinstance(child, FixedStatisticsAffine):
                    replacement = CurrentChannelNorm(child.weight.numel())
                    replacement.norm.weight, replacement.norm.bias = child.weight, child.bias
                    setattr(parent, name, replacement)
                else:
                    normalize(child)

        normalize(self)
        old = self.output
        self.output = ScaledReadout(channels, 18, (1, 5), stride=(1, 2), padding=(0, 2))
        self.output.load_state_dict(old.state_dict(), strict=True)

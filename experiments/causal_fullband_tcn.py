"""Experimental dense full-band causal TCN with explicit microphone phase cues.

No author-equivalence or quality claim. Each frame mixes all257 frequencies;
only left convolution histories connect frames. Original compressed-STFT I/O
and the reference0 initialization are retained.
"""

import torch
from torch import nn
import torch.nn.functional as F

from light_causal_fast_model import ActivationLayerNorm, ActivationPReLU


class FullbandTemporalBlock(nn.Module):
    def __init__(self, channels, hidden, dilation):
        super().__init__()
        self.history_frames = 2 * dilation
        self.norm = ActivationLayerNorm(channels, seq_last=True)
        self.input = nn.Conv1d(channels, 2 * hidden, 1)
        self.temporal = nn.Conv1d(hidden, hidden, 3, dilation=dilation, groups=hidden)
        self.activation = ActivationPReLU(hidden)
        self.output = nn.Conv1d(hidden, channels, 1)

    def process(self, value, history=None, cache=False):
        left, gate = self.input(self.norm(value)).chunk(2, dim=1)
        current = F.silu(left) * torch.sigmoid(gate)
        shape = (value.shape[0], current.shape[1], self.history_frames)
        if history is None:
            joined = F.pad(current, (self.history_frames, 0))
        else:
            if history.shape != shape:
                raise ValueError('Full-band TCN history differs from the input stream')
            joined = torch.cat((history, current), dim=-1)
        result = value + self.output(self.activation(self.temporal(joined)))
        updated = joined[..., -self.history_frames:].detach().clone() if cache else None
        return result, updated


class CausalFullbandTCN(nn.Module):
    is_causal = True
    refiner_causal = True

    def __init__(self, M=8, fullband_channels=512, temporal_hidden=1024, repeats=3, dilations=(1,2,4,8,16,32,64,128), **kwargs):
        super().__init__()
        channels, hidden = fullband_channels, temporal_hidden
        if M != 8 or kwargs.get('is_causal', True) is False or min(channels, hidden, repeats) < 1:
            raise ValueError('Use eight microphones, causal mode and positive dimensions')
        self.M = M
        self.input_features = 257 * (2 * M + 3 * (M - 1) + 1)
        self.input_norm = ActivationLayerNorm(self.input_features)
        self.encoder = nn.Linear(self.input_features, channels)
        self.encoder_activation = ActivationPReLU(channels)
        self.layers = nn.ModuleList([FullbandTemporalBlock(channels, hidden, dilation)
                                    for _ in range(repeats) for dilation in dilations])
        self.context = sum(layer.history_frames for layer in self.layers)
        self.output_norm = ActivationLayerNorm(channels, seq_last=True)
        self.readout_scale = channels ** -.5
        self.beamformer = nn.Conv1d(channels, 257 * 2 * M, 1)
        self.mapping = nn.Conv1d(channels, 257 * 2, 1)
        for module in (self.beamformer, self.mapping):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
        self.use_bfloat16_training = True

    @staticmethod
    def decompress(value):
        magnitude = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
        return value * torch.where(magnitude < 1e-4, 1e-4, magnitude)

    @staticmethod
    def compress(value):
        magnitude = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
        return value * magnitude.clamp_min(1e-8).pow(-.5)

    def process(self, inpt, state=None):
        if inpt.ndim != 5 or inpt.shape[-3:] != (257, 8, 2) or min(inpt.shape[:2]) < 1:
            raise ValueError('Expected compressed STFT(B,T,257,8,2)')
        cache = state is not None
        with torch.autocast(inpt.device.type, enabled=False):
            spectra = inpt.float()
            magnitude = torch.linalg.vector_norm(spectra, dim=-1)
            scale = magnitude[..., 0].mean(-1, keepdim=True).unsqueeze(-1).clamp_min(1e-4)
            normalized = (spectra / scale.unsqueeze(-1)).flatten(-2)
            unit = spectra / magnitude.clamp_min(1e-4).unsqueeze(-1)
            real = unit[..., 1:, 0] * unit[..., :1, 0] + unit[..., 1:, 1] * unit[..., :1, 1]
            imaginary = unit[..., 1:, 1] * unit[..., :1, 0] - unit[..., 1:, 0] * unit[..., :1, 1]
            ratio = 2 * torch.log((magnitude[..., 1:] + 1e-4) / (magnitude[..., :1] + 1e-4))
            ratio = 5 * torch.tanh(ratio / 5)
            envelope = torch.log1p(magnitude[..., :1] / scale)
            features = torch.cat((normalized, real, imaginary, ratio, envelope), dim=-1).flatten(-2)
        enabled = self.training and self.use_bfloat16_training and inpt.device.type == 'cuda'
        with torch.autocast(inpt.device.type, dtype=torch.bfloat16, enabled=enabled):
            value = self.encoder(self.input_norm(features)).transpose(1, 2)
            value = self.encoder_activation(value)
            histories = []
            for index, layer in enumerate(self.layers):
                history = None if state is None else state.get('layers', [None] * len(self.layers))[index]
                value, updated = layer.process(value, history, cache)
                if cache: histories.append(updated)
            value = self.output_norm(value) * self.readout_scale
            beam = self.beamformer(value).float().transpose(1, 2).reshape(inpt.shape[0], inpt.shape[1], 257, 8, 2)
            mapping = self.mapping(value).float().transpose(1, 2).reshape(inpt.shape[0], inpt.shape[1], 257, 2)
        with torch.autocast(inpt.device.type, enabled=False):
            raw = self.decompress(spectra)
            reference = raw[..., 0, :]
            real = (beam[..., 0] * raw[..., 0] - beam[..., 1] * raw[..., 1]).sum(-1)
            imaginary = (beam[..., 0] * raw[..., 1] + beam[..., 1] * raw[..., 0]).sum(-1)
            correction = self.compress(reference + torch.stack((real, imaginary), -1)) - self.compress(reference)
            output = spectra[..., 0, :] + correction + mapping * scale
        return output.permute(0, 3, 1, 2), {'layers': histories} if cache else None

    def forward(self, inpt, frame_lengths=None):
        if frame_lengths is not None:
            lengths = torch.as_tensor(frame_lengths)
            if lengths.numel() != inpt.shape[0] or torch.any(lengths < 1) or torch.any(lengths > inpt.shape[1]):
                raise ValueError('One valid frame length is required per sample')
        return self.process(inpt)[0]

    def forward_stream(self, inpt, state=None):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError('Streaming requires eval mode and disabled gradients')
        return self.process(inpt, {} if state is None else state)

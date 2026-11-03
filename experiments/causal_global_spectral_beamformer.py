"""Unregistered full-band causal GRU U-Net with physical multichannel complex filtering.

The existing compressed STFT interface is retained. This experiment has no
frozen backbone, external teacher, measured real-data quality or CUDA budget yet.
"""
import torch
from torch import nn

from experiments.causal_monaural_spectral_branch import frequency_block
from light_causal_fast_model import ActivationLayerNorm


class CausalGlobalSpectralBeamformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.M, self.is_causal, self.context = 8, True, 32
        self.encoders = nn.ModuleList(frequency_block(a, b, 2)
            for a, b in zip((38, 32, 48, 64), (32, 48, 64, 96)))
        self.recurrent_norm = ActivationLayerNorm(17*96)
        self.recurrent = nn.GRU(17*96+8, 384, num_layers=3, batch_first=True)
        self.projection = nn.Linear(384, 17*96)
        self.upsamplers = nn.ModuleList(nn.ConvTranspose2d(a, b, (1, 5), stride=(1, 2), padding=(0, 2))
            for a, b in zip((96, 64, 48, 32), (64, 48, 32, 32)))
        self.decoders = nn.ModuleList(frequency_block(a, b)
            for a, b in zip((128, 96, 64, 70), (64, 48, 32, 32)))
        self.output = nn.Conv2d(32, 16, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def process(self, inpt, state=None):
        if (inpt.ndim != 5 or inpt.shape[-3:] != (257, 8, 2) or min(inpt.shape[:2]) < 1
                or inpt.dtype != torch.float32):
            raise ValueError('Expected FP32 compressed STFT(B,T,257,8,2)')
        cache = state is not None
        memories = {} if state is None else state
        if not isinstance(memories, dict) or (memories and set(memories) != {'hidden'}):
            raise ValueError('Use a complete independently owned global spectral state')
        batch, frames = inpt.shape[:2]
        hidden = memories.get('hidden')
        if memories and (not torch.is_tensor(hidden) or hidden.shape != (3, batch, 384)
                         or hidden.dtype != torch.float32 or hidden.device != inpt.device):
            raise ValueError('Global spectral hidden state differs from this stream')
        with torch.autocast(inpt.device.type, enabled=False):
            magnitude = torch.linalg.vector_norm(inpt, dim=-1)
            scale = magnitude[..., :1].mean(-2, keepdim=True).clamp_min(1e-4)
            normalized = (inpt/scale.unsqueeze(-1)).flatten(-2)
            unit = inpt/magnitude.clamp_min(1e-4).unsqueeze(-1)
            real = unit[..., 1:, 0]*unit[..., :1, 0] + unit[..., 1:, 1]*unit[..., :1, 1]
            imaginary = unit[..., 1:, 1]*unit[..., :1, 0] - unit[..., 1:, 0]*unit[..., :1, 1]
            ratio = 2*torch.log((magnitude[..., 1:]+1e-4)/(magnitude[..., :1]+1e-4))
            ratio = 5*torch.tanh(ratio/5)
            envelope = torch.log1p(magnitude[..., :1]/scale)
            features = torch.cat((normalized, real, imaginary, ratio, envelope), -1).permute(0, 3, 1, 2)
            # All eight current microphone levels bypass spectral normalization.
            levels = magnitude.mean(-2).clamp_min(1e-4).log()
        enabled = self.training and inpt.device.type == 'cuda'
        with torch.autocast(inpt.device.type, dtype=torch.bfloat16, enabled=enabled):
            value, skips = features, [features]
            for encoder in self.encoders:
                value = encoder(value)
                skips.append(value)
        with torch.autocast(inpt.device.type, enabled=False):
            sequence = value.permute(0, 2, 1, 3).reshape(batch, frames, 17*96).float()
            sequence = torch.cat((self.recurrent_norm(sequence), levels), -1)
            sequence, hidden = self.recurrent(sequence, hidden)
        with torch.autocast(inpt.device.type, dtype=torch.bfloat16, enabled=enabled):
            value = value + self.projection(sequence).reshape(batch, frames, 96, 17).permute(0, 2, 1, 3)
            for up, decoder, skip in zip(self.upsamplers, self.decoders, reversed(skips[:-1])):
                value = up(value)
                if value.shape[-1] != skip.shape[-1]:
                    raise RuntimeError('Frequency decoder lost alignment')
                value = decoder(torch.cat((value, skip), 1))
            coefficients = self.output(value).float().permute(0, 2, 3, 1).reshape(batch, frames, 257, 8, 2)
        with torch.autocast(inpt.device.type, enabled=False):
            # Undo power0.5 before applying physical complex microphone weights.
            physical = inpt*magnitude.unsqueeze(-1)
            real = (coefficients[..., 0]*physical[..., 0]+coefficients[..., 1]*physical[..., 1]).sum(-1)
            imaginary = (coefficients[..., 0]*physical[..., 1]-coefficients[..., 1]*physical[..., 0]).sum(-1)
            enhanced = physical[..., 0, :]+torch.stack((real, imaginary), -1)
            compressed = enhanced/torch.linalg.vector_norm(enhanced, dim=-1).clamp_min(1e-8).sqrt().unsqueeze(-1)
        output = compressed.permute(0, 3, 1, 2).contiguous()
        return output, dict(hidden=hidden.detach().clone()) if cache else None

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

"""Causal dual-path refinement around an unchanged, fixed causal teacher.

The teacher estimate enters the encoder; learned multichannel filters use only
current/past frames. Zero initialization preserves the complete teacher output.
This is an experimental model, without an author-equivalence or quality claim.
"""
import torch
from torch import nn

from experiments.causal_gated_multichannel_crn import CausalGatedMultichannelCRN, FixedStatisticsAffine
from light_causal_spatial_model import CausalSpatialEnhancer


class CurrentFrameNorm(nn.Module):
    def __init__(self, old):
        super().__init__()
        self.norm = nn.LayerNorm(old.weight.numel())
        self.norm.weight = old.weight
        self.norm.bias = old.bias

    def forward(self, value):
        return self.norm(value.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class TeacherSpatialInput(nn.Conv2d):
    def __init__(self, old):
        super().__init__(47, old.out_channels, old.kernel_size, stride=old.stride, padding=old.padding)
        with torch.no_grad():
            self.weight.zero_()
            self.weight[:, :24].copy_(old.weight)
            self.bias.copy_(old.bias)

    @staticmethod
    def augment(features):
        if features.ndim != 4 or features.shape[1] != 26:
            raise ValueError('Expected24 mixture features and2 teacher components')
        with torch.autocast(features.device.type, enabled=False):
            value = features.float()
            mixture = value[:, :24]
            magnitude = mixture[:, 0::3]
            real = mixture[:, 1::3] / magnitude.clamp_min(1e-4)
            imaginary = mixture[:, 2::3] / magnitude.clamp_min(1e-4)
            cosine = real[:, 1:] * real[:, :1] + imaginary[:, 1:] * imaginary[:, :1]
            sine = imaginary[:, 1:] * real[:, :1] - real[:, 1:] * imaginary[:, :1]
            ratio = 2 * torch.log((magnitude[:, 1:] + 1e-4) / (magnitude[:, :1] + 1e-4))
            ratio = 5 * torch.tanh(ratio / 5)
            augmented = torch.cat((mixture, cosine, sine, ratio, value[:, 24:]), 1)
        return augmented.to(features.dtype)

    def forward(self, value):
        return super().forward(self.augment(value))


class CausalGatedTeacherFusion(nn.Module):
    is_causal = True
    refiner_causal = True

    def __init__(self, teacher, taps=1):
        super().__init__()
        if getattr(teacher, 'is_causal', None) is not True or getattr(teacher, 'M', None) != 8:
            raise ValueError('Use an eight-microphone causal teacher')
        if isinstance(taps, bool) or not isinstance(taps, int) or taps not in (1, 4):
            raise ValueError('Use one current tap or four current/past taps')
        self.teacher = teacher.requires_grad_(False).eval()
        self.teacher.zero_grad(set_to_none=True)
        self.M, self.context, self.taps = 8, max(32, teacher.context), taps
        self.core = CausalGatedMultichannelCRN()
        self.use_bfloat16_training = True

        def normalize(parent):
            for name, child in list(parent.named_children()):
                if isinstance(child, FixedStatisticsAffine):
                    setattr(parent, name, CurrentFrameNorm(child))
                else:
                    normalize(child)

        normalize(self.core)
        with torch.random.fork_rng(devices=[]):
            self.core.down[0][0] = TeacherSpatialInput(self.core.down[0][0])
            if taps == 4:
                old = self.core.output
                output = nn.ConvTranspose2d(64, taps * 16 + 2, (1, 5), stride=(1, 2), padding=(0, 2))
                nn.init.zeros_(output.weight)
                nn.init.zeros_(output.bias)
                self.core.output = output

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def process(self, inpt, state=None):
        if inpt.ndim != 5 or inpt.shape[-3:] != (257, 8, 2) or min(inpt.shape[:2]) < 1:
            raise ValueError('Expected compressed STFT(B,T,257,8,2)')
        memories = {} if state is None else state
        cache = state is not None
        batch, frames = inpt.shape[:2]
        if memories.get('batch_size', batch) != batch:
            raise ValueError('Batch state differs from the stream')
        with torch.autocast(inpt.device.type, enabled=False):
            teacher, teacher_state = self.teacher.process(inpt.float(), memories.get('teacher', {}) if cache else None)
            spectra = inpt.float()
            magnitude = torch.linalg.vector_norm(spectra, dim=-1, keepdim=True)
            features = torch.cat((magnitude, spectra), -1).flatten(-2).permute(0, 3, 1, 2)
            features = torch.cat((features, teacher.float()), 1).contiguous()
        updated = dict(batch_size=batch, teacher=teacher_state, encoder=[], recurrent=[], decoder=[]) if cache else None
        enabled = self.training and self.use_bfloat16_training and inpt.device.type == 'cuda'
        with torch.autocast(inpt.device.type, dtype=torch.bfloat16, enabled=enabled):
            skips = []
            value = features
            for layer in self.core.down:
                value = layer(value)
                skips.append(value)
            for index, layer in enumerate(self.core.encoder):
                value, memory = layer.process(value, memories.get('encoder', [None] * 3)[index], cache)
                skips.append(value)
                if cache:
                    updated['encoder'].append(memory)
            for index, layer in enumerate(self.core.recurrent):
                value, memory = layer.process(value, memories.get('recurrent', [None] * 4)[index], cache)
                if cache:
                    updated['recurrent'].append(memory)
            for index, layer in enumerate(self.core.decoder):
                value, memory = layer.process(value + skips.pop(), memories.get('decoder', [None] * 3)[index], cache)
                if cache:
                    updated['decoder'].append(memory)
            value = self.core.up(value + skips.pop())
            # Bound the readout's initialization scale across hidden channels/taps.
            value = (value + skips.pop()) * (self.core.channels * self.taps) ** -.5
            extra = self.core.output(value).float().permute(0, 2, 3, 1)
        assert not skips
        with torch.autocast(inpt.device.type, enabled=False):
            raw = CausalSpatialEnhancer.decompress(spectra)
            weights = extra[..., :self.taps * 16].reshape(batch, frames, 257, self.taps, 8, 2)
            if self.taps == 1:
                history = raw.unsqueeze(3)
            else:
                past = memories.get('past_spectra')
                shape = (batch, self.taps - 1, 257, 8, 2)
                if past is None:
                    past = raw.new_zeros(shape)
                if past.shape != shape or past.dtype != raw.dtype or past.device != raw.device:
                    raise ValueError('Past-spectrum state differs from the stream')
                joined = torch.cat((past, raw), 1)
                history = joined.unfold(1, self.taps, 1).movedim(-1, 3)
                if cache:
                    updated['past_spectra'] = joined[:, -(self.taps - 1):].detach().clone()
            real = (weights[..., 0] * history[..., 0] - weights[..., 1] * history[..., 1]).sum((-1, -2))
            imaginary = (weights[..., 0] * history[..., 1] + weights[..., 1] * history[..., 0]).sum((-1, -2))
            reference = teacher.float().permute(0, 2, 3, 1)
            teacher_raw = CausalSpatialEnhancer.decompress(reference)
            correction = (CausalSpatialEnhancer.compress(teacher_raw + torch.stack((real, imaginary), -1))
                          - CausalSpatialEnhancer.compress(teacher_raw))
            scale = magnitude[..., 0, :].mean(-2, keepdim=True).clamp_min(1e-4)
            result = reference + correction + extra[..., self.taps * 16:] * scale
        return result.permute(0, 3, 1, 2), updated

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

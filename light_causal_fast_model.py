"""Causal multi-resolution spatial beamforming with bounded streaming caches."""

import torch
from torch import nn
import torch.nn.functional as F

from light_causal_spatial_model import CausalSpatialEnhancer, CausalSpatialLayer


FAST_CANDIDATE = "causal_unet_spatial_v1"


class ActivationPReLU(nn.PReLU):
    def forward(self, value):
        return F.prelu(value, self.weight.to(dtype=value.dtype))


class ActivationLayerNorm(nn.LayerNorm):
    def __init__(self, shape, eps=1e-5, seq_last=False):
        super().__init__(shape, eps=eps)
        self.seq_last = seq_last

    def forward(self, value):
        if self.seq_last: value = value.transpose(-1, 1)
        dtype = value.dtype
        # PyTorch2.5 CUDA rejects BF16 input with FP32 affine parameters.
        with torch.autocast(value.device.type, enabled=False):
            value = F.layer_norm(value.float(), self.normalized_shape, self.weight, self.bias, self.eps).to(dtype)
        return value.transpose(-1, 1) if self.seq_last else value


def activation_norms(module):
    for name, child in tuple(module.named_children()):
        if isinstance(child, nn.LayerNorm):
            replacement = ActivationLayerNorm(child.normalized_shape, child.eps, getattr(child, "seq_last", False))
            replacement.weight, replacement.bias = child.weight, child.bias
            setattr(module, name, replacement)
        else:
            activation_norms(child)


class FrameChannelNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = ActivationLayerNorm(channels)

    def forward(self, value):
        return self.norm(value.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class CausalFrequencyConv(nn.Module):
    def __init__(self, incoming, outgoing, frequency_stride=1, frequency_kernel=3):
        super().__init__()
        self.conv = nn.Conv2d(incoming, outgoing, (3, frequency_kernel),
            stride=(1, frequency_stride), padding=(0, frequency_kernel // 2))
        self.norm = FrameChannelNorm(outgoing)
        self.activation = ActivationPReLU(outgoing)

    def process(self, value, history=None, cache=False):
        if history is None:
            history = value.new_zeros(value.shape[0], value.shape[1], 2, value.shape[3])
        if history.shape != (value.shape[0], value.shape[1], 2, value.shape[3]):
            raise ValueError("Frequency encoder cache differs from input stream")
        joined = torch.cat((history, value), dim=2)
        output = self.activation(self.norm(self.conv(joined)))
        return output, joined[:, :, -2:].detach().clone() if cache else None


class CausalUNetSpatial(nn.Module):
    """Full-resolution skips and a spatial bottleneck; no future time samples."""

    is_causal = True
    refiner_causal = True

    def __init__(self, M=8, *, hidden=128, ffn=256, layers=8, heads=4, context=128, **kwargs):
        super().__init__()
        if M != 8 or kwargs.get("is_causal", True) is False:
            raise ValueError("Use eight microphones and causal mode")
        self.M, self.context = M, context
        self.stem = CausalFrequencyConv(16, 32, frequency_kernel=5)
        widths = (32, 48, 64, 96, hidden)
        self.encoders = nn.ModuleList([CausalFrequencyConv(a, b, 2, 5)
            for a, b in zip(widths[:-1], widths[1:])])
        self.layers = nn.ModuleList()
        full = None
        for _ in range(layers):
            layer = CausalSpatialLayer(hidden, ffn, 17, heads, context, full=full)
            layer.fconv1[-1] = ActivationPReLU(hidden)
            layer.fconv2[-1] = ActivationPReLU(hidden)
            activation_norms(layer)
            self.layers.append(layer)
            full = layer.full
        reverse = tuple(reversed(widths))
        self.upsamplers = nn.ModuleList([nn.ConvTranspose2d(a, b, (1, 4), stride=(1, 2), padding=(0, 1))
            for a, b in zip(reverse[:-1], reverse[1:])])
        self.decoders = nn.ModuleList([CausalFrequencyConv(2 * b, b) for b in reverse[1:]])
        self.beamformer = nn.Conv2d(32, 2 * M, 1)
        nn.init.zeros_(self.beamformer.weight)
        nn.init.zeros_(self.beamformer.bias)
        self.use_bfloat16_training = True

    def process(self, inpt, state=None):
        if inpt.ndim != 5 or inpt.shape[-3:] != (257, self.M, 2) or min(inpt.shape[:2]) < 1:
            raise ValueError("Expected nonempty compressed STFT (B,T,257,8,2)")
        cache = state is not None
        memories = {} if state is None else state
        with torch.autocast(inpt.device.type, enabled=False):
            mixture = CausalSpatialEnhancer.decompress(inpt.float())
            reference = mixture[..., 0, :]
            scale = torch.linalg.vector_norm(reference, dim=-1).mean(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(1e-4)
            features = (mixture.flatten(-2) / scale).permute(0, 3, 1, 2).contiguous()
        updated = {"encoder": [], "middle": [], "decoder": []} if cache else None
        enabled = self.training and self.use_bfloat16_training and inpt.device.type == "cuda"
        with torch.autocast(inpt.device.type, dtype=torch.bfloat16, enabled=enabled):
            value, memory = self.stem.process(features, memories.get("stem"), cache)
            if cache: updated["stem"] = memory
            skips = [value]
            for index, encoder in enumerate(self.encoders):
                history = memories.get("encoder", [None] * len(self.encoders))[index]
                value, memory = encoder.process(value, history, cache)
                if cache: updated["encoder"].append(memory)
                if index < len(self.encoders) - 1: skips.append(value)
            value = value.permute(0, 3, 2, 1)
            for index, layer in enumerate(self.layers):
                if cache:
                    value, memory = layer.process(value, memories.get("middle", [{}] * len(self.layers))[index])
                    updated["middle"].append(memory)
                else:
                    value = layer(value)
            value = value.permute(0, 3, 2, 1)
            for index, (up, decoder, skip) in enumerate(zip(self.upsamplers, self.decoders, reversed(skips))):
                value = up(value)[..., :skip.shape[-1]]
                history = memories.get("decoder", [None] * len(self.decoders))[index]
                value, memory = decoder.process(torch.cat((value, skip), dim=1), history, cache)
                if cache: updated["decoder"].append(memory)
            weights = self.beamformer(value).float().reshape(inpt.shape[0], self.M, 2, inpt.shape[1], 257)
            weights = weights.permute(0, 3, 4, 1, 2)
        with torch.autocast(inpt.device.type, enabled=False):
            real = (weights[..., 0] * mixture[..., 0] - weights[..., 1] * mixture[..., 1]).sum(dim=-1)
            imag = (weights[..., 0] * mixture[..., 1] + weights[..., 1] * mixture[..., 0]).sum(dim=-1)
            enhanced = reference + torch.stack((real, imag), dim=-1)
            correction = CausalSpatialEnhancer.compress(enhanced) - CausalSpatialEnhancer.compress(reference)
            output = (inpt[..., 0, :].float() + correction).permute(0, 3, 1, 2)
        return output, updated

    def forward(self, inpt, frame_lengths=None):
        if frame_lengths is not None:
            lengths = torch.as_tensor(frame_lengths)
            if lengths.numel() != inpt.shape[0] or torch.any(lengths < 1) or torch.any(lengths > inpt.shape[1]):
                raise ValueError("One valid frame length is required per sample")
        return self.process(inpt)[0]

    def forward_stream(self, inpt, state=None):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("Streaming requires eval mode and disabled gradients")
        return self.process(inpt, {} if state is None else state)

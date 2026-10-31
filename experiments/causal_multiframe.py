"""Experimental learned causal multi-frame beamforming; all source weights train."""

import torch
from torch import nn

from light_causal_recurrent65_model import CausalRecurrentUNet65
from light_causal_spatial_model import CausalSpatialEnhancer


class CausalMultiFrame(CausalRecurrentUNet65):
    def __init__(self, M=8, past_frames=4, **kwargs):
        super().__init__(M=M, **kwargs)
        if not isinstance(past_frames, int) or past_frames < 1:
            raise ValueError('Use a positive number of past frames')
        self.past_frames = past_frames
        with torch.random.fork_rng(devices=[]):
            self.past_beamformer = nn.Conv2d(32, 2 * M * past_frames, 1)
            nn.init.zeros_(self.past_beamformer.weight)
            nn.init.zeros_(self.past_beamformer.bias)

    def process(self, inpt, state=None):
        # Keep the source U-Net path and arithmetic unchanged; only the output
        # filter gains learned taps on strictly previous physical mixture frames.
        if inpt.ndim != 5 or inpt.shape[-3:] != (257, self.M, 2) or min(inpt.shape[:2]) < 1:
            raise ValueError('Expected nonempty compressed STFT (B,T,257,8,2)')
        batch, frames = inpt.shape[:2]
        cache = state is not None
        memories = {} if state is None else state
        with torch.autocast(inpt.device.type, enabled=False):
            mixture = CausalSpatialEnhancer.decompress(inpt.float())
            reference = mixture[..., 0, :]
            scale = torch.linalg.vector_norm(reference, dim=-1).mean(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(1e-4)
            features = (mixture.flatten(-2) / scale).permute(0, 3, 1, 2).contiguous()
        updated = {'encoder': [], 'middle': [], 'decoder': []} if cache else None
        enabled = self.training and self.use_bfloat16_training and inpt.device.type == 'cuda'
        with torch.autocast(inpt.device.type, dtype=torch.bfloat16, enabled=enabled):
            value, memory = self.stem.process(features, memories.get('stem'), cache)
            if cache: updated['stem'] = memory
            skips = [value]
            for index, encoder in enumerate(self.encoders):
                history = memories.get('encoder', [None] * len(self.encoders))[index]
                value, memory = encoder.process(value, history, cache)
                if cache: updated['encoder'].append(memory)
                if index < len(self.encoders) - 1: skips.append(value)
            value = value.permute(0, 3, 2, 1)
            for index, layer in enumerate(self.layers):
                if cache:
                    value, memory = layer.process(value, memories.get('middle', [{}] * len(self.layers))[index])
                    updated['middle'].append(memory)
                else:
                    value = layer(value)
            value = value.permute(0, 3, 2, 1)
            for index, (up, decoder, skip) in enumerate(zip(self.upsamplers, self.decoders, reversed(skips))):
                value = up(value)[..., :skip.shape[-1]]
                history = memories.get('decoder', [None] * len(self.decoders))[index]
                value, memory = decoder.process(torch.cat((value, skip), dim=1), history, cache)
                if cache: updated['decoder'].append(memory)
            weights = self.beamformer(value).float().reshape(batch, self.M, 2, frames, 257).permute(0, 3, 4, 1, 2)
            past_weights = self.past_beamformer(value).float().reshape(batch, self.past_frames, self.M, 2, frames, 257)
            past_weights = past_weights.permute(0, 4, 5, 1, 2, 3)
        with torch.autocast(inpt.device.type, enabled=False):
            real = (weights[..., 0] * mixture[..., 0] - weights[..., 1] * mixture[..., 1]).sum(dim=-1)
            imag = (weights[..., 0] * mixture[..., 1] + weights[..., 1] * mixture[..., 0]).sum(dim=-1)
            history = memories.get('past_mixture')
            shape = (batch, self.past_frames, 257, self.M, 2)
            if history is None: history = mixture.new_zeros(shape)
            if history.shape != shape: raise ValueError('Past mixture cache differs from input stream')
            joined = torch.cat((history, mixture), dim=1)
            for tap in range(self.past_frames):
                previous = joined[:, self.past_frames - tap - 1:self.past_frames - tap - 1 + frames]
                w = past_weights[..., tap, :, :]
                real = real + (w[..., 0] * previous[..., 0] - w[..., 1] * previous[..., 1]).sum(dim=-1)
                imag = imag + (w[..., 0] * previous[..., 1] + w[..., 1] * previous[..., 0]).sum(dim=-1)
            enhanced = reference + torch.stack((real, imag), dim=-1)
            correction = CausalSpatialEnhancer.compress(enhanced) - CausalSpatialEnhancer.compress(reference)
            output = (inpt[..., 0, :].float() + correction).permute(0, 3, 1, 2)
            if cache: updated['past_mixture'] = joined[:, -self.past_frames:].detach().clone()
        return output, updated

"""SpatialNet residual experiment around a frozen, already trained quality model.

This is an adaptation, not a reproduction; third_party/spatialnet/SOURCE.json
records the author revision, configuration and interface differences.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from light_quality_model import EaBNetQuality
from third_party.spatialnet.SpatialNet import SpatialNet


SPATIAL_CANDIDATE = "cbam_spatial_refiner_v1"


class SpatialResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = SpatialNet(dim_input=18, dim_output=2, dim_squeeze=8,
            num_layers=8, num_freqs=257, encoder_kernel_size=5, dim_hidden=96,
            dim_ffn=192, num_heads=4, kernel_size=(5, 3), conv_groups=(8, 8),
            norms=("LN", "LN", "GN", "LN", "LN", "LN"), full_share=0)
        self.gradient_checkpointing = True
        nn.init.zeros_(self.network.decoder.weight)
        nn.init.zeros_(self.network.decoder.bias)
        for layer in self.network.layers:
            layer.need_weights = False

    def forward(self, features):
        batch, frequencies, frames, channels = features.shape
        value = self.network.encoder(features.reshape(batch * frequencies, frames, channels)
                                     .transpose(1, 2)).transpose(1, 2)
        value = value.reshape(batch, frequencies, frames, -1)
        for layer in self.network.layers:
            if self.training and self.gradient_checkpointing:
                value, _ = checkpoint(layer, value, use_reentrant=False)
            else:
                value, _ = layer(value)
        return self.network.decoder(value)


class EaBNetSpatialQuality(nn.Module):
    """Mixture-only, noncausal refinement with the unchanged compressed RI API."""

    refiner_causal = False

    def __init__(self, **base_kwargs):
        super().__init__()
        self.base = EaBNetQuality(**base_kwargs).requires_grad_(False).eval()
        self.refiner = SpatialResidual()
        for name in ("M", "embed_dim", "bf_type", "topo_type", "norm_type"):
            setattr(self, name, getattr(self.base, name))
        self.use_bfloat16_training = True

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    @staticmethod
    def decompress(compressed):
        magnitude = torch.linalg.vector_norm(compressed, dim=-1, keepdim=True)
        return compressed * magnitude.clamp_min(1e-4)

    @staticmethod
    def compress(spectrum):
        magnitude = torch.linalg.vector_norm(spectrum, dim=-1, keepdim=True)
        return spectrum * magnitude.clamp_min(1e-8).pow(-0.5)

    def _forward(self, inpt):
        self.base.eval()
        with torch.no_grad(), torch.autocast(inpt.device.type, enabled=False):
            baseline = self.base(inpt.float())
            mixture = self.decompress(inpt.float())
            previous = self.decompress(baseline.permute(0, 2, 3, 1))
            scale = torch.linalg.vector_norm(mixture[..., 0, :], dim=-1).mean(dim=1, keepdim=True)
            scale = scale.clamp_min(1e-4).unsqueeze(-1)
            features = torch.cat((mixture.flatten(-2), previous), dim=-1) / scale
        enabled = self.training and self.use_bfloat16_training and inpt.device.type == "cuda"
        with torch.autocast(inpt.device.type, dtype=torch.bfloat16, enabled=enabled):
            residual = self.refiner(features.transpose(1, 2)).transpose(1, 2).float()
        with torch.autocast(inpt.device.type, enabled=False):
            # Subtract the same compression roundoff to preserve the source at zero head.
            correction = self.compress(previous + scale * residual) - self.compress(previous)
            return baseline + correction.permute(0, 3, 1, 2)

    def forward(self, inpt, frame_lengths=None):
        if inpt.ndim != 5 or inpt.shape[-3:] != (257, self.M, 2):
            raise ValueError("expected compressed STFT shape (B,T,257,8,2)")
        if not inpt.is_floating_point() or min(inpt.shape[:2]) < 1:
            raise ValueError("expected nonempty floating-point STFT")
        if frame_lengths is None:
            return self._forward(inpt)
        lengths = torch.as_tensor(frame_lengths)
        if lengths.ndim != 1 or lengths.numel() != inpt.shape[0] or lengths.dtype not in (
                torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("one integer frame length is required per sample")
        lengths = lengths.detach().cpu().tolist()
        if any(length < 1 or length > inpt.shape[1] for length in lengths):
            raise ValueError("frame lengths must fit the input")
        if len(set(lengths)) == 1:
            return F.pad(self._forward(inpt[:, :lengths[0]]), (0, 0, 0, inpt.shape[1] - lengths[0]))
        return torch.cat([F.pad(self._forward(inpt[i:i + 1, :length]),
            (0, 0, 0, inpt.shape[1] - length)) for i, length in enumerate(lengths)])

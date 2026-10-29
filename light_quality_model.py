"""Frozen EaBNet and an offline complex-spectral residual enhancement model.

The original 8-microphone, power-compressed STFT interface is retained. This
model is a new enhancement experiment, not a paper reproduction candidate.
"""

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from light_reference_variants import build_reference_candidate


QUALITY_CANDIDATE = "cbam_refiner_v1"
BASE_CANDIDATE = "cbam_flat_projection64"


class ChannelLayerNorm(nn.Module):
    """Normalize channels independently at each time-frequency position."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, features: Tensor) -> Tensor:
        return self.norm(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class SpectralConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, frequency_stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, (3, 5), stride=(1, frequency_stride),
        )
        self.norm = ChannelLayerNorm(out_channels)
        self.activation = nn.PReLU(out_channels)

    def forward(self, features: Tensor) -> Tensor:
        features = self.conv(F.pad(features, (2, 2, 2, 0)))
        return self.activation(self.norm(features))


class ResidualSpectralRefiner(nn.Module):
    """Frequency U-Net with bidirectional, full-band recurrent context."""

    def __init__(self):
        super().__init__()
        widths = (18, 32, 48, 64, 96)
        self.encoder = nn.ModuleList([
            SpectralConvBlock(widths[index], widths[index + 1], frequency_stride=2)
            for index in range(4)
        ])
        self.context_in = nn.Linear(96 * 17, 256)
        self.context = nn.GRU(
            256, 128, num_layers=2, batch_first=True, bidirectional=True,
        )
        self.context_out = nn.Linear(256, 96 * 17)
        self.decoder = nn.ModuleList([
            SpectralConvBlock(96 + 64, 64),
            SpectralConvBlock(64 + 48, 48),
            SpectralConvBlock(48 + 32, 32),
            SpectralConvBlock(32, 32),
        ])
        self.head = nn.Conv2d(32, 2, kernel_size=1)
        # Preserve the loaded baseline exactly before the first optimizer step.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 4 or features.shape[1] != 18 or features.shape[-1] != 257:
            raise ValueError("refiner expects features shaped (B, 18, T, 257)")
        original_size = features.shape[-2:]
        skips = []
        for encoder in self.encoder:
            features = encoder(features)
            skips.append(features)

        batch_size, channels, frames, frequencies = features.shape
        sequence = features.permute(0, 2, 1, 3).reshape(batch_size, frames, -1)
        sequence, _ = self.context(self.context_in(sequence))
        sequence = self.context_out(sequence)
        features = features + sequence.reshape(batch_size, frames, channels, frequencies).permute(0, 2, 1, 3)

        for decoder, skip in zip(self.decoder[:3], reversed(skips[:-1])):
            features = F.interpolate(features, size=skip.shape[-2:], mode="nearest")
            features = decoder(torch.cat((features, skip), dim=1))
        features = F.interpolate(features, size=original_size, mode="nearest")
        return self.head(self.decoder[-1](features))


class EaBNetQuality(nn.Module):
    """Add an input-scaled RI residual to a permanently frozen baseline.

    Both input and output use the original power-compressed complex spectrum.
    Bidirectional context makes the new refiner explicitly non-causal.
    """

    refiner_causal = False

    def __init__(self, **base_kwargs):
        super().__init__()
        self.base = build_reference_candidate(BASE_CANDIDATE, **base_kwargs)
        self.base.requires_grad_(False)
        self.base.eval()
        self.refiner = ResidualSpectralRefiner()
        self.M = self.base.M
        self.embed_dim = self.base.embed_dim
        self.bf_type = self.base.bf_type
        self.topo_type = self.base.topo_type
        self.norm_type = self.base.norm_type

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def _forward(self, inpt: Tensor) -> Tensor:
        self.base.eval()
        # AMP applies to the trainable refiner; preserve the baseline in FP32.
        with torch.no_grad(), torch.autocast(device_type=inpt.device.type, enabled=False):
            baseline = self.base(inpt.float())
        batch_size, frames, frequencies, _, _ = inpt.shape
        mixture_features = inpt.float().reshape(batch_size, frames, frequencies, 16).permute(0, 3, 1, 2)
        residual = self.refiner(torch.cat((mixture_features, baseline), dim=1))
        scale = torch.linalg.vector_norm(inpt.float(), dim=-1).mean(dim=-1).unsqueeze(1)
        return baseline + scale * residual

    def forward(self, inpt: Tensor, frame_lengths=None) -> Tensor:
        if not isinstance(inpt, Tensor) or inpt.ndim != 5 or inpt.shape[-3:] != (257, self.M, 2):
            raise ValueError("expected STFT shape (B, T, 257, 8, 2)")
        if not inpt.is_floating_point() or inpt.shape[0] < 1 or inpt.shape[1] < 1:
            raise ValueError("expected a nonempty floating-point STFT tensor")
        if frame_lengths is None:
            return self._forward(inpt)

        lengths = torch.as_tensor(frame_lengths)
        if lengths.ndim != 1 or lengths.numel() != inpt.shape[0]:
            raise ValueError("frame_lengths must contain one integer per sample")
        if lengths.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("frame_lengths must contain integers")
        lengths = lengths.detach().cpu().tolist()
        frames = inpt.shape[1]
        if any(length < 1 or length > frames for length in lengths):
            raise ValueError("frame_lengths must lie between 1 and the input frame count")
        if len(set(lengths)) == 1:
            return F.pad(self._forward(inpt[:, :lengths[0]]), (0, 0, 0, frames - lengths[0]))

        # CBAM pooling and the BiGRU must never see another sample's padding.
        return torch.cat([
            F.pad(self._forward(inpt[index:index + 1, :length]), (0, 0, 0, frames - length))
            for index, length in enumerate(lengths)
        ], dim=0)


def build_quality_model(**base_kwargs) -> EaBNetQuality:
    return EaBNetQuality(**base_kwargs)

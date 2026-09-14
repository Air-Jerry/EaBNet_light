"""Paper-text implementation of Li et al., TASLP 2021, DOI 10.1109/TASLP.2021.3079813.

Equations (7)-(11), (17)-(21), Figs. 1-2 and Table I are the source of truth.
The existing EaBNet model is a different, multichannel architecture.

IN is the normalization explicitly named in the paper. Standard IN uses
whole-utterance statistics even when the convolutions are causal. ``cIN``
is an explicitly separate cumulative variant for causality experiments;
it must not be represented as an author-confirmed detail of the paper.
"""

from typing import Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CumulativeInstanceNorm(nn.Module):
    """Per-channel IN over observed frames (and frequency for 2-D inputs)."""

    def __init__(self, channels: int, dimensions: int, eps: float = 1e-5):
        super().__init__()
        self.dimensions = dimensions
        self.eps = eps
        shape = (1, channels, 1) if dimensions == 1 else (1, channels, 1, 1)
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        y = x.float()
        n = torch.arange(1, y.shape[2] + 1, device=y.device, dtype=y.dtype)
        if self.dimensions == 2:
            count = n.view(1, 1, -1, 1) * y.shape[3]
            total = y.sum(3, keepdim=True).cumsum(2)
            square = y.square().sum(3, keepdim=True).cumsum(2)
        else:
            count = n.view(1, 1, -1)
            total, square = y.cumsum(2), y.square().cumsum(2)
        mean = total / count
        variance = (square / count - mean.square()).clamp_min(0)
        return ((y - mean) * torch.rsqrt(variance + self.eps) * self.weight + self.bias).to(dtype)


def _norm(channels: int, dimensions: int, norm_type: str) -> nn.Module:
    if norm_type == "IN":
        cls = nn.InstanceNorm1d if dimensions == 1 else nn.InstanceNorm2d
        return cls(channels, eps=1e-5, affine=True, track_running_stats=False)
    if norm_type == "cIN":
        return CumulativeInstanceNorm(channels, dimensions)
    raise ValueError("norm_type must be IN (paper text) or cIN (explicit cumulative variant)")


class ConvGLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_frequency: int, norm_type: str):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 2 * out_channels, (2, kernel_frequency), (1, 2))
        self.norm = _norm(out_channels, 2, norm_type)
        self.activation = nn.PReLU(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        value, gate = self.conv(F.pad(x, (0, 0, 1, 0))).chunk(2, dim=1)
        return self.activation(self.norm(value * gate.sigmoid()))


class DeconvGLU(nn.Module):
    def __init__(self, out_channels: int, kernel_frequency: int, norm_type: str):
        super().__init__()
        self.conv = nn.ConvTranspose2d(128, 2 * out_channels, (2, kernel_frequency), (1, 2))
        self.norm = _norm(out_channels, 2, norm_type)
        self.activation = nn.PReLU(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        # A 2-frame transposed kernel produces T+1 frames; trim the last one.
        value, gate = self.conv(x)[:, :, :-1, :].chunk(2, dim=1)
        return self.activation(self.norm(value * gate.sigmoid()))


class DilatedBranch(nn.Module):
    def __init__(self, dilation: int, is_causal: bool, norm_type: str):
        super().__init__()
        self.left = 4 * dilation if is_causal else 2 * dilation
        self.right = 0 if is_causal else 2 * dilation
        self.activation = nn.PReLU(64)
        self.norm = _norm(64, 1, norm_type)
        # Regular convolutions, NOT depthwise, and an independent gated branch.
        self.conv = nn.Conv1d(64, 64, kernel_size=5, dilation=dilation)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.pad(self.norm(self.activation(x)), (self.left, self.right)))


class SqueezedTCM(nn.Module):
    def __init__(self, dilation: int, is_causal: bool, norm_type: str):
        super().__init__()
        self.input_conv = nn.Conv1d(256, 64, kernel_size=1)
        self.main_branch = DilatedBranch(dilation, is_causal, norm_type)
        self.gate_branch = DilatedBranch(dilation, is_causal, norm_type)
        self.activation = nn.PReLU(64)
        self.norm = _norm(64, 1, norm_type)
        self.output_conv = nn.Conv1d(64, 256, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        squeezed = self.input_conv(x)
        y = self.main_branch(squeezed) * self.gate_branch(squeezed).sigmoid()
        return x + self.output_conv(self.norm(self.activation(y)))


class Encoder(nn.Module):
    def __init__(self, input_channels: int, norm_type: str):
        super().__init__()
        self.layers = nn.ModuleList([
            ConvGLU(input_channels if i == 0 else 64, 64, k, norm_type)
            for i, k in enumerate((5, 3, 3, 3, 3))
        ])

    def forward(self, x: Tensor) -> Tuple[Tensor, list]:
        skips = []
        for layer in self.layers:
            x = layer(x)
            skips.append(x)
        # (B,64,T,4) -> (B,256,T); preserve the channel/frequency ordering.
        return x.transpose(2, 3).flatten(1, 2), skips


class Decoder(nn.Module):
    def __init__(self, magnitude: bool, norm_type: str):
        super().__init__()
        self.layers = nn.ModuleList([
            DeconvGLU(1 if i == 4 else 64, k, norm_type)
            for i, k in enumerate((3, 3, 3, 3, 5))
        ])
        self.linear = nn.Linear(161, 161)
        self.output_activation = nn.Softplus() if magnitude else nn.Identity()

    def forward(self, x: Tensor, skips: list) -> Tensor:
        x = x.reshape(x.shape[0], 64, 4, x.shape[-1]).transpose(2, 3)
        for layer, skip in zip(self.layers, reversed(skips)):
            if x.shape[-2:] != skip.shape[-2:]:
                raise RuntimeError("Table I decoder/skip shape mismatch; cropping is not a valid repair")
            x = layer(torch.cat((x, skip), dim=1))
        return self.output_activation(self.linear(x.squeeze(1)))


class SpectralNetwork(nn.Module):
    def __init__(self, magnitude: bool, is_causal: bool, norm_type: str):
        super().__init__()
        self.encoder = Encoder(1 if magnitude else 4, norm_type)
        # Every group and every stage has its own parameters; no weight sharing.
        self.tcm_groups = nn.ModuleList([
            nn.Sequential(*(SqueezedTCM(d, is_causal, norm_type) for d in (1, 2, 4, 8, 16, 32)))
            for _ in range(3)
        ])
        self.decoders = nn.ModuleList([Decoder(magnitude, norm_type) for _ in range(1 if magnitude else 2)])

    def forward(self, x: Tensor) -> Tensor:
        x, skips = self.encoder(x)
        for group in self.tcm_groups:
            x = group(x)
        return torch.stack([decoder(x, skips) for decoder in self.decoders], dim=1)


class CTSNet(nn.Module):
    """161-bin, monaural CTS-Net; the public RI interface matches the old scripts.

    Input: (B,T,161,1,2). Default output: (B,2,T,161).
    Training may request (B,3,T,161), where channel 2 is the ME magnitude.
    This carries the auxiliary loss through DP/DDP without mutable caches.
    """

    def __init__(self, is_causal: bool = True, norm_type: str = "IN", return_auxiliary: bool = False):
        super().__init__()
        self.M = 1
        self.is_causal = is_causal
        self.norm_type = norm_type
        self.return_auxiliary = return_auxiliary
        self.stage = "joint"
        self.me_net = SpectralNetwork(True, is_causal, norm_type)
        self.cs_net = SpectralNetwork(False, is_causal, norm_type)

    def set_stage(self, stage: str) -> None:
        if stage not in ("me", "joint"):
            raise ValueError("stage must be me or joint")
        self.stage = stage
        self.cs_net.requires_grad_(stage == "joint")

    def forward_stages(self, inputs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if inputs.ndim != 5 or inputs.shape[2:] != (161, 1, 2) or inputs.shape[1] < 2:
            raise ValueError("CTS-Net requires input (B,T,161,1,2), T>=2; use a 320-point FFT and one mic")
        noisy = inputs[..., 0, :].permute(0, 3, 1, 2).float()
        magnitude = torch.linalg.vector_norm(noisy, dim=1, keepdim=True)
        estimated_magnitude = self.me_net(magnitude).float()
        # At exactly zero noisy energy define phase=0, as atan2(0,0) does,
        # but avoid atan2's undefined derivative there.
        safe_magnitude = torch.where(magnitude > 0, magnitude, torch.ones_like(magnitude))
        unit_phase = noisy / safe_magnitude
        zero_phase = torch.cat((torch.ones_like(magnitude), torch.zeros_like(magnitude)), dim=1)
        unit_phase = torch.where(magnitude > 0, unit_phase, zero_phase)
        coarse = estimated_magnitude * unit_phase
        residual = self.cs_net(torch.cat((noisy, coarse), dim=1)).float() if self.stage == "joint" else torch.zeros_like(coarse)
        return estimated_magnitude, coarse, coarse + residual

    def forward(self, inputs: Tensor, frame_lengths: Tensor = None) -> Tensor:
        if frame_lengths is not None:
            lengths = frame_lengths.detach().cpu().tolist()
            if len(lengths) != inputs.shape[0] or any(n < 2 or n > inputs.shape[1] or int(n) != n for n in lengths):
                raise ValueError("frame_lengths must contain one valid frame count >=2 per utterance")
            # IN includes time in its statistics. Padded frames would otherwise
            # change even the valid predictions when utterances are batched.
            if any(n != inputs.shape[1] for n in lengths):
                outputs = [
                    F.pad(self.forward(inputs[i:i+1, :int(n)]), (0, 0, 0, inputs.shape[1] - int(n)))
                    for i, n in enumerate(lengths)
                ]
                return torch.cat(outputs, dim=0)
        magnitude, _, refined = self.forward_stages(inputs)
        return torch.cat((refined, magnitude), dim=1) if self.return_auxiliary else refined


def cts_loss(
    estimate: Tensor, label: Tensor, frame_list: Sequence[int], stage: str = "joint",
    alpha: float = 0.5, lambda_me: float = 0.1,
) -> Tensor:
    """Equations (17)-(21), normalized by valid T*F (not by 2*T*F for RI).

    A common mean over valid TF cells fixes the scale without changing the
    relative coefficients. The paper writes Frobenius sums, without specifying
    a minibatch reduction; this normalization convention is recorded explicitly.
    """
    if estimate.ndim != 4 or estimate.shape[1] != 3:
        raise ValueError("cts_loss requires auxiliary output (B,3,T,F)")
    if label.shape != (estimate.shape[0], 2, estimate.shape[2], estimate.shape[3]):
        raise ValueError("label must have shape (B,2,T,F) matching estimate")
    if stage not in ("me", "joint") or not 0 <= alpha <= 1 or lambda_me < 0:
        raise ValueError("Invalid stage or loss coefficients")
    lengths = torch.as_tensor(frame_list, device=estimate.device)
    if lengths.shape != (estimate.shape[0],) or torch.any(lengths <= 0) or torch.any(lengths > estimate.shape[2]):
        raise ValueError("frame_list must contain one positive valid frame count per sample")
    valid = torch.arange(estimate.shape[2], device=estimate.device)[None, :, None] < lengths[:, None, None]
    # Remove padding BEFORE norm/subtraction, so padded NaNs cannot pollute loss/gradients.
    prediction = estimate.float().masked_fill(~valid[:, None], 0)
    target = label.float().masked_fill(~valid[:, None], 0)
    count = lengths.sum() * estimate.shape[3]
    target_magnitude = torch.linalg.vector_norm(target, dim=1)
    me = (prediction[:, 2] - target_magnitude).square().sum() / count
    if stage == "me":
        return me
    ri = (prediction[:, :2] - target).square().sum() / count
    mag = (torch.linalg.vector_norm(prediction[:, :2], dim=1) - target_magnitude).square().sum() / count
    return alpha * ri + (1.0 - alpha) * mag + lambda_me * me

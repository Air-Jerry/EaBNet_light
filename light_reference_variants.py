"""Source-derived diagnostics, never an author-verified FCAE implementation.

CBAM ChannelGate follows reference [13]'s author code, including its shared
two-layer MLP and both biases.  It deliberately differs from target Eq. (11).
The last candidate follows the real-valued memory primitive in reference
[17]'s public implementation.  Combining either reference with the target
network remains a hypothesis.  The default model and trainer are untouched.
"""

from copy import deepcopy

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from EaBNet_light import ChannelAttention, EaBNet
from light_structure_variants import StructureConfig, build_candidate


CBAM_SOURCE = {
    "url": "https://github.com/Jongchan/attention-module/blob/"
    "459efad0e05ee7dde50c41ca10a3d0800bc3792a/MODELS/cbam.py",
    "commit": "459efad0e05ee7dde50c41ca10a3d0800bc3792a",
    "sha256": "6a2115f71541c77ab09c666e4cf32ab9928520f9c2bba9c1ceab8edc8fdd8ac5",
    "class": "ChannelGate",
    "lines": [26, 60],
}
FRCRN_SOURCE = {
    "url": "https://github.com/modelscope/ClearerVoice-Studio/blob/main/"
    "clearvoice/clearvoice/models/frcrn_se/complex_nn.py",
    "sha256": "3dcda8502c6d588493a59dcb0910624a088be3e1c8b82b9d4b9408e1c5f3b5cb",
    "class": "UniDeepFsmn",
    "lines": [5, 60],
    "qualification": "Public maintained FRCRN source, not the target FCAE author code.",
}

_CANDIDATES = (
    "literal_per_frequency",
    "literal_flat_projection64",
    "cbam_per_frequency",
    "cbam_flat_projection64",
    "cbam_flat_hidden64",
)


def all_reference_candidates() -> tuple:
    return _CANDIDATES


class CBAMChannelAttention(nn.Module):
    """Reference [13] ChannelGate only; retain target spatial attention.

    Both pooled vectors pass through the *same complete* MLP before their
    outputs are added.  Consequently the final bias participates twice,
    exactly as in the author's code, and is stored only once.
    """

    def __init__(self, channels: int = 64, reduction_ratio: int = 16):
        super().__init__()
        if type(channels) is not int or channels < 1:
            raise ValueError("channels must be a positive integer")
        if type(reduction_ratio) is not int or not 1 <= reduction_ratio <= channels:
            raise ValueError("reduction_ratio must be an integer between 1 and channels")
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=True),
            nn.ReLU(),
            nn.Linear(channels // reduction_ratio, channels, bias=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError("CBAM attention expects (B, channels, T, F)")
        mean = x.mean(dim=(-2, -1))
        maximum = x.amax(dim=(-2, -1))
        weights = torch.sigmoid(self.mlp(mean) + self.mlp(maximum))
        return x * weights[:, :, None, None]


class FRCRNMemory(nn.Module):
    """Real-valued UniDeepFsmn rule, applied to flattened C*F frame vectors.

    Source [17]: D -> H affine/ReLU -> D bias-free projection -> depthwise
    memory + projected vector + input vector.  Here H=64 is the hidden size
    and D=448 the projection/output size, unlike a 64-wide projection.  The
    memory convolution has 20 taps, including the present frame; it sees 19
    past frames.  No previous-layer memory is added by this reference rule.
    """

    def __init__(
        self, channels: int = 64, frequencies: int = 7,
        hidden_units: int = 64, memory_size: int = 20,
    ):
        super().__init__()
        for name, value in (
            ("channels", channels), ("frequencies", frequencies),
            ("hidden_units", hidden_units), ("memory_size", memory_size),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.feature_channels = channels
        self.frequencies = frequencies
        self.hidden_units = hidden_units
        self.memory_size = memory_size
        self.is_causal = True
        self.right_memory_size = 0
        features = channels * frequencies
        self.linear = nn.Linear(features, hidden_units, bias=True)
        self.project = nn.Linear(hidden_units, features, bias=False)
        self.conv = nn.Conv1d(
            features, features, kernel_size=memory_size, groups=features, bias=False,
        )

    def forward(
        self, x: Tensor, previous_memory: Tensor = None, *, return_memory: bool = False,
    ):
        if x.ndim != 4:
            raise ValueError("FRCRN memory expects (B, C, T, F)")
        batch, channels, frames, frequencies = x.shape
        if (channels, frequencies) != (self.feature_channels, self.frequencies):
            raise ValueError("FRCRN memory input channels/frequencies do not match its layout")
        # c*F+f is the feature index, and each time step remains separate.
        inputs = x.permute(0, 2, 1, 3).reshape(batch, frames, channels * frequencies)
        projected = self.project(torch.relu(self.linear(inputs)))
        memory = self.conv(F.pad(projected.transpose(1, 2), (self.memory_size - 1, 0)))
        output = inputs + projected + memory.transpose(1, 2)
        output = output.reshape(batch, frames, channels, frequencies).permute(0, 2, 1, 3)
        output = output.contiguous()
        # CRED uses one shared cell three times; the reference adds each call's
        # input residual, not the reference-[16] previous-memory connection.
        return (output, None) if return_memory else output


def _replace_channel_attention(parent: nn.Module) -> int:
    count = 0
    for name, child in list(parent.named_children()):
        if isinstance(child, ChannelAttention):
            replacement = CBAMChannelAttention(child.conv.out_channels).to(
                device=child.conv.weight.device, dtype=child.conv.weight.dtype,
            )
            setattr(parent, name, replacement)
            count += 1
        else:
            count += _replace_channel_attention(child)
    return count


def build_reference_candidate(
    candidate_id: str, *, without_skip_attention: bool = False, **model_kwargs,
) -> EaBNet:
    """Build a labeled diagnostic using the existing public STFT interface."""
    if candidate_id not in _CANDIDATES:
        raise ValueError(f"unknown reference candidate: {candidate_id}")
    if type(without_skip_attention) is not bool:
        raise ValueError("without_skip_attention must be a bool")
    # The five stable identifiers describe these exact diagnostic dimensions.
    # Custom widths belong in the original model, not in metadata claiming64/448.
    for name, expected in {
        "c": 64, "embed_dim": 64, "cd1": 64, "M": 8,
        "dfsmn_layers": 3, "dfsmn_memory_size": 20, "norm_type": "BN",
        "bf_type": "lstm", "topo_type": "mimo", "is_causal": True,
        "intra_connect": "cat",
    }.items():
        if name in model_kwargs and model_kwargs[name] != expected:
            raise ValueError(f"reference candidates require {name}={expected}")

    layout = "flattened" if candidate_id.endswith("flat_projection64") else "per_frequency"
    model = build_candidate(
        StructureConfig(dfsmn_layout=layout),
        without_skip_attention=without_skip_attention, **model_kwargs,
    )
    if candidate_id.startswith("cbam_"):
        replacements = _replace_channel_attention(model.cred)
        expected = 10 if without_skip_attention else 15
        if replacements != expected:
            raise RuntimeError(f"expected {expected} channel attention replacements, got {replacements}")
    if candidate_id == "cbam_flat_hidden64":
        old = model.cred.dfsmn
        model.cred.dfsmn = FRCRNMemory().to(
            device=old.in_conv.weight.device, dtype=old.in_conv.weight.dtype,
        )
    return model


def reference_candidate_metadata(candidate_id: str) -> dict:
    """Explain source support separately from compatibility with target text."""
    if candidate_id not in _CANDIDATES:
        raise ValueError(f"unknown reference candidate: {candidate_id}")
    cbam = candidate_id.startswith("cbam_")
    flat_projection = candidate_id.endswith("flat_projection64")
    frcrn = candidate_id == "cbam_flat_hidden64"
    metadata = {
        "candidate_id": candidate_id,
        "author_confirmed": False,
        "exact_reproduction_verified": False,
        "target_equation_11_matches": not cbam,
        "equation_deviations": [
            "Target Eq.11 concatenates pooled descriptors before one convolution; "
            "CBAM sends them separately through a shared two-layer nonlinear MLP and sums."
        ] if cbam else [],
        "channel_attention": "reference_13_cbam_r16_bias_true" if cbam else "target_eq11_dense_concat",
        "dfsmn_layout": "flattened" if flat_projection or frcrn else "per_frequency",
        "dfsmn_rule": "reference_17_input_residual" if frcrn else "reference_16_memory_carry",
        "dfsmn_hidden_units": 64 if frcrn or not flat_projection else 448,
        "dfsmn_projection_units": 448 if frcrn else 64,
        "dfsmn_memory_taps_including_present": 20 if frcrn else 21,
        "dfsmn_past_frames": 19 if frcrn else 20,
        "sources": ([deepcopy(CBAM_SOURCE)] if cbam else []) + ([deepcopy(FRCRN_SOURCE)] if frcrn else []),
        "unresolved": [
            "Target authors do not specify channel-attention reduction, sharing or biases.",
            "Target authors do not specify DFSMN tensor layout, projection width or memory order.",
            "Rounded parameter/MAC agreement cannot establish structural equivalence.",
            "Combining cited modules with this target is not evidence of the authors' implementation.",
        ],
    }
    if flat_projection:
        metadata["unresolved"].append(
            "448-wide output activations and a 64-wide projection do not establish the target's meaning of 64 hidden units."
        )
    if frcrn:
        metadata["unresolved"].append(
            "Public FRCRN has evolved since its paper; its real cell and C*F adaptation are diagnostic references."
        )
    return metadata


candidate_metadata = reference_candidate_metadata

__all__ = [
    "CBAMChannelAttention", "FRCRNMemory", "all_reference_candidates",
    "build_reference_candidate", "reference_candidate_metadata", "candidate_metadata",
]

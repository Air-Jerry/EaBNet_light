"""Isolated diagnostic hypotheses for the unresolved lightweight architecture.

These variants are NOT author-confirmed implementations.  Matching a rounded
parameter count does not establish architectural or experimental equivalence.
Nothing in this module changes the default model, trainer, or data interface.

The four axes are deliberately narrow:
* grouped CA tests connectivity, retaining the original all-average followed
  by all-maximum concatenation (there is no hidden channel interleaving);
* addition tests skip fusion, with C rather than 2C decoder input channels;
* shared skips reuse one attention object at all five resolutions;
* flattened DFSMN tests one C*F sequence rather than F independent sequences.
  Its 448 -> 64 -> 448 mapping uses a 64-wide projection.  This is not evidence
  that it preserves the paper's meaning of "64 hidden units".

The default configuration is exactly EaBNet_light.EaBNet, including random
initialization, state_dict keys, and output values.  All other configurations
must remain labeled hypotheses until supported by source evidence.
"""

from dataclasses import dataclass
from itertools import product
from typing import List

import torch
from torch import Tensor, nn

from EaBNet_light import CRED, ChannelAttention, EaBNet, SharedDFSMN, _match_tf


@dataclass(frozen=True)
class StructureConfig:
    """Explicit hypothesis axes, with stable identifiers for result files."""

    ca_groups: int = 1
    skip_fusion: str = "cat"
    share_skip_attention: bool = False
    dfsmn_layout: str = "per_frequency"

    def __post_init__(self) -> None:
        if type(self.ca_groups) is not int or self.ca_groups not in (1, 2, 4):
            raise ValueError("ca_groups must be an integer in {1, 2, 4}")
        if self.skip_fusion not in ("cat", "add"):
            raise ValueError("skip_fusion must be 'cat' or 'add'")
        if type(self.share_skip_attention) is not bool:
            raise ValueError("share_skip_attention must be a bool")
        if self.dfsmn_layout not in ("per_frequency", "flattened"):
            raise ValueError("dfsmn_layout must be 'per_frequency' or 'flattened'")

    @property
    def candidate_id(self) -> str:
        sharing = "shared" if self.share_skip_attention else "independent"
        return (
            f"ca{self.ca_groups}_{self.skip_fusion}_"
            f"skip-{sharing}_dfsmn-{self.dfsmn_layout}"
        )


def all_configs() -> List[StructureConfig]:
    """Return the 24 diagnostic combinations in deterministic order."""
    return [
        StructureConfig(*values)
        for values in product(
            (1, 2, 4), ("cat", "add"), (False, True),
            ("per_frequency", "flattened"),
        )
    ]


class FlattenedDFSMN(SharedDFSMN):
    """Hypothesis: channel-major C*F vectors, using the existing memory rule.

    C is the outer feature index and F the inner one: flattened index c*F+f.
    Batch and time never enter the feature index.  Returned memory has shape
    (B, projection_width, T), so the inherited inter-depth memory connection
    is reused unchanged.  CRED still invokes this single object three times.
    """

    def __init__(
        self,
        channels: int = 64,
        frequencies: int = 7,
        hidden_units: int = 64,
        memory_size: int = 20,
        norm_type: str = "BN",
        is_causal: bool = True,
        right_memory_size: int = 0,
    ):
        if channels < 1 or frequencies < 1:
            raise ValueError("flattened DFSMN requires positive C and F")
        super().__init__(
            channels=channels * frequencies,
            hidden_units=hidden_units,
            memory_size=memory_size,
            norm_type=norm_type,
            is_causal=is_causal,
            right_memory_size=right_memory_size,
        )
        self.feature_channels = channels
        self.frequencies = frequencies

    def forward(
        self, x: Tensor, previous_memory: Tensor = None, *, return_memory: bool = False
    ):
        if x.ndim != 4:
            raise ValueError("flattened DFSMN expects (B, C, T, F)")
        batch, channels, frames, frequencies = x.shape
        if (channels, frequencies) != (self.feature_channels, self.frequencies):
            raise ValueError(
                "flattened DFSMN expected C,F="
                f"{self.feature_channels},{self.frequencies}; got {channels},{frequencies}"
            )
        flattened = x.permute(0, 1, 3, 2).reshape(
            batch, channels * frequencies, frames, 1
        )
        output, memory = super().forward(
            flattened, previous_memory, return_memory=True
        )
        output = output.reshape(batch, channels, frequencies, frames)
        output = output.permute(0, 1, 3, 2).contiguous()
        return (output, memory) if return_memory else output


class _AdditiveCRED(CRED):
    """Reuse existing child modules and change only skip fusion in forward."""

    def __init__(self, source: CRED):
        nn.Module.__init__(self)
        for name, child in source.named_children():
            self.add_module(name, child)
        self.dfsmn_layers = source.dfsmn_layers

    def forward(self, x: Tensor) -> Tensor:
        input_ref = x
        skips = []
        for encoder in self.encoder:
            x = encoder(x)
            skips.append(x)

        memory = None
        for _ in range(self.dfsmn_layers):
            x, memory = self.dfsmn(x, memory, return_memory=True)

        for index, decoder in enumerate(self.decoder):
            skip = self.skip_attention[-(index + 1)](skips[-(index + 1)])
            x = _match_tf(x, skip) + skip
            target = skips[-(index + 2)] if index + 1 < len(skips) else input_ref
            x = decoder(x, target=target)
        return self.out_conv(x)


def _group_channel_attention(attention: ChannelAttention, groups: int) -> None:
    """Retain the corresponding dense weight blocks as a controlled initial state.

    This is native grouped convolution on [avg_0..avg_C, max_0..max_C], not
    per-channel pairing.  At groups=2, one output group sees only average
    pooling and the other only maximum pooling.  This is an intentionally
    visible connectivity hypothesis, not an inferred paper detail.
    """
    old = attention.conv
    if old.in_channels % groups or old.out_channels % groups:
        raise ValueError(f"CA channels must be divisible by ca_groups={groups}")
    replacement = nn.Conv2d(
        old.in_channels, old.out_channels, kernel_size=1, groups=groups,
        bias=old.bias is not None, device=old.weight.device, dtype=old.weight.dtype,
    )
    inputs_per_group = old.in_channels // groups
    outputs_per_group = old.out_channels // groups
    with torch.no_grad():
        for group in range(groups):
            out_slice = slice(group * outputs_per_group, (group + 1) * outputs_per_group)
            in_slice = slice(group * inputs_per_group, (group + 1) * inputs_per_group)
            replacement.weight[out_slice].copy_(old.weight[out_slice, in_slice])
        if old.bias is not None:
            replacement.bias.copy_(old.bias)
    attention.conv = replacement


def _use_additive_decoder_inputs(cred: CRED) -> None:
    """Replace only FCAD's input transposed convolution, preserving its wrapper.

    The two input-channel weight halves are averaged for initialization.  This
    preserves the original pre-activation when both incoming features agree;
    it is a diagnostic initialization convention, not a paper specification.
    """
    for decoder in cred.decoder:
        wrapper = decoder.in_deconv[0]
        container = wrapper.deconv
        old = container[0] if isinstance(container, nn.Sequential) else container
        if not isinstance(old, nn.ConvTranspose2d) or old.groups != 1:
            raise ValueError("additive decoder expects an ungrouped ConvTranspose2d")
        channels = old.out_channels
        if old.in_channels != 2 * channels:
            raise ValueError("additive decoder expects concatenated equal-width inputs")
        replacement = nn.ConvTranspose2d(
            channels, channels, old.kernel_size, stride=old.stride,
            padding=old.padding, output_padding=old.output_padding,
            groups=old.groups, bias=old.bias is not None, dilation=old.dilation,
            padding_mode=old.padding_mode, device=old.weight.device,
            dtype=old.weight.dtype,
        )
        with torch.no_grad():
            replacement.weight.copy_(
                (old.weight[:channels] + old.weight[channels:]) * 0.5
            )
            if old.bias is not None:
                replacement.bias.copy_(old.bias)
        if isinstance(container, nn.Sequential):
            container[0] = replacement
        else:
            wrapper.deconv = replacement


def build_candidate(
    config: StructureConfig, *, without_skip_attention: bool = False, **model_kwargs
) -> EaBNet:
    """Build a hypothesis while retaining EaBNet's public STFT contract.

    ``model_kwargs`` are passed unchanged to the original constructor.  With
    the default config and no ablation this performs no module replacements
    and consumes exactly the original constructor's random draws.

    ``without_skip_attention`` removes its parameters with five Identities;
    it does not zero them or retain an unused attention object in the model.
    Standard ``model.parameters()`` counts shared parameters only once, even
    though state_dict lists their keys under each of the five skip positions.
    """
    if not isinstance(config, StructureConfig):
        raise TypeError("config must be a StructureConfig")
    if type(without_skip_attention) is not bool:
        raise ValueError("without_skip_attention must be a bool")

    model = EaBNet(**model_kwargs)
    cred = model.cred

    if config.ca_groups != 1:
        for module in cred.modules():
            if isinstance(module, ChannelAttention):
                _group_channel_attention(module, config.ca_groups)

    if without_skip_attention:
        cred.skip_attention = nn.ModuleList(nn.Identity() for _ in range(5))
    elif config.share_skip_attention:
        shared = cred.skip_attention[0]
        cred.skip_attention = nn.ModuleList([shared] * 5)

    if config.dfsmn_layout == "flattened":
        old = cred.dfsmn
        cred.dfsmn = FlattenedDFSMN(
            channels=old.in_conv.in_channels,
            frequencies=7,  # 257 -> 127 -> 63 -> 31 -> 15 -> 7, fixed by EaBNet.
            hidden_units=old.in_conv.out_channels,
            memory_size=old.memory_size,
            norm_type=model.norm_type,
            is_causal=old.is_causal,
            right_memory_size=old.right_memory_size,
        ).to(device=old.in_conv.weight.device, dtype=old.in_conv.weight.dtype)

    if config.skip_fusion == "add":
        _use_additive_decoder_inputs(cred)
        model.cred = _AdditiveCRED(cred)

    return model


__all__ = ["StructureConfig", "all_configs", "build_candidate", "FlattenedDFSMN"]

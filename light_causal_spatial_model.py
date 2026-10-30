"""End-to-end causal spatial enhancement with bounded streaming state.

Frequency modules use the MIT-licensed SpatialNet source recorded in
third_party/spatialnet/SOURCE.json. Temporal modules are causal adaptations;
this is not the author's original model or a paper reproduction.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from third_party.spatialnet.SpatialNet import SpatialNetLayer


CAUSAL_CANDIDATE = "causal_spatial_v1"


def left_convolution(module, value, history=None):
    width = module.dilation[0] * (module.kernel_size[0] - 1)
    if history is None:
        history = value.new_zeros(value.shape[0], value.shape[1], width)
    if history.shape != (value.shape[0], value.shape[1], width):
        raise ValueError("Temporal convolution cache shape differs from the stream")
    joined = torch.cat((history, value), dim=-1)
    result = module(joined)
    return result, joined[..., -width:].detach().clone() if width else None


class CausalSpatialLayer(SpatialNetLayer):
    def __init__(self, hidden, ffn, frequencies, heads, context, full=None):
        super().__init__(dim_hidden=hidden, dim_ffn=ffn, dim_squeeze=8,
            num_freqs=frequencies, num_heads=heads, kernel_size=(5, 3),
            conv_groups=(8, 8), norms=("LN",) * 6, full=full)
        self.context = context
        for module in self.tconvffn:
            if isinstance(module, nn.Conv1d):
                module.padding = (0,)

    def attention(self, value, state):
        batch, frequencies, frames, hidden = value.shape
        heads = self.mhsa.num_heads
        normalized = self.norm_mhsa(value).reshape(batch * frequencies, frames, hidden)
        q, k, v = F.linear(normalized, self.mhsa.in_proj_weight, self.mhsa.in_proj_bias).chunk(3, dim=-1)
        q, k, v = [x.reshape(batch * frequencies, frames, heads, hidden // heads)
                     .transpose(1, 2) for x in (q, k, v)]
        if state is not None and state.get("k") is not None:
            k = torch.cat((state["k"], k), dim=-2)
            v = torch.cat((state["v"], v), dim=-2)
        previous = k.shape[-2] - frames
        queries = previous + torch.arange(frames, device=value.device)
        keys = torch.arange(k.shape[-2], device=value.device)
        allowed = (keys[None] <= queries[:, None]) & (keys[None] > queries[:, None] - self.context)
        result = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.0)
        result = result.transpose(1, 2).reshape(batch * frequencies, frames, hidden)
        result = F.linear(result, self.mhsa.out_proj.weight, self.mhsa.out_proj.bias)
        cached = {"k": k[..., -(self.context - 1):, :].detach().clone(),
                  "v": v[..., -(self.context - 1):, :].detach().clone()} if state is not None else None
        return result.reshape(batch, frequencies, frames, hidden), cached

    def process(self, value, state=None):
        value = value + self._fconv(self.fconv1, value)
        value = value + self._full(value)
        value = value + self._fconv(self.fconv2, value)
        attended, cached = self.attention(value, state)
        value = value + self.dropout_mhsa(attended)
        batch, frequencies, frames, hidden = value.shape
        temporal = value.transpose(-1, -2).reshape(batch * frequencies, hidden, frames)
        histories = {} if state is None else state.get("convolutions", {})
        updated = {}
        for index, module in enumerate(self.tconvffn):
            if isinstance(module, nn.Conv1d):
                temporal, memory = left_convolution(module, temporal, histories.get(index))
                if state is not None and memory is not None:
                    updated[index] = memory
            else:
                temporal = module(temporal)
        temporal = temporal.reshape(batch, frequencies, hidden, frames).transpose(-1, -2)
        if cached is not None:
            cached["convolutions"] = updated
        return value + self.dropout_tconvffn(temporal), cached

    def forward(self, value):
        return self.process(value)[0]


class CausalSpatialEnhancer(nn.Module):
    """Current/past frames only; no frozen enhancement model or utterance pooling."""

    is_causal = True
    refiner_causal = True

    def __init__(self, M=8, *, hidden=96, ffn=192, layers=8, heads=4, context=128, **kwargs):
        super().__init__()
        if M != 8 or kwargs.get("is_causal", True) is False:
            raise ValueError("Causal spatial candidate requires eight microphones and causal mode")
        if context < 2 or layers < 1 or hidden % 8 or ffn % 8 or hidden % heads:
            raise ValueError("Invalid causal spatial configuration")
        self.M, self.context = M, context
        self.encoder = nn.Conv1d(2 * M, hidden, kernel_size=5, padding=0)
        self.layers = nn.ModuleList()
        full = None
        for _ in range(layers):
            layer = CausalSpatialLayer(hidden, ffn, 257, heads, context, full=full)
            self.layers.append(layer)
            full = layer.full
        self.decoder = nn.Linear(hidden, 2)
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)
        self.gradient_checkpointing = True
        self.use_bfloat16_training = True

    @staticmethod
    def decompress(value):
        magnitude = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
        return value * magnitude.clamp_min(1e-4)

    @staticmethod
    def compress(value):
        magnitude = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
        return value * magnitude.clamp_min(1e-8).pow(-0.5)

    def process(self, inpt, state=None):
        if inpt.ndim != 5 or inpt.shape[-3:] != (257, self.M, 2) or min(inpt.shape[:2]) < 1:
            raise ValueError("expected nonempty compressed STFT (B,T,257,8,2)")
        with torch.autocast(inpt.device.type, enabled=False):
            mixture = self.decompress(inpt.float())
            reference = mixture[..., 0, :]
            # The scale uses this frame's reference-mixture frequencies only.
            scale = torch.linalg.vector_norm(reference, dim=-1).mean(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(1e-4)
            features = (mixture.flatten(-2) / scale).transpose(1, 2)
        batch, frequencies, frames, channels = features.shape
        enabled = self.training and self.use_bfloat16_training and inpt.device.type == "cuda"
        with torch.autocast(inpt.device.type, dtype=torch.bfloat16, enabled=enabled):
            encoded = features.reshape(batch * frequencies, frames, channels).transpose(1, 2)
            encoded, history = left_convolution(self.encoder, encoded, None if state is None else state.get("encoder"))
            value = encoded.transpose(1, 2).reshape(batch, frequencies, frames, -1)
            states = []
            for index, layer in enumerate(self.layers):
                if state is not None:
                    value, cached = layer.process(value, state.get("layers", [{}] * len(self.layers))[index])
                    states.append(cached)
                elif self.training and self.gradient_checkpointing:
                    value = checkpoint(layer, value, use_reentrant=False)
                else:
                    value = layer(value)
            residual = self.decoder(value).transpose(1, 2).float()
        with torch.autocast(inpt.device.type, enabled=False):
            correction = self.compress(reference + scale * residual) - self.compress(reference)
            output = (inpt[..., 0, :].float() + correction).permute(0, 3, 1, 2)
        return output, {"encoder": history, "layers": states} if state is not None else None

    def forward(self, inpt, frame_lengths=None):
        if frame_lengths is not None:
            lengths = torch.as_tensor(frame_lengths)
            if lengths.numel() != inpt.shape[0] or torch.any(lengths < 1) or torch.any(lengths > inpt.shape[1]):
                raise ValueError("one valid frame length is required per sample")
        return self.process(inpt)[0]

    def forward_stream(self, inpt, state=None):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("Streaming inference requires eval mode and disabled gradients")
        return self.process(inpt, {} if state is None else state)

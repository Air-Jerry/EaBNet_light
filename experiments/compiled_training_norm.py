"""Compile the existing FP32-statistics LayerNorm only during CUDA training."""

from contextlib import contextmanager

import torch
import torch.nn.functional as F

from light_causal_fast_model import ActivationLayerNorm


def fp32_layer_norm(value, weight, bias, eps):
    return F.layer_norm(value.float(), (value.shape[-1],), weight, bias, eps).to(value.dtype)


@contextmanager
def compiled_training_norms():
    original = ActivationLayerNorm.forward
    compiled = None
    previous_limit = torch._dynamo.config.cache_size_limit
    torch._dynamo.config.cache_size_limit = max(previous_limit, 64)

    def forward(self, value):
        nonlocal compiled
        if not self.training or value.device.type != 'cuda' or len(self.normalized_shape) != 1:
            return original(self, value)
        if self.seq_last:
            value = value.transpose(-1, 1)
        if compiled is None:
            compiled = torch.compile(fp32_layer_norm, fullgraph=True, dynamic=True)
        with torch.autocast('cuda', enabled=False):
            result = compiled(value, self.weight, self.bias, self.eps)
        return result.transpose(-1, 1) if self.seq_last else result

    ActivationLayerNorm.forward = forward
    try:
        yield
    finally:
        ActivationLayerNorm.forward = original
        torch._dynamo.config.cache_size_limit = previous_limit

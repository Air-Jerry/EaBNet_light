"""Compile FP32 retention during CUDA training; preserve original inference."""

from contextlib import contextmanager

import torch

from experiments.causal_retention_mapping import CausalRetention


@contextmanager
def compiled_training_retention():
    original = CausalRetention.process
    compiled = None

    def process(self, features, state=None, cache=False):
        nonlocal compiled
        if not self.training or features.device.type != 'cuda' or state is not None or cache:
            return original(self, features, state, cache)
        if compiled is None:
            # Keep small recurrent matrix products in the original FP32 arithmetic.
            compiled = torch.compile(original, fullgraph=True, dynamic=False,
                options={'emulate_precision_casts': True, 'force_same_precision': True})
        with torch.autocast('cuda', enabled=False):
            result, _ = compiled(self, features.float(), None, False)
        return result.to(features.dtype), None

    CausalRetention.process = process
    try:
        yield
    finally:
        CausalRetention.process = original

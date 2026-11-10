"""Recompute spatial blocks during training without changing parameters or stream inference."""
from contextlib import contextmanager

import torch
from torch.utils.checkpoint import checkpoint


@contextmanager
def checkpoint_spatial_training(root):
    saved = []

    def wrap(layer, original):
        def forward(value):
            if layer.training and torch.is_grad_enabled():
                return checkpoint(original, value, use_reentrant=False)
            return original(value)
        return forward

    try:
        for layer in root.layers:
            saved.append((layer, layer.__dict__.get('forward')))
            layer.forward = wrap(layer, layer.forward)
        yield
    finally:
        for layer, original in reversed(saved):
            if original is None:
                del layer.forward
            else:
                layer.forward = original

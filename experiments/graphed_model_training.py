"""Capture model forward/backward only; keep the original loss, finite checks, Adam and DDP outside."""
from contextlib import contextmanager

import torch
from torch import nn


@contextmanager
def graphed_model_training(model, sample):
    if sample.device.type != 'cuda' or not model.training:
        raise ValueError('Capture a CUDA model in its original training mode before DDP wrapping')
    if not callable(getattr(model, 'process', None)):
        raise ValueError('Use the existing stateless training process')
    original = model.process
    had_override = 'process' in model.__dict__

    class Surface(nn.Module):
        def __init__(self):
            super().__init__()
            self.source = model

        def forward(self, value):
            return original(value)[0]

    # Internal mixed precision inherits disabled cast caching, as required by PyTorch.
    with torch.autocast('cuda', enabled=False, cache_enabled=False):
        surface = torch.cuda.make_graphed_callables(Surface(), (sample,), num_warmup_iters=3)

        def process(value, state=None):
            if (model.training and torch.is_grad_enabled() and state is None
                    and value.shape == sample.shape and value.device == sample.device
                    and value.dtype == sample.dtype and value.requires_grad == sample.requires_grad):
                return surface(value), None
            return original(value, state)

        model.process = process
        try:
            yield
        finally:
            if had_override:
                model.process = original
            else:
                del model.process

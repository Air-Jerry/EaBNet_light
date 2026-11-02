"""Compile only the folded model's CUDA training forward; preserve saved weights."""

from contextlib import contextmanager

import torch


@contextmanager
def compiled_folded_forward(model, cache):
    original=model.forward
    if id(model) not in cache:
        compiled=torch.compile(original,fullgraph=True,dynamic=False,
            options={'emulate_precision_casts':True,'force_same_precision':True})

        def forward(value, frame_lengths=None):
            if frame_lengths is not None:
                lengths=torch.as_tensor(frame_lengths)
                if lengths.numel()!=value.shape[0] or torch.any(lengths<1) or torch.any(lengths>value.shape[1]):
                    raise ValueError('One valid frame length is required per sample')
            return compiled(value)

        cache[id(model)]=forward
    model.forward=cache[id(model)]
    previous=torch._dynamo.config.optimize_ddp
    torch._dynamo.config.optimize_ddp=False
    try:
        yield
    finally:
        torch._dynamo.config.optimize_ddp=previous
        model.forward=original

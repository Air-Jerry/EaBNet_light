"""Training-only power-of-two loss scaling with unscale before the existing update rules."""
from contextlib import contextmanager
import math

import torch


@contextmanager
def backward_loss_scaling(trainer, factor):
    if not math.isfinite(factor) or not 0 < factor <= 1 or math.frexp(factor)[0] != .5:
        raise ValueError('Use a finite power-of-two backward factor in (0,1]')
    if factor == 1:
        yield
        return
    original = trainer.GradScaler

    def factory(device='cuda', enabled=False):
        if enabled:
            raise ValueError('Backward scaling requires the existing outer AMP to stay disabled')
        return original(device=device, enabled=device != 'cuda' or torch.cuda.is_available(),
                        init_scale=factor, growth_interval=2**31-1)

    trainer.GradScaler = factory
    try:
        yield
    finally:
        trainer.GradScaler = original

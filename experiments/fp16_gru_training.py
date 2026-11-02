"""Training-only CuDNN GRU autocast with FP32 weights and FP32 caller interfaces."""

from contextlib import contextmanager

import torch


@contextmanager
def fp16_gru_training():
    original=torch.nn.GRU.forward

    def forward(self,value,hidden=None):
        if (self.training and isinstance(value,torch.Tensor)
                and value.device.type=='cuda' and value.dtype==torch.float32):
            with torch.autocast('cuda',dtype=torch.float16):
                result,state=original(self,value,hidden)
            return result.float(),state.float()
        return original(self,value,hidden)

    torch.nn.GRU.forward=forward
    try:yield
    finally:torch.nn.GRU.forward=original

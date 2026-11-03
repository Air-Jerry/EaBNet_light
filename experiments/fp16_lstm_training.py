"""Training-only CuDNN LSTM autocast, retaining FP32 parameters and caller state."""
from contextlib import contextmanager

import torch


@contextmanager
def fp16_lstm_training():
    original=torch.nn.LSTM.forward

    def forward(self,value,hidden=None):
        if (self.training and isinstance(value,torch.Tensor)
                and value.device.type=='cuda' and value.dtype==torch.float32):
            with torch.autocast('cuda',dtype=torch.float16):
                result,state=original(self,value,hidden)
            return result.float(),tuple(item.float() for item in state)
        return original(self,value,hidden)

    torch.nn.LSTM.forward=forward
    try:yield
    finally:torch.nn.LSTM.forward=original

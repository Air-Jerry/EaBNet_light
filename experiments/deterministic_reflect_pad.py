"""Equivalent one-dimensional reflection padding without the CUDA atomic backward."""
from contextlib import contextmanager

import torch
import torch.nn.functional as functional


@contextmanager
def deterministic_reflection_padding():
    original=functional.pad

    def pad(input,pad,mode='constant',value=None):
        if (mode=='reflect' and len(pad)==2 and input.ndim in (2,3)
                and value is None and 0<=min(pad) and max(pad)<input.shape[-1]):
            left,right=pad
            prefix=input[...,1:left+1].flip(-1)
            suffix=input[...,-right-1:-1].flip(-1) if right else input[...,:0]
            return torch.cat((prefix,input,suffix),dim=-1)
        return original(input,pad,mode,value)

    functional.pad=pad
    try:yield
    finally:functional.pad=original

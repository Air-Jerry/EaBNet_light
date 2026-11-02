"""Experimental128-state subband head; warm output retained from a32-state source.

The causal backbone stays frozen. Real-data quality, full6s CUDA cost and
streaming deadlines have not yet been qualified for this prototype.
"""

import math

import torch
from torch import nn

from experiments.causal_spatial_conditioned_subband import CausalSpatialConditionedSubband
from light_causal_fast_model import ActivationLayerNorm


class IndependentBlockNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.groups=nn.ModuleList([ActivationLayerNorm(32) for _ in range(4)])

    def forward(self,value):
        if value.shape[-1]!=128:raise ValueError('Expected128 subband features')
        return torch.cat([group(value[...,index*32:(index+1)*32]) for index,group in enumerate(self.groups)],dim=-1)


class CausalWideConditionedSubband(CausalSpatialConditionedSubband):
    def __init__(self,base_model):
        super().__init__(base_model,hidden=128)
        self.recurrent_norm=IndependentBlockNorm()

    @torch.no_grad()
    def load_warm_32(self,source):
        shapes={
            'input.weight':(32,40,1,3),'input.bias':(32,),
            'fullband_mix.weight':(32,16,1,1),'fullband_mix.bias':(32,),
            'recurrent.weight_ih_l0':(96,32),'recurrent.weight_hh_l0':(96,32),
            'recurrent.bias_ih_l0':(96,),'recurrent.bias_hh_l0':(96,),
            'recurrent_norm.weight':(32,),'recurrent_norm.bias':(32,),
            'output.weight':(18,32,1,1)}
        if any(key not in source or tuple(source[key].shape)!=shape for key,shape in shapes.items()):
            raise ValueError('Warm source must have the original32-state spatial-conditioned architecture')
        target=self.state_dict()
        old_norm={'recurrent_norm.weight','recurrent_norm.bias'}
        new_norm={f'recurrent_norm.groups.{index}.{name}' for index in range(4) for name in ('weight','bias')}
        if set(source)-old_norm!=set(target)-new_norm:
            raise ValueError('Warm source parameter keys differ from the spatial-conditioned model')
        for key,value in source.items():
            if not torch.isfinite(value).all():raise ValueError(f'Nonfinite warm parameter: {key}')
            if key not in shapes and target[key].shape!=value.shape:
                raise ValueError(f'Warm source shape differs: {key}')
        for key,value in source.items():
            if key not in shapes:target[key]=value.detach().clone()
        for key in ('input.weight','input.bias','fullband_mix.weight','fullband_mix.bias'):
            target[key][:32].copy_(source[key])
        for name in ('weight','bias'):
            target[f'recurrent_norm.groups.0.{name}'].copy_(source[f'recurrent_norm.{name}'])
        for gate in range(3):
            old_rows=slice(gate*32,(gate+1)*32);new_rows=slice(gate*128,gate*128+32)
            for key in ('recurrent.weight_ih_l0','recurrent.weight_hh_l0'):
                target[key][new_rows].zero_()
                target[key][new_rows,:32].copy_(source[key][old_rows])
            for key in ('recurrent.bias_ih_l0','recurrent.bias_hh_l0'):
                target[key][new_rows].copy_(source[key][old_rows])
        target['output.weight'].zero_()
        # Parent scales recurrent output by hidden**-.5; compensate the old channels.
        target['output.weight'][:,:32].copy_(source['output.weight']*math.sqrt(128/32))
        self.load_state_dict(target,strict=True)

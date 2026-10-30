"""Causal spatial model with global-frequency temporal recurrence."""

import torch
from torch import nn

from light_causal_fast_model import CausalUNetSpatial,ActivationLayerNorm


GLOBAL_CANDIDATE='causal_unet_global_v1'


class GlobalTemporalLayer(nn.Module):
    def __init__(self,frequencies,channels,hidden):
        super().__init__()
        features=frequencies*channels
        self.norm=ActivationLayerNorm(features)
        self.project=nn.Linear(features,hidden)
        self.rnn=nn.GRU(hidden,hidden,batch_first=True,bidirectional=False)
        self.output=nn.Linear(hidden,features)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def process(self,value,state=None):
        batch,frequencies,frames,channels=value.shape
        with torch.autocast(value.device.type,enabled=False):
            temporal=value.float().transpose(1,2).reshape(batch,frames,frequencies*channels)
            temporal=torch.nn.functional.silu(self.project(self.norm(temporal)))
            temporal,hidden=self.rnn(temporal,None if state is None else state.get('hidden'))
            residual=self.output(temporal).reshape(batch,frames,frequencies,channels).transpose(1,2)
        updated={'hidden':hidden.detach().clone()} if state is not None else None
        return value+residual.to(value.dtype),updated

    def forward(self,value):
        return self.process(value)[0]


class CausalGlobalSpatial(CausalUNetSpatial):
    def __init__(self,M=8,global_hidden=128,**kwargs):
        super().__init__(M=M,**kwargs)
        # Keep subsequent DataLoader crop seeds independent of the added parameters.
        with torch.random.fork_rng(devices=[]):
            self.layers.append(GlobalTemporalLayer(17,self.encoders[-1].conv.out_channels,global_hidden))

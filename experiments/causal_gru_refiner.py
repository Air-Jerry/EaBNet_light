"""Experimental current/past-frame recurrent refinement, no deployment claim."""

import torch
from torch import nn

from light_causal_fast_model import CausalUNetSpatial,CausalFrequencyConv


class CausalGRURefiner(nn.Module):
    is_causal=True

    def __init__(self,backbone=None,hidden=96,channels=64):
        super().__init__()
        self.backbone=CausalUNetSpatial() if backbone is None else backbone
        if not self.backbone.is_causal or self.backbone.M!=8:
            raise ValueError('The entire backbone must be causal and use eight microphones')
        self.M,self.context=8,self.backbone.context
        self.encoder=CausalFrequencyConv(18,channels,frequency_kernel=5)
        self.rnn1=nn.GRU(channels,hidden,batch_first=True,bidirectional=False)
        self.rnn2=nn.GRU(hidden,hidden,batch_first=True,bidirectional=False)
        self.output=nn.Linear(hidden,2)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def train(self,mode=True):
        super().train(mode)
        if not any(p.requires_grad for p in self.backbone.parameters()):self.backbone.eval()
        return self

    def refine(self,inpt,baseline,state=None):
        batch,frames,frequencies=inpt.shape[:3]
        memories={} if state is None else state
        with torch.autocast(inpt.device.type,enabled=False):
            scale=torch.linalg.vector_norm(inpt.float(),dim=-1)[...,0].mean(-1,keepdim=True).unsqueeze(-1).clamp_min(1e-4)
            reference=baseline.float().permute(0,2,3,1)
            features=torch.cat((inpt.float().flatten(-2),reference),dim=-1)/scale
            value,history=self.encoder.process(features.permute(0,3,1,2),memories.get('encoder'),state is not None)
            value=value.permute(0,3,2,1).reshape(batch*frequencies,frames,-1)
            value,h1=self.rnn1(value,memories.get('rnn1'))
            value,h2=self.rnn2(value,memories.get('rnn2'))
            residual=self.output(value).reshape(batch,frequencies,frames,2).transpose(1,2)
            output=(reference+scale*residual).permute(0,3,1,2)
        updated=dict(encoder=history,rnn1=h1.detach().clone(),rnn2=h2.detach().clone()) if state is not None else None
        return output,updated

    def forward(self,inpt,frame_lengths=None):
        baseline=self.backbone(inpt,frame_lengths=frame_lengths)
        return self.refine(inpt,baseline)[0]

    def forward_stream(self,inpt,state=None):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError('Streaming requires eval mode and disabled gradients')
        memories={} if state is None else state
        baseline,backbone=self.backbone.forward_stream(inpt,memories.get('backbone'))
        output,updated=self.refine(inpt,baseline,memories)
        updated['backbone']=backbone
        return output,updated

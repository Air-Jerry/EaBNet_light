"""Unregistered independent monaural branch with a zero-initialized causal fusion.

The complete existing source is fixed in inference mode. The new branch sees
only microphone0 and can receive its own training target via an external hook.
Real-data quality and the complete model's CUDA cost still need measurement.
"""
import math

import torch
from torch import nn

from experiments.causal_fullchannel_dualpath_crn import FullChannelDualPath
from experiments.causal_monaural_spectral_branch import frequency_block


class IndependentMonauralBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders=nn.ModuleList(frequency_block(a,b,2)
            for a,b in zip((4,16,24,32),(16,24,32,64)))
        self.blocks=nn.ModuleList(FullChannelDualPath(64,17,32,64,1/math.sqrt(4)) for _ in range(2))
        self.upsamplers=nn.ModuleList(nn.ConvTranspose2d(a,b,(1,5),stride=(1,2),padding=(0,2))
            for a,b in zip((64,32,24,16),(32,24,16,16)))
        self.decoders=nn.ModuleList(frequency_block(a,b)
            for a,b in zip((64,48,32,20),(32,24,16,16)))
        self.output=nn.Conv2d(16,2,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def process(self,reference,state=None):
        if (reference.ndim!=4 or reference.shape[-2:]!=(257,2)
                or min(reference.shape[:2])<1 or reference.dtype!=torch.float32):
            raise ValueError('Expected FP32 reference STFT(B,T,257,2)')
        cache=state is not None
        if cache and (not isinstance(state,dict) or (state and
                (set(state)!={'blocks'} or not isinstance(state['blocks'],tuple) or len(state['blocks'])!=2))):
            raise ValueError('Use a complete independently owned monaural state')
        memories={} if state is None else state
        if memories and any(item is None for item in memories['blocks']):
            raise ValueError('Use a complete independently owned monaural state')
        with torch.autocast(reference.device.type,enabled=False):
            magnitude=torch.linalg.vector_norm(reference,dim=-1)
            scale=magnitude.mean(-1,keepdim=True).clamp_min(1e-4)
            absolute=scale.log().unsqueeze(-1).expand(*magnitude.shape,1)
            features=torch.cat((reference/scale.unsqueeze(-1),
                torch.log1p(magnitude/scale).unsqueeze(-1),absolute),-1).permute(0,3,1,2)
        enabled=self.training and reference.device.type=='cuda'
        with torch.autocast(reference.device.type,dtype=torch.bfloat16,enabled=enabled):
            value=features;skips=[features]
            for encoder in self.encoders:
                value=encoder(value);skips.append(value)
        updated=[]
        for index,block in enumerate(self.blocks):
            previous=memories.get('blocks',(None,None))[index]
            value,saved=block.process(value,previous,cache)
            updated.append(saved)
        with torch.autocast(reference.device.type,dtype=torch.bfloat16,enabled=enabled):
            for up,decoder,skip in zip(self.upsamplers,self.decoders,reversed(skips[:-1])):
                value=up(value)
                if value.shape[-1]!=skip.shape[-1]:raise RuntimeError('Frequency decoder lost alignment')
                value=decoder(torch.cat((value,skip),1))
            correction=self.output(value).float()*scale[:,None]
        result=reference.permute(0,3,1,2)+correction
        return result,dict(blocks=tuple(updated)) if cache else None

    def forward(self,reference):
        return self.process(reference)[0]


class CausalAuxiliaryMonauralFusion(nn.Module):
    def __init__(self,source):
        super().__init__()
        if (getattr(source,'is_causal',None) is not True or getattr(source,'M',None)!=8
                or not callable(getattr(source,'process',None))):
            raise ValueError('Use a complete eight-microphone causal source')
        self.base=source.requires_grad_(False).eval()
        self.base.zero_grad(set_to_none=True)
        self.M,self.is_causal,self.context=8,True,int(getattr(source,'context',128))
        self.monaural=IndependentMonauralBranch()
        self.gate_features=frequency_block(9,16)
        self.gate=nn.Conv2d(16,1,1)
        nn.init.zeros_(self.gate.weight);nn.init.zeros_(self.gate.bias)

    def train(self,mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def process(self,inpt,state=None):
        if (inpt.ndim!=5 or inpt.shape[-3:]!=(257,8,2) or min(inpt.shape[:2])<1
                or inpt.dtype!=torch.float32):
            raise ValueError('Expected FP32 compressed STFT(B,T,257,8,2)')
        cache=state is not None
        if cache and (not isinstance(state,dict) or (state and
                (set(state)!={'base','monaural'} or any(not isinstance(v,dict) or not v for v in state.values())))):
            raise ValueError('Use a complete independently owned fusion state')
        memories={} if state is None else state
        source,source_state=self.base.process(inpt,memories.get('base',{}) if cache else None)
        reference=inpt[...,0,:]
        if cache:
            mono,mono_state=self.monaural.process(reference,memories.get('monaural',{}))
        else:
            mono=self.monaural(reference);mono_state=None
        with torch.autocast(inpt.device.type,enabled=False):
            if source.shape!=mono.shape or source.dtype!=torch.float32:
                raise ValueError('Source returned an incompatible FP32 spectrum')
            views=torch.stack((reference,source.permute(0,2,3,1),mono.permute(0,2,3,1)),-2)
            magnitudes=torch.linalg.vector_norm(views,dim=-1)
            scale=magnitudes[...,0].mean(-1,keepdim=True).clamp_min(1e-4)
            features=torch.cat(((views/scale[...,None,None]).flatten(-2),
                torch.log1p(magnitudes/scale[...,None])),-1).permute(0,3,1,2)
        enabled=self.training and inpt.device.type=='cuda'
        with torch.autocast(inpt.device.type,dtype=torch.bfloat16,enabled=enabled):
            gate=self.gate(self.gate_features(features)).float().tanh()
        result=source+gate*(mono-source)
        return result,dict(base=source_state,monaural=mono_state) if cache else None

    def forward(self,inpt,frame_lengths=None):
        if frame_lengths is not None:
            lengths=torch.as_tensor(frame_lengths)
            if lengths.numel()!=inpt.shape[0] or torch.any(lengths<1) or torch.any(lengths>inpt.shape[1]):
                raise ValueError('One valid frame length is required per sample')
        return self.process(inpt)[0]

    def forward_stream(self,inpt,state=None):
        if self.training or torch.is_grad_enabled():raise RuntimeError('Streaming requires eval mode and disabled gradients')
        return self.process(inpt,{} if state is None else state)

"""Unregistered current-frame spectral U-Net and causal full-band GRU branch.

Existing causal source training/freeze policy is retained. Real-data quality,
full6s CUDA budget and complete streaming deadlines still require measurement.
"""
import torch
from torch import nn

from light_causal_fast_model import ActivationLayerNorm,FrameChannelNorm


def frequency_block(incoming,outgoing,stride=1):
    return nn.Sequential(nn.Conv2d(incoming,outgoing,(1,5),stride=(1,stride),padding=(0,2)),
        FrameChannelNorm(outgoing),nn.SiLU())


class CausalMonauralSpectralBranch(nn.Module):
    def __init__(self,source):
        super().__init__()
        if getattr(source,'is_causal',None) is not True or not callable(getattr(source,'process',None)):
            raise ValueError('The existing source must itself implement a causal process')
        self.base=source;self.M=8;self.is_causal=True
        self.context=int(getattr(source,'context',128))
        self.encoders=nn.ModuleList([frequency_block(a,b,2) for a,b in zip((4,16,24,32),(16,24,32,48))])
        self.recurrent_norm=ActivationLayerNorm(17*48)
        self.recurrent=nn.GRU(17*48,128,batch_first=True,bidirectional=False)
        self.projection=nn.Linear(128,17*48)
        self.upsamplers=nn.ModuleList([nn.ConvTranspose2d(a,b,(1,5),stride=(1,2),padding=(0,2))
            for a,b in zip((48,32,24,16),(32,24,16,16))])
        self.decoders=nn.ModuleList([frequency_block(a,b) for a,b in zip((64,48,32,20),(32,24,16,16))])
        self.output=nn.Conv2d(16,2,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def process(self,inpt,state=None):
        if inpt.ndim!=5 or inpt.shape[-3:]!=(257,8,2) or min(inpt.shape[:2])<1:
            raise ValueError('Expected compressed STFT(B,T,257,8,2)')
        cache=state is not None
        if cache and (not isinstance(state,dict) or (state and set(state)!={'base','monaural_hidden'})):
            raise ValueError('Use a complete independently owned monaural stream state')
        memories={} if state is None else state
        batch,frames=inpt.shape[:2]
        hidden=memories.get('monaural_hidden')
        if memories and (not isinstance(memories['base'],dict) or not isinstance(hidden,torch.Tensor)):
            raise ValueError('Use a complete independently owned monaural stream state')
        if hidden is not None and (hidden.shape!=(1,batch,128) or hidden.dtype!=torch.float32 or hidden.device!=inpt.device):
            raise ValueError('Monaural hidden state differs from this stream')
        estimate,base_state=self.base.process(inpt,memories.get('base',{}) if cache else None)
        if estimate.shape!=(batch,2,frames,257):raise ValueError('Causal source returned an incompatible spectrum')
        with torch.autocast(inpt.device.type,enabled=False):
            reference=inpt[...,0,:].float()
            scale=torch.linalg.vector_norm(reference,dim=-1).mean(-1,keepdim=True).clamp_min(1e-4)
            features=(torch.cat((reference,estimate.float().permute(0,2,3,1)),-1)/scale[...,None]).permute(0,3,1,2)
        enabled=self.training and inpt.device.type=='cuda'
        with torch.autocast(inpt.device.type,dtype=torch.bfloat16,enabled=enabled):
            value=features;skips=[features]
            for encoder in self.encoders:
                value=encoder(value);skips.append(value)
        with torch.autocast(inpt.device.type,enabled=False):
            sequence=value.permute(0,2,1,3).reshape(batch,frames,17*48).float()
            sequence,hidden=self.recurrent(self.recurrent_norm(sequence),hidden)
        with torch.autocast(inpt.device.type,dtype=torch.bfloat16,enabled=enabled):
            value=value+self.projection(sequence).reshape(batch,frames,48,17).permute(0,2,1,3)
            for up,decoder,skip in zip(self.upsamplers,self.decoders,reversed(skips[:-1])):
                value=up(value)
                if value.shape[-1]!=skip.shape[-1]:raise RuntimeError('Frequency decoder lost alignment')
                value=decoder(torch.cat((value,skip),1))
            correction=self.output(value).float()*scale[:,None]
        output=estimate.float()+correction
        updated=dict(base=base_state,monaural_hidden=hidden.detach().clone()) if cache else None
        return output,updated

    def forward(self,inpt,frame_lengths=None):
        if frame_lengths is not None:
            lengths=torch.as_tensor(frame_lengths)
            if lengths.numel()!=inpt.shape[0] or torch.any(lengths<1) or torch.any(lengths>inpt.shape[1]):
                raise ValueError('One valid frame length is required per sample')
        return self.process(inpt)[0]

    def forward_stream(self,inpt,state=None):
        if self.training or torch.is_grad_enabled():raise RuntimeError('Streaming requires eval mode and disabled gradients')
        return self.process(inpt,{} if state is None else state)

"""Unregistered causal physical refinement, preserving a trained source at zero readout.

The source remains part of inference and keeps its existing training policy.
Quality and combined-model CUDA cost require measurement, not additive guesses.
"""
import torch

from experiments.causal_global_spectral_beamformer import CausalGlobalSpectralBeamformer
from experiments.causal_monaural_spectral_branch import frequency_block


class CausalWarmConditionedGlobalBeamformer(CausalGlobalSpectralBeamformer):
    def __init__(self,source):
        if (getattr(source,'is_causal',None) is not True or getattr(source,'M',None)!=8
                or not callable(getattr(source,'process',None))):
            raise ValueError('Use an eight-microphone causal source with an owned-state process')
        super().__init__()
        self.base=source
        self.context=max(self.context,int(getattr(source,'context',128)))
        self.encoders[0]=frequency_block(40,32,2)
        self.decoders[-1]=frequency_block(72,32)

    def process(self,inpt,state=None):
        if (inpt.ndim!=5 or inpt.shape[-3:]!=(257,8,2) or min(inpt.shape[:2])<1
                or inpt.dtype!=torch.float32):
            raise ValueError('Expected FP32 compressed STFT(B,T,257,8,2)')
        cache=state is not None
        memories={} if state is None else state
        if not isinstance(memories,dict) or (memories and set(memories)!={'base','hidden'}):
            raise ValueError('Use a complete independently owned conditioned-global state')
        batch,frames=inpt.shape[:2]
        hidden=memories.get('hidden')
        if memories and (not isinstance(memories['base'],dict) or not torch.is_tensor(hidden)):
            raise ValueError('Use a complete independently owned conditioned-global state')
        if hidden is not None and (hidden.shape!=(3,batch,384) or hidden.dtype!=torch.float32 or hidden.device!=inpt.device):
            raise ValueError('Global hidden state differs from this stream')
        estimate,base_state=self.base.process(inpt,memories.get('base',{}) if cache else None)
        if estimate.shape!=(batch,2,frames,257) or estimate.dtype!=torch.float32 or estimate.device!=inpt.device:
            raise ValueError('Causal source returned an incompatible FP32 spectrum')
        with torch.autocast(inpt.device.type,enabled=False):
            magnitude=torch.linalg.vector_norm(inpt,dim=-1)
            scale=magnitude[...,:1].mean(-2,keepdim=True).clamp_min(1e-4)
            normalized=(inpt/scale.unsqueeze(-1)).flatten(-2)
            unit=inpt/magnitude.clamp_min(1e-4).unsqueeze(-1)
            real=unit[...,1:,0]*unit[...,:1,0]+unit[...,1:,1]*unit[...,:1,1]
            imaginary=unit[...,1:,1]*unit[...,:1,0]-unit[...,1:,0]*unit[...,:1,1]
            ratio=2*torch.log((magnitude[...,1:]+1e-4)/(magnitude[...,:1]+1e-4))
            ratio=5*torch.tanh(ratio/5)
            envelope=torch.log1p(magnitude[...,:1]/scale)
            conditioned=estimate.permute(0,2,3,1)
            features=torch.cat((normalized,real,imaginary,ratio,envelope,conditioned/scale),-1).permute(0,3,1,2)
            levels=magnitude.mean(-2).clamp_min(1e-4).log()
        enabled=self.training and inpt.device.type=='cuda'
        with torch.autocast(inpt.device.type,dtype=torch.bfloat16,enabled=enabled):
            value,skips=features,[features]
            for encoder in self.encoders:
                value=encoder(value);skips.append(value)
        with torch.autocast(inpt.device.type,enabled=False):
            sequence=value.permute(0,2,1,3).reshape(batch,frames,17*96).float()
            sequence=torch.cat((self.recurrent_norm(sequence),levels),-1)
            sequence,hidden=self.recurrent(sequence,hidden)
        with torch.autocast(inpt.device.type,dtype=torch.bfloat16,enabled=enabled):
            value=value+self.projection(sequence).reshape(batch,frames,96,17).permute(0,2,1,3)
            for up,decoder,skip in zip(self.upsamplers,self.decoders,reversed(skips[:-1])):
                value=up(value)
                if value.shape[-1]!=skip.shape[-1]:raise RuntimeError('Frequency decoder lost alignment')
                value=decoder(torch.cat((value,skip),1))
            coefficients=self.output(value).float().permute(0,2,3,1).reshape(batch,frames,257,8,2)
        with torch.autocast(inpt.device.type,enabled=False):
            physical=inpt*magnitude.unsqueeze(-1)
            real=(coefficients[...,0]*physical[...,0]+coefficients[...,1]*physical[...,1]).sum(-1)
            imaginary=(coefficients[...,0]*physical[...,1]-coefficients[...,1]*physical[...,0]).sum(-1)
            baseline_physical=(conditioned*torch.linalg.vector_norm(conditioned,dim=-1).unsqueeze(-1)).contiguous()
            enhanced=baseline_physical+torch.stack((real,imaginary),-1)
            compressed=enhanced/torch.linalg.vector_norm(enhanced,dim=-1).clamp_min(1e-8).sqrt().unsqueeze(-1)
            baseline=baseline_physical/torch.linalg.vector_norm(baseline_physical,dim=-1).clamp_min(1e-8).sqrt().unsqueeze(-1)
            output=(conditioned+(compressed-baseline)).permute(0,3,1,2).contiguous()
        updated=dict(base=base_state,hidden=hidden.detach().clone()) if cache else None
        return output,updated

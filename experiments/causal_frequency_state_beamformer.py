"""Unregistered frequency/state-memory network with physical eight-microphone output.

Independent adaptation, not author code. No measured quality or training budget
is claimed. All frequency directions are confined to the current time frame.
"""
import torch
import torch.nn.functional as F

from experiments.causal_frequency_state_memory import CausalFrequencyStateMemory,BufferedTimeConv


class CausalFrequencyStateBeamformer(CausalFrequencyStateMemory):
    def __init__(self,width=32,blocks=6):
        super().__init__(width,blocks)
        self.encoder=BufferedTimeConv(39,width)
        self.decoder=BufferedTimeConv(width,16)
        torch.nn.init.zeros_(self.decoder.conv.weight)
        torch.nn.init.zeros_(self.decoder.conv.bias)

    def process(self,inpt,state=None):
        if (inpt.ndim!=5 or inpt.shape[-3:]!=(257,8,2) or min(inpt.shape[:2])<1
                or inpt.dtype!=torch.float32):
            raise ValueError('Expected FP32 compressed STFT(B,T,257,8,2)')
        cache=state is not None
        memories={} if state is None else state
        if not isinstance(memories,dict) or (memories and set(memories)!={'encoder','blocks','decoder'}):
            raise ValueError('Use a complete independently owned spectral-memory state')
        if memories and (not isinstance(memories['blocks'],tuple) or len(memories['blocks'])!=len(self.blocks)
                or memories['encoder'] is None or memories['decoder'] is None
                or any(not isinstance(item,dict) or set(item)!={'local','hidden','cell'}
                       or item['local'] is None for item in memories['blocks'])):
            raise ValueError('Use a complete independently owned spectral-memory state')
        magnitude=torch.linalg.vector_norm(inpt,dim=-1)
        scale=magnitude[...,:1].mean(-2,keepdim=True).clamp_min(1e-4)
        normalized=(inpt/scale.unsqueeze(-1)).flatten(-2)
        unit=inpt/magnitude.clamp_min(1e-4).unsqueeze(-1)
        real=unit[...,1:,0]*unit[...,:1,0]+unit[...,1:,1]*unit[...,:1,1]
        imaginary=unit[...,1:,1]*unit[...,:1,0]-unit[...,1:,0]*unit[...,:1,1]
        ratio=2*torch.log((magnitude[...,1:]+1e-4)/(magnitude[...,:1]+1e-4))
        ratio=5*torch.tanh(ratio/5)
        envelope=torch.log1p(magnitude[...,:1]/scale)
        features=torch.cat((normalized,real,imaginary,ratio,envelope,scale.log().expand(-1,-1,257,-1)),-1)
        value,encoder=self.encoder.process(features,memories.get('encoder'),cache)
        value=F.silu(self.encoder_norm(value))
        initial,states=None,[]
        for index,block in enumerate(self.blocks):
            previous=memories['blocks'][index] if memories else None
            value,initial,updated=block.process(value,initial,previous,cache)
            states.append(updated)
        coefficients,decoder=self.decoder.process(value,memories.get('decoder'),cache)
        coefficients=coefficients.reshape(*inpt.shape[:3],8,2)
        reference_weight=coefficients.new_zeros(1,1,1,8,2)
        reference_weight[...,0,0]=1
        coefficients=coefficients+reference_weight
        physical=inpt*magnitude.unsqueeze(-1)
        real=(coefficients[...,0]*physical[...,0]+coefficients[...,1]*physical[...,1]).sum(-1)
        imaginary=(coefficients[...,0]*physical[...,1]-coefficients[...,1]*physical[...,0]).sum(-1)
        enhanced=torch.stack((real,imaginary),-1)
        compressed=enhanced/torch.linalg.vector_norm(enhanced,dim=-1).clamp_min(1e-8).sqrt().unsqueeze(-1)
        reference=physical[...,0,:].contiguous()
        baseline=reference/torch.linalg.vector_norm(reference,dim=-1).clamp_min(1e-8).sqrt().unsqueeze(-1)
        output=(inpt[...,0,:]+(compressed-baseline)).permute(0,3,1,2).contiguous()
        updated=dict(encoder=encoder,blocks=tuple(states),decoder=decoder) if cache else None
        return output,updated

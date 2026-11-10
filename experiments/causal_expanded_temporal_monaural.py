"""Unregistered wider causal monaural temporal bottleneck; real quality and GPU cost remain unverified."""
import torch
from torch import nn

from experiments.causal_recurrent_adapt_monaural import CausalRecurrentAdaptMonaural


def expand_temporal_tensor(name,value,target,*,zero=False):
    result=torch.zeros_like(target) if zero else target.detach().clone()
    if name.startswith('recurrent.'):
        for gate in range(3):
            old=slice(gate*128,(gate+1)*128);new=slice(gate*256,gate*256+128)
            if value.ndim==2:
                result[new].zero_();result[new,:value.shape[1]].copy_(value[old])
            else:result[new].copy_(value[old])
    elif name=='projection.weight':
        result.zero_();result[:,:128].copy_(value)
    else:raise ValueError('Only the original monaural GRU and projection are widened')
    return result


class CausalExpandedTemporalMonaural(CausalRecurrentAdaptMonaural):
    def __init__(self,wide_source):
        super().__init__(wide_source)
        with torch.random.fork_rng(devices=[]):
            self.recurrent=nn.GRU(816,256,batch_first=True,bidirectional=False)
            self.projection=nn.Linear(256,816)

    @torch.no_grad()
    def load_warm(self,state):
        target=self.state_dict()
        if set(state)!=set(target):raise ValueError('Use the complete original causal monaural weights')
        shapes={'recurrent.weight_ih_l0':(384,816),'recurrent.weight_hh_l0':(384,128),
                'recurrent.bias_ih_l0':(384,),'recurrent.bias_hh_l0':(384,), 'projection.weight':(816,128)}
        for name,value in state.items():
            expected=shapes.get(name,tuple(target[name].shape))
            if not isinstance(value,torch.Tensor) or tuple(value.shape)!=expected or value.dtype!=target[name].dtype or not torch.isfinite(value).all():
                raise ValueError('Warm weights must have the original finite shapes and dtypes')
        for name,value in state.items():
            target[name]=expand_temporal_tensor(name,value,target[name]) if name in shapes else value.detach().clone()
        self.load_state_dict(target,strict=True)

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
        if hidden is not None and (hidden.shape!=(1,batch,256) or hidden.dtype!=torch.float32 or hidden.device!=inpt.device):
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

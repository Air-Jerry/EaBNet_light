"""Unregistered current-frame frequency recurrence; no future time frames or added stream state."""
import torch
from torch import nn

from light_causal_fast_model import ActivationLayerNorm
from experiments.causal_recurrent_adapt_monaural import CausalRecurrentAdaptMonaural


class CurrentFrameFrequencyMix(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm=ActivationLayerNorm(16)
        # The GRU sequence axis is frequency; different time frames are independent batch items.
        self.frequency_gru=nn.GRU(16,24,batch_first=True,bidirectional=True)
        self.projection=nn.Linear(48,16)
        nn.init.zeros_(self.projection.weight);nn.init.zeros_(self.projection.bias)

    def forward(self,value):
        if value.ndim!=4 or value.shape[1]!=16:
            raise ValueError('Expected current-frame features(B,16,T,F)')
        batch,channels,frames,frequencies=value.shape
        with torch.autocast(value.device.type,enabled=False):
            sequence=value.float().permute(0,2,3,1).reshape(batch*frames,frequencies,channels)
            mixed,_=self.frequency_gru(self.norm(sequence))
            delta=self.projection(mixed).reshape(batch,frames,frequencies,channels).permute(0,3,1,2)
            result=value.float()+delta
        return result.to(value.dtype)


class CausalFrequencyMixMonaural(CausalRecurrentAdaptMonaural):
    def __init__(self,wide_source):
        super().__init__(wide_source)
        with torch.random.fork_rng(devices=[]):self.frequency_mix=CurrentFrameFrequencyMix()

    @torch.no_grad()
    def load_warm(self,state):
        target=self.state_dict();extra={name for name in target if name.startswith('frequency_mix.')}
        if set(state)!=set(target)-extra:
            raise ValueError('Use the complete original causal monaural weights')
        if any(not isinstance(v,torch.Tensor) or v.shape!=target[k].shape or v.dtype!=target[k].dtype
               or not torch.isfinite(v).all() for k,v in state.items()):
            raise ValueError('Warm weights must have the original finite shapes and dtypes')
        missing=self.load_state_dict(state,strict=False)
        assert set(missing.missing_keys)==extra and not missing.unexpected_keys
        nn.init.zeros_(self.frequency_mix.projection.weight);nn.init.zeros_(self.frequency_mix.projection.bias)

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
            value=self.frequency_mix(value)
            correction=self.output(value).float()*scale[:,None]
        output=estimate.float()+correction
        updated=dict(base=base_state,monaural_hidden=hidden.detach().clone()) if cache else None
        return output,updated

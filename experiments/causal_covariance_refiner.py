"""Unregistered causal mask/covariance refinement of a frozen causal model.

Souden algebra is documented in output/causal_covariance_design_20261102.md.
No real-data quality, CUDA cost or deadline claim is made by this prototype.
"""
import torch
from torch import nn
import torch.nn.functional as F

from experiments.causal_fullband_tcn import CausalFullbandTCN
from light_causal_fast_model import ActivationLayerNorm


def _covariance_block(signal,mask,speech,noise,powers):
    """Every output is an EMA prefix, including within this at-most64-frame block."""
    outer=signal.unsqueeze(-1)*signal.conj().unsqueeze(-2)
    weighted=outer/powers[None,:,None,None,None]
    prefix_s=(weighted*mask[...,None,None]).cumsum(1)
    prefix_n=(weighted*(1-mask[...,None,None])).cumsum(1)
    factor=powers[None,:,None,None,None]
    return factor*(speech[:,None]+.02*prefix_s),factor*(noise[:,None]+.02*prefix_n)


def _souden_weights(speech,noise,identity):
    """Reference0; independently trace-scaled covariances and FP32 diagonal loading."""
    channels=speech.shape[-1]
    scale_s=(speech.diagonal(dim1=-2,dim2=-1).real.sum(-1)/channels).clamp_min(1e-8)
    scale_n=(noise.diagonal(dim1=-2,dim2=-1).real.sum(-1)/channels).clamp_min(1e-8)
    normalized_s=speech/scale_s[...,None,None]
    normalized_n=noise/scale_n[...,None,None]
    # For PSD noise with mean diagonal1, loading .01 bounds its condition by801 at8 mics.
    product=torch.linalg.solve(normalized_n+.01*identity,normalized_s)
    trace=product.diagonal(dim1=-2,dim2=-1).real.sum(-1).clamp_min(1e-8)
    return product[..., :,0]/trace[...,None]


class CausalCovarianceRefiner(nn.Module):
    def __init__(self,base_model):
        super().__init__()
        if getattr(base_model,'is_causal',None) is not True or not callable(getattr(base_model,'process',None)):
            raise ValueError('The frozen source must itself implement a causal process')
        self.base=base_model;self.base.requires_grad_(False);self.base.zero_grad(set_to_none=True);self.base.eval()
        self.M=8;self.is_causal=True
        self.context=int(getattr(base_model,'context',128))
        self.feature_norm=ActivationLayerNorm(24)
        self.input=nn.Conv2d(24,32,(1,3),padding=(0,1))
        self.recurrent_norm=ActivationLayerNorm(32)
        self.recurrent=nn.GRU(32,32,batch_first=True,bidirectional=False)
        self.output=nn.Conv2d(32,2,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.register_buffer('ema_powers',torch.tensor(.98,dtype=torch.float32).pow(torch.arange(1,65,dtype=torch.float32)),persistent=False)
        self.register_buffer('identity',torch.eye(8,dtype=torch.complex64),persistent=False)

    def train(self,mode=True):
        super().train(mode);self.base.eval();return self

    def process(self,inpt,state=None):
        if inpt.ndim!=5 or inpt.shape[-3:]!=(257,8,2) or min(inpt.shape[:2])<1:
            raise ValueError('Expected compressed STFT(B,T,257,8,2)')
        cache=state is not None
        if cache and (not isinstance(state,dict) or (state and set(state)!={'base','speech','noise','mask_hidden'})):
            raise ValueError('Use a complete independently owned covariance stream state')
        memories={} if state is None else state
        batch,frames=inpt.shape[:2]
        estimate,base_state=self.base.process(inpt,memories.get('base',{}) if cache else None)
        if estimate.shape!=(batch,2,frames,257):raise ValueError('Causal source returned an incompatible spectrum')
        with torch.autocast(inpt.device.type,enabled=False):
            spectra=torch.view_as_complex(CausalFullbandTCN.decompress(inpt.float()).contiguous())
            base_ri=estimate.float().permute(0,2,3,1)
            base_raw=torch.view_as_complex(CausalFullbandTCN.decompress(base_ri).contiguous())
            reference=spectra[...,0]
            speech_power=base_raw.abs().square()
            noise_power=(reference-base_raw).abs().square()
            reference_power=reference.abs().square()
            scale=reference_power.mean(-1,keepdim=True).clamp_min(1e-8)
            logs=torch.stack([torch.log1p(value/scale) for value in (speech_power,noise_power,reference_power)],-1)
            magnitude=spectra.abs();unit=spectra/magnitude.clamp_min(1e-8)
            phase=unit[...,1:]*unit[...,:1].conj()
            level=5*torch.tanh(torch.log((magnitude[...,1:]+1e-8)/(magnitude[...,:1]+1e-8))/5)
            features=torch.cat((logs,phase.real,phase.imag,level),-1)
            value=F.silu(self.input(self.feature_norm(features).permute(0,3,1,2)))
            value=value.permute(0,3,2,1).reshape(batch*257,frames,32)
            hidden=memories.get('mask_hidden')
            if hidden is not None and (hidden.shape!=(1,batch*257,32) or hidden.dtype!=torch.float32 or hidden.device!=inpt.device):
                raise ValueError('Mask hidden state differs from this stream')
            value,hidden=self.recurrent(self.recurrent_norm(value),hidden)
            value=value.reshape(batch,257,frames,32).permute(0,3,2,1)
            adjustment=self.output(value).permute(0,2,3,1)
            prior=(torch.log(speech_power+1e-8)-torch.log(noise_power+1e-8)).clamp(-8,8)
            mask=torch.sigmoid(prior+adjustment[...,0]).clamp(.001,.999)
            gate=torch.tanh(adjustment[...,1])
            covariance_shape=(batch,257,8,8)
            covariances=[]
            for name in ('speech','noise'):
                previous=memories.get(name)
                if previous is None:previous=spectra.new_zeros(covariance_shape)
                elif previous.shape!=covariance_shape or previous.dtype!=torch.complex64 or previous.device!=inpt.device:
                    raise ValueError('Covariance state differs from this stream')
                covariances.append(previous)
            speech,noise=covariances;parts=[]
            for start in range(0,frames,64):
                stop=min(start+64,frames)
                speech_frames,noise_frames=_covariance_block(spectra[:,start:stop],mask[:,start:stop],speech,noise,self.ema_powers[:stop-start])
                weights=_souden_weights(speech_frames,noise_frames,self.identity)
                enhanced=(weights.conj()*spectra[:,start:stop]).sum(-1)
                enhanced_ri=CausalFullbandTCN.compress(torch.view_as_real(enhanced))
                # Zero readout keeps the trained source output without a compress/decompress round trip.
                parts.append(base_ri[:,start:stop]+gate[:,start:stop].unsqueeze(-1)*(enhanced_ri-base_ri[:,start:stop]))
                speech,noise=speech_frames[:,-1],noise_frames[:,-1]
            output=torch.cat(parts,1).permute(0,3,1,2)
        updated=dict(base=base_state,speech=speech.detach().clone(),noise=noise.detach().clone(),
            mask_hidden=hidden.detach().clone()) if cache else None
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

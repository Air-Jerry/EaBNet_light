"""Experimental prefix-PSD Souden beamformer guided by a causal neural estimate.

Own implementation of the published covariance formula, not author code.
Only current/past mixture and neural estimates enter the mask and matrices.
No dataset-quality or CUDA realtime claim is made by this prototype.
"""

import math

import torch
from torch import nn


class CausalSoudenBeamformer(nn.Module):
    def __init__(self, decay=.98, diagonal_loading=.001, mask_floor=.001, chunk_size=32, reference_mic=0):
        super().__init__()
        if not all(math.isfinite(v) for v in (decay,diagonal_loading,mask_floor)):
            raise ValueError('Use finite covariance settings')
        if not 0<decay<1 or diagonal_loading<=0 or not 0<mask_floor<.5 or chunk_size<1 or reference_mic<0:
            raise ValueError('Use stable decay, positive loading, bounded mask and valid dimensions')
        self.decay,self.diagonal_loading,self.mask_floor=float(decay),float(diagonal_loading),float(mask_floor)
        self.chunk_size,self.reference_mic=int(chunk_size),int(reference_mic)

    def _weights(self, speech, noise):
        microphones=noise.shape[-1]
        speech=(speech+speech.mH)*.5
        noise=(noise+noise.mH)*.5
        energy=noise.diagonal(dim1=-2,dim2=-1).real.mean(-1)
        loading=(self.diagonal_loading*energy).clamp_min(1e-8)
        regularized=noise+loading[...,None,None]*torch.eye(microphones,device=noise.device,dtype=noise.dtype)
        transfer,_=torch.linalg.solve_ex(regularized,speech,check_errors=False)
        trace=transfer.diagonal(dim1=-2,dim2=-1).sum(-1).real.clamp_min(1e-12)
        return transfer[..., :,self.reference_mic]/trace[...,None]

    def process(self, mixture, estimate, state=None, cache=False):
        if mixture.ndim!=4 or estimate.shape!=mixture.shape[:-1] or min(mixture.shape)<1:
            raise ValueError('Expected mixture(B,T,F,M), estimate(B,T,F)')
        if not mixture.is_complex() or not estimate.is_complex() or mixture.device!=estimate.device:
            raise ValueError('Use complex spectra on the same device')
        if self.reference_mic>=mixture.shape[-1]:raise ValueError('Reference microphone out of range')
        batch,frames,frequencies,microphones=mixture.shape
        with torch.autocast(mixture.device.type,enabled=False):
            value,prediction=mixture.to(torch.complex128),estimate.to(torch.complex128)
            shape=(batch,frequencies,microphones,microphones)
            if state is None:
                speech,noise=value.new_zeros(shape),value.new_zeros(shape)
            else:
                speech,noise=state['speech_covariance'],state['noise_covariance']
                if any(x.shape!=shape or x.dtype!=value.dtype or x.device!=value.device for x in (speech,noise)):
                    raise ValueError('Covariance state differs from the stream')
            signal_power=prediction.abs().square()
            residual_power=(value[...,self.reference_mic]-prediction).abs().square()
            mask=(signal_power/(signal_power+residual_power+1e-12)).clamp(self.mask_floor,1-self.mask_floor)
            outputs=[]
            for start in range(0,frames,self.chunk_size):
                current=value[:,start:start+self.chunk_size]
                probability=mask[:,start:start+self.chunk_size][...,None,None]
                outer=current.unsqueeze(-1)*current.conj().unsqueeze(-2)
                powers=self.decay**torch.arange(1,current.shape[1]+1,device=value.device,dtype=torch.float64)
                powers=powers[None,:,None,None,None]
                speech_sequence=speech[:,None]*powers+(1-self.decay)*powers*torch.cumsum(probability*outer/powers,dim=1)
                noise_sequence=noise[:,None]*powers+(1-self.decay)*powers*torch.cumsum((1-probability)*outer/powers,dim=1)
                weights=self._weights(speech_sequence,noise_sequence)
                outputs.append((weights.conj()*current).sum(-1))
                speech,noise=speech_sequence[:,-1],noise_sequence[:,-1]
            result=torch.cat(outputs,dim=1).to(estimate.dtype)
            updated=dict(speech_covariance=speech.detach().clone(),noise_covariance=noise.detach().clone()) if cache else None
        return result,updated

    def forward(self, mixture, estimate):
        return self.process(mixture,estimate)[0]

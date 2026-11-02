"""Batch equal-length utterances in the existing quality loss without changing weights."""

from contextlib import contextmanager
import math

import torch

from light_quality_loss import QualityObjective


@contextmanager
def batched_quality_training():
    original=QualityObjective.__call__

    def objective(self,estimate,target_stft,frame_list):
        lengths=self._lengths
        valid=(lengths is not None and self._target is not None and len(lengths)>0
            and len(set(lengths))==1 and estimate.ndim==4 and estimate.shape==target_stft.shape
            and estimate.shape[0]==len(lengths) and estimate.shape[1]==2
            and estimate.shape[-1]==self.n_fft//2+1
            and estimate.device==target_stft.device==self._target.device
            and [int(v) for v in frame_list]==[n//self.hop_length+1 for n in lengths]
            and estimate.shape[2]>=lengths[0]//self.hop_length+1)
        if not valid:return original(self,estimate,target_stft,frame_list)
        frames=lengths[0]//self.hop_length+1
        with torch.autocast(estimate.device.type,enabled=False):
            estimated=estimate[:,:,:frames].float();reference=target_stft[:,:,:frames].float()
            estimated_mag=torch.linalg.vector_norm(estimated,dim=1)
            target_mag=torch.linalg.vector_norm(reference,dim=1)
            magnitude=(estimated_mag-target_mag).abs().mean((1,2))/target_mag.mean((1,2)).clamp_min(1e-3)
            complex_loss=(estimated-reference).abs().mean((1,2,3))/reference.abs().mean((1,2,3)).clamp_min(1e-3)
            spectral=(.5*(magnitude+complex_loss)).mean()
            reconstructed=self.reconstruct(estimated,lengths[0])
            target=self._target[:,:lengths[0]]
            terms=[]
            for size in (256,512,1024):
                settings=dict(n_fft=size,hop_length=size//4,win_length=size,
                    window=self._window(size,estimate.device),center=True,
                    pad_mode='reflect' if lengths[0]>size//2 else 'constant',
                    normalized=False,onesided=True,return_complex=True)
                estimated_mag=torch.stft(reconstructed,**settings).abs()
                target_mag=torch.stft(target,**settings).abs()
                denominator=torch.linalg.vector_norm(target_mag.flatten(1),dim=1).clamp_min(math.sqrt(target_mag[0].numel())*1e-3)
                convergence=torch.linalg.vector_norm((estimated_mag-target_mag).flatten(1),dim=1)/denominator
                logarithm=(torch.log(estimated_mag+1e-5)-torch.log(target_mag+1e-5)).abs().mean((1,2))
                terms.append(convergence+logarithm)
            multiscale=torch.stack(terms,dim=1).mean(1).mean()
            target_rms=target.square().mean(1).sqrt().clamp_min(1e-3)
            waveform=((reconstructed-target).abs().mean(1)/target_rms).mean()
            total=spectral+.1*multiscale+.1*waveform
        components=torch.stack((spectral,multiscale,waveform,total)).detach()*len(lengths)
        if self._component_totals is None:self._component_totals=components
        else:self._component_totals.add_(components)
        self._sample_count+=len(lengths);self._batch_count+=1
        return total

    QualityObjective.__call__=objective
    try:yield
    finally:QualityObjective.__call__=original

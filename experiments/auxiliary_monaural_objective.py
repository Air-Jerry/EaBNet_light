"""Unregistered training-only auxiliary supervision through the existing loss hooks."""
import math

import torch
from torch import nn

from train_light_causal_pmsqe import FrontendPMSQE


class AuxiliaryMonauralObjective(FrontendPMSQE):
    """Capture one mono forward per batch without adding target inputs to the model."""

    def __init__(self,monaural,weight=.3):
        if not isinstance(monaural,nn.Module):raise ValueError('Bind the actual monaural module')
        if not math.isfinite(weight) or weight<0:raise ValueError('Auxiliary weight must be finite and nonnegative')
        self.monaural=monaural;self.auxiliary_weight=float(weight)
        self._hook=self._captured=self._auxiliary=None;self._armed=False
        super().__init__(512,160,320,.5)
        self._auxiliary=FrontendPMSQE(512,160,320,.5)

    def __enter__(self):
        if self._hook is not None:raise RuntimeError('Auxiliary objective is already bound')
        self._hook=self.monaural.register_forward_hook(self._capture)
        return self

    def __exit__(self,*unused):
        self._hook.remove();self._hook=self._captured=None;self._armed=False

    def _capture(self,module,arguments,output):
        if not self._armed:return
        if self._captured is not None:raise RuntimeError('Expected one monaural forward per feature batch')
        if not torch.is_tensor(output):raise ValueError('Monaural forward must return its compressed RI spectrum')
        self._captured=output

    def build_features(self,*args,**kwargs):
        if self._hook is None:raise RuntimeError('Use the bound auxiliary objective context')
        if self._armed:raise RuntimeError('Consume the previous feature batch loss before building another')
        values=super().build_features(*args,**kwargs)
        self._armed=True
        return values

    def __call__(self,estimate,target_stft,frame_list):
        if not self._armed or self._captured is None:
            raise RuntimeError('Build features and run the bound monaural branch before the auxiliary loss')
        mono=self._captured;self._captured=None;self._armed=False
        if mono.shape!=estimate.shape or mono.device!=estimate.device:
            raise ValueError('Monaural and fusion spectra must describe the same batch')
        primary=super().__call__(estimate,target_stft,frame_list)
        if self.auxiliary_weight==0:return primary
        self._auxiliary._target=self._target;self._auxiliary._lengths=self._lengths
        auxiliary=self._auxiliary(mono,target_stft,frame_list)
        return primary+self.auxiliary_weight*auxiliary

    def reset_stats(self):
        super().reset_stats();self._captured=None;self._armed=False
        if self._auxiliary is not None:self._auxiliary.reset_stats()

    def stats(self):
        primary=super().stats()
        auxiliary=self._auxiliary.stats()['combined_total'] if self._auxiliary is not None else 0.
        return dict(primary,auxiliary_monaural_total=auxiliary,auxiliary_monaural_weight=self.auxiliary_weight,
            objective_total=primary['combined_total']+self.auxiliary_weight*auxiliary)

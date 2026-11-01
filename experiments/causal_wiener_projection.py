"""Experimental causal ridge projection onto the observed microphone signals.

The cross-covariance uses the neural estimate, never a clean reference. This is
not Souden MVDR or an author implementation. Dataset quality and GPU cost are
unverified. Complex128 accumulation/solves preserve batch/stream agreement.
"""

import torch
from torch import nn


class CausalWienerProjection(nn.Module):
    def __init__(self, decay=.96, diagonal_loading=.01, chunk_size=32):
        super().__init__()
        if not 0 < decay < 1 or diagonal_loading <= 0 or chunk_size < 1:
            raise ValueError('Use a stable decay, positive diagonal loading and chunk size')
        self.decay, self.diagonal_loading = float(decay), float(diagonal_loading)
        self.chunk_size = int(chunk_size)

    def process(self, mixture, estimate, state=None, cache=False):
        if mixture.ndim != 4 or estimate.shape != mixture.shape[:-1] or min(mixture.shape) < 1:
            raise ValueError('Expected mixture(B,T,F,M) and estimate(B,T,F)')
        if not mixture.is_complex() or not estimate.is_complex() or mixture.device != estimate.device:
            raise ValueError('Use complex spectra on the same device')
        batch, frames, frequencies, microphones = mixture.shape
        with torch.autocast(mixture.device.type, enabled=False):
            value, target = mixture.to(torch.complex128), estimate.to(torch.complex128)
            shape = (batch, frequencies, microphones, microphones)
            if state is None:
                covariance = value.new_zeros(shape)
                correlation = value.new_zeros(shape[:-1])
            else:
                covariance, correlation = state['covariance'], state['correlation']
                if covariance.shape != shape or correlation.shape != shape[:-1]:
                    raise ValueError('Projection state differs from the stream shape')
                if covariance.dtype != value.dtype or covariance.device != value.device:
                    raise ValueError('Projection state dtype/device differs from the stream')
            identity = torch.eye(microphones, device=value.device, dtype=value.dtype)
            outputs = []
            for start in range(0, frames, self.chunk_size):
                current = value[:, start:start + self.chunk_size]
                desired = target[:, start:start + self.chunk_size]
                length = current.shape[1]
                powers = self.decay ** torch.arange(1, length + 1, device=value.device, dtype=torch.float64)
                outer = current.unsqueeze(-1) * current.conj().unsqueeze(-2)
                cross = current * desired.conj().unsqueeze(-1)
                p = powers[None, :, None, None, None]
                matrices = covariance[:, None] * p + (1 - self.decay) * p * torch.cumsum(outer / p, dim=1)
                p = powers[None, :, None, None]
                vectors = correlation[:, None] * p + (1 - self.decay) * p * torch.cumsum(cross / p, dim=1)
                matrices = (matrices + matrices.mH) * .5
                energy = matrices.diagonal(dim1=-2, dim2=-1).real.mean(-1)
                loading = (self.diagonal_loading * energy).clamp_min(1e-8)
                regularized = matrices + loading[..., None, None] * identity
                weights, _ = torch.linalg.solve_ex(regularized, vectors.unsqueeze(-1), check_errors=False)
                outputs.append((weights.squeeze(-1).conj() * current).sum(-1))
                covariance, correlation = matrices[:, -1], vectors[:, -1]
            result = torch.cat(outputs, dim=1).to(estimate.dtype)
            updated = dict(covariance=covariance.detach().clone(), correlation=correlation.detach().clone()) if cache else None
        return result, updated

    def forward(self, mixture, estimate):
        return self.process(mixture, estimate)[0]

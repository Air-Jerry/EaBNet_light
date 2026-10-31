"""Training-only PMSQE regularization; model inputs and inference remain unchanged."""

import math

import torch

from light_quality_loss import QualityObjective
from third_party.asteroid_pmsqe.pmsqe import SingleSrcPMSQE


def physical_power(compressed, power=0.5):
    """Invert the frontend's compression before computing physical STFT power."""
    magnitude = torch.linalg.vector_norm(compressed.float(), dim=1)
    factor = magnitude.clamp_min(1e-8 ** power).pow(1.0 / power - 1.0)
    return (compressed.float() * factor.unsqueeze(1)).square().sum(dim=1)


class PMSQEObjective(QualityObjective):
    def __init__(self, weight=0.1):
        super().__init__()
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('PMSQE weight must be finite and nonnegative')
        self.weight = float(weight)
        # Upstream assumes a full512 Hann; the existing320 Hann has120 window energy.
        self.perceptual = SingleSrcPMSQE(window_name='hann', window_weight=512 / 320)
        self._perceptual_device = None

    def __call__(self, estimate, target_stft, frame_list):
        physical = super().__call__(estimate, target_stft, frame_list)
        if self.weight == 0:
            return physical
        if self._perceptual_device != estimate.device:
            self.perceptual.to(estimate.device)
            self._perceptual_device = estimate.device
        with torch.autocast(estimate.device.type, enabled=False):
            counts = torch.as_tensor(frame_list, device=estimate.device)
            valid = torch.arange(estimate.shape[2], device=estimate.device)[None, :] < counts[:, None]
            mask = valid[:, None, :, None]
            est = physical_power(estimate.masked_fill(~mask, 0), self.power) + 1e-12
            ref = physical_power(target_stft.masked_fill(~mask, 0), self.power) + 1e-12
            # Frequency-first avoids upstream axis inference ambiguity when T equals257.
            per_utterance = self.perceptual(est.transpose(1, 2), ref.transpose(1, 2), valid[:, None, :])
            lengths = torch.as_tensor(self._lengths, device=estimate.device)
            samples = torch.arange(self._target.shape[1], device=estimate.device)[None, :] < lengths[:, None]
            energy = self._target.square().masked_fill(~samples, 0).sum(dim=1) / lengths
            audible = (energy > 1e-12).to(per_utterance.dtype)
            perceptual = (per_utterance * audible).sum() / audible.sum().clamp_min(1)
            combined = physical + self.weight * perceptual
        total = perceptual.detach() * len(frame_list)
        self._pmsqe_total = total if self._pmsqe_total is None else self._pmsqe_total + total
        return combined

    def reset_stats(self):
        super().reset_stats()
        self._pmsqe_total = None

    def stats(self):
        result = super().stats()
        result['pmsqe'] = 0.0 if self._pmsqe_total is None else float(self._pmsqe_total / self._sample_count)
        result['combined_total'] = result['total'] + self.weight * result['pmsqe']
        return result

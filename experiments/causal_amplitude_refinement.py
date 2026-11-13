"""Predict a bounded magnitude correction from current-frame eight-microphone spectra."""
import torch
from torch import nn

from experiments.causal_phase_filter_monaural import CausalParallelPhaseFilterMonaural


class CausalAmplitudeRefinement(CausalParallelPhaseFilterMonaural):
    def __init__(self, wide_source):
        super().__init__(wide_source)
        with torch.random.fork_rng(devices=[]):
            self.amplitude_head = nn.Sequential(
                nn.Conv2d(9, 16, (1, 9), padding=(0, 4)),
                nn.SiLU(),
                nn.Conv2d(16, 16, (1, 5), padding=(0, 2)),
                nn.SiLU(),
                nn.Conv2d(16, 1, 1),
            )
        nn.init.zeros_(self.amplitude_head[-1].weight)
        nn.init.zeros_(self.amplitude_head[-1].bias)

    def process(self, inpt, state=None):
        estimate, cache = super().process(inpt, state)
        mixture = torch.linalg.vector_norm(inpt.float(), dim=-1).permute(0, 3, 1, 2)
        magnitude = torch.linalg.vector_norm(estimate.detach().float(), dim=1, keepdim=True)
        scale = mixture[:, :1].mean(-1, keepdim=True).clamp_min(1e-4)
        features = (torch.cat((mixture, magnitude), dim=1) / scale).log1p()
        enabled = self.training and inpt.device.type == 'cuda'
        with torch.autocast(inpt.device.type, dtype=torch.bfloat16, enabled=enabled):
            gain = .25 * self.amplitude_head(features).float().tanh()
        return estimate * (1 + gain), cache

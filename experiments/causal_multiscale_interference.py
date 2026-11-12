"""Causal multi-scale temporal estimator for the interference branch."""
import torch
from torch import nn

from experiments.causal_interference_monaural import CausalInterferenceMonaural


class MultiScaleNoiseContext(nn.Module):
    dilations = (1, 4, 16)
    history_frames = 2 * max(dilations)

    def __init__(self, width):
        super().__init__()
        self.filters = nn.ModuleList([
            nn.Conv2d(width, width, (3, 1), dilation=(dilation, 1), groups=width)
            for dilation in self.dilations
        ])
        self.combine = nn.Conv2d(width * len(self.dilations), width, 1)

    def forward(self, padded):
        bands = [layer(padded[:, :, self.history_frames - 2 * dilation:])
                 for layer, dilation in zip(self.filters, self.dilations, strict=True)]
        return self.combine(torch.cat(bands, dim=1))


class CausalMultiscaleInterference(CausalInterferenceMonaural):
    history_frames = MultiScaleNoiseContext.history_frames

    def __init__(self, wide_source):
        super().__init__(wide_source)
        with torch.random.fork_rng(devices=[]):
            self.interference['temporal'] = MultiScaleNoiseContext(self.width)

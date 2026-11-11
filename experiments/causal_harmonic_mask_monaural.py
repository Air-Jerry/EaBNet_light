"""Current-frame harmonic mask on the recurrent monaural estimate."""
import torch
from torch import nn

from experiments.causal_recurrent_adapt_monaural import CausalRecurrentAdaptMonaural


class CausalHarmonicMaskMonaural(CausalRecurrentAdaptMonaural):
    def __init__(self, wide_source):
        super().__init__(wide_source)
        with torch.random.fork_rng(devices=[]):
            self.harmonic_mask = nn.Sequential(nn.Conv2d(10, 24, 1), nn.SiLU(), nn.Conv2d(24, 2, 1))
        nn.init.zeros_(self.harmonic_mask[-1].weight)
        nn.init.zeros_(self.harmonic_mask[-1].bias)
        bins = torch.arange(257)
        positions = bins[None] * torch.tensor([1 / 3, .5, 2., 3.])[:, None]
        lower = positions.long().clamp_max(256)
        self.register_buffer('harmonic_lower', lower.flatten(), persistent=False)
        self.register_buffer('harmonic_upper', (lower + 1).clamp_max(256).flatten(), persistent=False)
        self.register_buffer('harmonic_fraction', (positions - positions.floor()).flatten(), persistent=False)
        self.register_buffer('harmonic_valid', (positions <= 256).flatten(), persistent=False)

    def load_warm(self, weights):
        current = self.state_dict()
        added = {name for name in current if name.startswith('harmonic_mask.')}
        if set(weights) != set(current) - added:
            raise ValueError('Use all weights from the original recurrent model')
        self.load_state_dict({**weights, **{name: current[name] for name in added}})
        nn.init.zeros_(self.harmonic_mask[-1].weight)
        nn.init.zeros_(self.harmonic_mask[-1].bias)

    def process(self, inpt, state=None):
        estimate, updated = super().process(inpt, state)
        with torch.autocast(inpt.device.type, enabled=False):
            reference = torch.linalg.vector_norm(inpt[..., 0, :].float(), dim=-1)
            enhanced = torch.linalg.vector_norm(estimate.detach().float(), dim=1)
            magnitudes = torch.stack((reference, enhanced), dim=1)
            magnitudes = magnitudes / magnitudes.mean(-1, keepdim=True).clamp_min(1e-4)
            low = magnitudes.index_select(-1, self.harmonic_lower)
            high = magnitudes.index_select(-1, self.harmonic_upper)
            related = torch.lerp(low, high, self.harmonic_fraction) * self.harmonic_valid
            batch, _, frames, bins = magnitudes.shape
            related = related.reshape(batch, 2, frames, 4, bins).transpose(2, 3).reshape(batch, 8, frames, bins)
            cues = torch.cat((magnitudes, related), dim=1)
        enabled = self.training and inpt.device.type == 'cuda'
        with torch.autocast(inpt.device.type, dtype=torch.bfloat16, enabled=enabled):
            gain = self.harmonic_mask(cues).float().tanh() * .5
        real, imag = estimate[:, 0], estimate[:, 1]
        correction = torch.stack((gain[:, 0] * real - gain[:, 1] * imag,
                                  gain[:, 0] * imag + gain[:, 1] * real), dim=1)
        return estimate + correction, updated

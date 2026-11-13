"""Refine the recurrent estimate with a causal complex filter across STFT frames."""
import math

import torch
from torch import nn

from experiments.causal_recurrent_adapt_monaural import CausalRecurrentAdaptMonaural


class CausalPhaseFilterMonaural(CausalRecurrentAdaptMonaural):
    history_frames = 2

    def __init__(self, wide_source):
        super().__init__(wide_source)
        with torch.random.fork_rng(devices=[]):
            self.phase_filter = nn.Sequential(
                nn.Conv2d(10, 24, (1, 5), padding=(0, 2)),
                nn.SiLU(),
                nn.Conv2d(24, 24, (1, 3), padding=(0, 1), groups=24),
                nn.SiLU(),
                nn.Conv2d(24, 6, 1),
            )
        self._reset_filter()
        # The bin-center carrier advances by k * 160 / 512 cycles per hop.
        bins = torch.arange(257)
        delays = torch.arange(1, self.history_frames + 1)[:, None]
        angle = (bins * delays * 5).remainder(16).float() * (math.tau / 16)
        self.register_buffer('advance_cos', angle.cos(), persistent=False)
        self.register_buffer('advance_sin', angle.sin(), persistent=False)

    def _reset_filter(self):
        nn.init.zeros_(self.phase_filter[-1].weight)
        nn.init.zeros_(self.phase_filter[-1].bias)

    def load_warm(self, weights):
        current = self.state_dict()
        added = {name for name in current if name.startswith('phase_filter.')}
        if set(weights) != set(current) - added:
            raise ValueError('Use all weights from the original recurrent model')
        self.load_state_dict({**weights, **{name: current[name] for name in added}})
        self._reset_filter()

    def process(self, inpt, state=None):
        streaming = state is not None
        if streaming and (not isinstance(state, dict) or (state and set(state) != {'speech', 'phase_history'})):
            raise ValueError('Use a complete independently owned phase filter state')
        memories = {} if state is None else state
        estimate, updated = super().process(inpt, memories.get('speech', {}) if streaming else None)
        with torch.autocast(inpt.device.type, enabled=False):
            reference = inpt[..., 0, :].float().permute(0, 3, 1, 2)
            current = torch.cat((reference, estimate.detach().float()), dim=1)
            batch, _, frames, bins = current.shape
            shape = (batch, 4, self.history_frames, bins)
            history = memories.get('phase_history')
            if history is None:
                history = current.new_zeros(shape)
            elif history.shape != shape or history.dtype != current.dtype or history.device != current.device:
                raise ValueError('Phase history does not match the stream')
            extended = torch.cat((history, current), dim=2).reshape(batch, 2, 2, -1, bins)
            signals = extended[:, :, :, self.history_frames:]
            past = torch.stack([extended[:, :, :, offset:offset + frames]
                                for offset in reversed(range(self.history_frames))], dim=2)
            past_real, past_imag = past.unbind(dim=3)
            cosine = self.advance_cos[:, None]
            sine = self.advance_sin[:, None]
            past_real, past_imag = (past_real * cosine - past_imag * sine,
                                   past_real * sine + past_imag * cosine)
            real, imag = signals.unbind(dim=2)
            magnitude = torch.linalg.vector_norm(signals, dim=2)
            scale = magnitude[:, :1].mean(-1, keepdim=True).clamp_min(1e-4)
            previous_power = past_real.square() + past_imag.square()
            floor = (.1 * scale.unsqueeze(2)).square()
            denominator = ((magnitude.unsqueeze(2).square() + floor) * (previous_power + floor)).sqrt()
            cross_real = (real.unsqueeze(2) * past_real + imag.unsqueeze(2) * past_imag) / denominator
            cross_imag = (imag.unsqueeze(2) * past_real - real.unsqueeze(2) * past_imag) / denominator
            features = torch.cat(((magnitude / scale).log1p(), cross_real.flatten(1, 2),
                                  cross_imag.flatten(1, 2)), dim=1)
        enabled = self.training and inpt.device.type == 'cuda'
        with torch.autocast(inpt.device.type, dtype=torch.bfloat16, enabled=enabled):
            coefficients = .25 * self.phase_filter(features).float().tanh()
        cr, ci = coefficients.reshape(batch, 3, 2, frames, bins).unbind(dim=2)
        er = torch.cat((estimate[:, 0:1], past_real[:, 1]), dim=1)
        ei = torch.cat((estimate[:, 1:2], past_imag[:, 1]), dim=1)
        correction = torch.stack(((cr * er - ci * ei).sum(1), (cr * ei + ci * er).sum(1)), dim=1)
        cache = None
        if streaming:
            tail = extended[:, :, :, -self.history_frames:].reshape(shape)
            cache = dict(speech=updated, phase_history=tail.detach().clone())
        return estimate + correction, cache

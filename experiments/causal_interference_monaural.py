"""Estimate interference from eight microphones and refine speech by subtraction."""
import torch
from torch import nn
from torch.nn import functional as F

from experiments.causal_recurrent_adapt_monaural import CausalRecurrentAdaptMonaural


def expand_spectrum(value):
    return value * torch.linalg.vector_norm(value, dim=1, keepdim=True)


def compress_spectrum(value):
    magnitude = torch.linalg.vector_norm(value, dim=1, keepdim=True)
    return value * magnitude.clamp_min(1e-8).rsqrt()


def interference_loss(noise, mixture, target, frame_lengths):
    """Compare against the residual in the reference microphone's physical spectrum."""
    reference = mixture[..., 0, :].permute(0, 3, 1, 2)
    with torch.autocast(noise.device.type, enabled=False):
        residual = compress_spectrum(expand_spectrum(reference.float()) - expand_spectrum(target.float()))
        lengths = torch.as_tensor(frame_lengths, device=noise.device)
        valid = torch.arange(noise.shape[2], device=noise.device)[None] < lengths[:, None]
        valid = valid[:, None, :, None]
        bins = lengths * noise.shape[1] * noise.shape[-1]
        error = (noise.float() - residual).masked_fill(~valid, 0).abs().sum((1, 2, 3))
        magnitude = residual.masked_fill(~valid, 0).abs().sum((1, 2, 3))
        return (error / magnitude.clamp_min(1e-3 * bins)).mean()


class CausalInterferenceMonaural(CausalRecurrentAdaptMonaural):
    history_frames = 8
    width = 24

    def __init__(self, wide_source):
        super().__init__(wide_source)
        with torch.random.fork_rng(devices=[]):
            self.interference = nn.ModuleDict({
                'stem': nn.Conv2d(18, self.width, (1, 5), padding=(0, 2)),
                'temporal': nn.Conv2d(self.width, self.width, (3, 1), dilation=(4, 1), groups=self.width),
                'noise': nn.Conv2d(self.width, 2, 1),
                'gate': nn.Conv2d(self.width, 1, 1),
            })
        self._reset_gate()

    def _reset_gate(self):
        nn.init.zeros_(self.interference['gate'].weight)
        nn.init.zeros_(self.interference['gate'].bias)

    def load_warm(self, weights):
        original = self.state_dict()
        added = {key for key in original if key.startswith('interference.')}
        if set(weights) != set(original) - added:
            raise ValueError('Use all weights from the original recurrent model')
        self.load_state_dict({**weights, **{key: original[key] for key in added}})
        self._reset_gate()

    def _estimate(self, inpt, state):
        streaming = state is not None
        if streaming and (not isinstance(state, dict) or (state and set(state) != {'speech', 'noise_history'})):
            raise ValueError('Use a complete independently owned interference stream state')
        memories = {} if state is None else state
        estimate, updated = super().process(inpt, memories.get('speech', {}) if streaming else None)
        with torch.autocast(inpt.device.type, enabled=False):
            reference = inpt[..., 0, :].float().permute(0, 3, 1, 2)
            scale = torch.linalg.vector_norm(reference, dim=1, keepdim=True).mean(-1, keepdim=True).clamp_min(1e-4)
            raw = inpt.float().flatten(-2).permute(0, 3, 1, 2)
            features = torch.cat((raw, estimate.detach()), dim=1) / scale
        enabled = self.training and inpt.device.type == 'cuda'
        with torch.autocast(inpt.device.type, dtype=torch.bfloat16, enabled=enabled):
            current = F.silu(self.interference['stem'](features))
            history = memories.get('noise_history')
            shape = (current.shape[0], self.width, self.history_frames, current.shape[-1])
            if history is None:
                history = current.new_zeros(shape)
            elif (history.shape != shape or history.dtype != current.dtype or history.device != current.device):
                raise ValueError('Interference history does not match the stream')
            extended = torch.cat((history, current), dim=2)
            hidden = current + F.silu(self.interference['temporal'](extended))
            noise = self.interference['noise'](hidden).float() * scale
            gate = .5 * self.interference['gate'](hidden).float().tanh()
        with torch.autocast(inpt.device.type, enabled=False):
            cleaned = compress_spectrum(expand_spectrum(reference) - expand_spectrum(noise))
            speech = estimate + gate * (cleaned - estimate)
        cache = dict(speech=updated, noise_history=extended[:, :, -self.history_frames:].detach().clone()) if streaming else None
        return speech, noise, cache

    def process(self, inpt, state=None):
        speech, _, cache = self._estimate(inpt, state)
        return speech, cache

    def forward(self, inpt, frame_lengths=None, *, return_noise=False):
        if not return_noise:
            return super().forward(inpt, frame_lengths)
        if frame_lengths is not None:
            lengths = torch.as_tensor(frame_lengths)
            if lengths.numel() != inpt.shape[0] or torch.any(lengths < 1) or torch.any(lengths > inpt.shape[1]):
                raise ValueError('One valid frame length is required per sample')
        speech, noise, _ = self._estimate(inpt, None)
        return speech, noise

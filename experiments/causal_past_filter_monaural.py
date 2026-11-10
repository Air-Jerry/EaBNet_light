"""Unregistered current-and-past multichannel filtering head over the complete causal source.

No lookahead or waveform postprocessing is added. Real-data quality and CUDA
training/streaming cost must be measured before any formal-model promotion.
"""
import torch
from torch import nn

from experiments.causal_fullband_tcn import CausalFullbandTCN
from experiments.causal_recurrent_adapt_monaural import CausalRecurrentAdaptMonaural


def past_filter_residual(raw, coefficients, history=None):
    if raw.ndim != 5 or raw.shape[-1] != 2 or coefficients.ndim != 6:
        raise ValueError('Use complex spectra(B,T,F,M,2) and coefficients(B,T,F,K,M,2)')
    batch, frames, frequencies, microphones, _ = raw.shape
    taps = coefficients.shape[3]
    if taps < 2 or coefficients.shape != (batch, frames, frequencies, taps, microphones, 2):
        raise ValueError('Filter coefficient shape differs from the spectra')
    shape = (batch, taps-1, frequencies, microphones, 2)
    if history is None:
        history = raw.new_zeros(shape)
    if history.shape != shape or history.dtype != raw.dtype or history.device != raw.device:
        raise ValueError('Past input state differs from this stream')
    extended = torch.cat((history, raw), 1)
    # Windows end at the current frame; the last coefficient is the current-frame tap.
    windows = extended.unfold(1, taps, 1).permute(0, 1, 2, 5, 3, 4)
    real = (coefficients[..., 0]*windows[..., 0]-coefficients[..., 1]*windows[..., 1]).sum((-2, -1))
    imaginary = (coefficients[..., 0]*windows[..., 1]+coefficients[..., 1]*windows[..., 0]).sum((-2, -1))
    return torch.stack((real, imaginary), -1), extended[:, -(taps-1):].detach().clone()


class CausalPastFilterMonaural(CausalRecurrentAdaptMonaural):
    def __init__(self, wide_source, taps=5):
        if isinstance(taps, bool) or not isinstance(taps, int) or taps < 2:
            raise ValueError('Use current-frame filtering with at least one past frame')
        super().__init__(wide_source)
        self.filter_taps = taps
        with torch.random.fork_rng(devices=[]):
            self.deep_filter = nn.Conv2d(16, taps*8*2, 1)
            nn.init.zeros_(self.deep_filter.weight)
            nn.init.zeros_(self.deep_filter.bias)

    @torch.no_grad()
    def load_warm(self, state):
        target = self.state_dict()
        extra = {name for name in target if name.startswith('deep_filter.')}
        if set(state) != set(target)-extra:
            raise ValueError('Use the complete original monaural/spatial state')
        if any(not isinstance(value, torch.Tensor) or value.shape != target[name].shape
               or value.dtype != target[name].dtype or not torch.isfinite(value).all() for name, value in state.items()):
            raise ValueError('Warm weights must have the original finite shapes and dtypes')
        missing = self.load_state_dict(state, strict=False)
        assert set(missing.missing_keys) == extra and not missing.unexpected_keys
        nn.init.zeros_(self.deep_filter.weight)
        nn.init.zeros_(self.deep_filter.bias)

    def process(self, inpt, state=None):
        if inpt.ndim != 5 or inpt.shape[-3:] != (257, 8, 2) or min(inpt.shape[:2]) < 1:
            raise ValueError('Expected compressed STFT(B,T,257,8,2)')
        cache = state is not None
        keys = {'base', 'monaural_hidden', 'input_history'}
        if cache and (not isinstance(state, dict) or (state and set(state) != keys)):
            raise ValueError('Use complete independently owned past-filter stream state')
        memories = {} if state is None else state
        batch, frames = inpt.shape[:2]
        hidden = memories.get('monaural_hidden')
        history = memories.get('input_history')
        if memories and (not isinstance(memories['base'], dict) or not isinstance(hidden, torch.Tensor)
                or not isinstance(history, torch.Tensor)):
            raise ValueError('Use complete independently owned past-filter stream state')
        if hidden is not None and (hidden.shape != (1, batch, 128)
                or hidden.dtype != torch.float32 or hidden.device != inpt.device):
            raise ValueError('Monaural hidden state differs from this stream')
        if history is not None and (history.shape != (batch, self.filter_taps-1, 257, 8, 2)
                or history.dtype != torch.float32 or history.device != inpt.device):
            raise ValueError('Past input state differs from this stream')
        estimate, base_state = self.base.process(inpt, memories.get('base', {}) if cache else None)
        if estimate.shape != (batch, 2, frames, 257):
            raise ValueError('Causal source returned an incompatible spectrum')
        with torch.autocast(inpt.device.type, enabled=False):
            reference = inpt[..., 0, :].float()
            scale = torch.linalg.vector_norm(reference, dim=-1).mean(-1, keepdim=True).clamp_min(1e-4)
            features = (torch.cat((reference, estimate.float().permute(0, 2, 3, 1)), -1)
                        / scale[..., None]).permute(0, 3, 1, 2)
        enabled = self.training and inpt.device.type == 'cuda'
        with torch.autocast(inpt.device.type, dtype=torch.bfloat16, enabled=enabled):
            value = features; skips = [features]
            for encoder in self.encoders:
                value = encoder(value); skips.append(value)
        with torch.autocast(inpt.device.type, enabled=False):
            sequence = value.permute(0, 2, 1, 3).reshape(batch, frames, 816).float()
            sequence, hidden = self.recurrent(self.recurrent_norm(sequence), hidden)
        with torch.autocast(inpt.device.type, dtype=torch.bfloat16, enabled=enabled):
            value = value+self.projection(sequence).reshape(batch, frames, 48, 17).permute(0, 2, 1, 3)
            for up, decoder, skip in zip(self.upsamplers, self.decoders, reversed(skips[:-1])):
                value = up(value)
                if value.shape[-1] != skip.shape[-1]:
                    raise RuntimeError('Frequency decoder lost alignment')
                value = decoder(torch.cat((value, skip), 1))
            original = estimate.float()+self.output(value).float()*scale[:, None]
            coefficients = self.deep_filter(value).float().permute(0, 2, 3, 1).reshape(batch, frames, 257, self.filter_taps, 8, 2)
        with torch.autocast(inpt.device.type, enabled=False):
            correction, history = past_filter_residual(CausalFullbandTCN.decompress(inpt.float()), coefficients, history)
            raw_original = CausalFullbandTCN.decompress(original.permute(0, 2, 3, 1))
            delta = CausalFullbandTCN.compress(raw_original+correction)-CausalFullbandTCN.compress(raw_original)
            output = original+delta.permute(0, 3, 1, 2)
        updated = dict(base=base_state, monaural_hidden=hidden.detach().clone(), input_history=history) if cache else None
        return output, updated

"""Experimental causal recurrent capacity alongside the original spectral branch.

The extra readout starts at zero to retain the complete trained source.
Real-data quality, dual-GPU cost and deployment deadlines are unverified.
"""
import torch
from torch import nn

from experiments.causal_decoder_monaural import CausalDecoderMonaural


class CausalCapacityMonaural(CausalDecoderMonaural):
    extra_hidden = 256

    def __init__(self, wide_source):
        super().__init__(wide_source)
        with torch.random.fork_rng(devices=[]):
            self.capacity_recurrent = nn.GRU(816, self.extra_hidden, batch_first=True)
            self.capacity_projection = nn.Linear(self.extra_hidden, 816)
        nn.init.zeros_(self.capacity_projection.weight)
        nn.init.zeros_(self.capacity_projection.bias)

    @torch.no_grad()
    def load_warm(self, state):
        target = self.state_dict()
        extra = {name for name in target if name.startswith(('capacity_recurrent.', 'capacity_projection.'))}
        if set(state) != set(target)-extra:
            raise ValueError('Use the complete original monaural/spatial state')
        if any(value.shape != target[name].shape or not torch.isfinite(value).all() for name, value in state.items()):
            raise ValueError('Warm weights must have the original finite shapes')
        missing = self.load_state_dict(state, strict=False)
        assert set(missing.missing_keys) == extra and not missing.unexpected_keys
        nn.init.zeros_(self.capacity_projection.weight)
        nn.init.zeros_(self.capacity_projection.bias)

    def process(self, inpt, state=None):
        if inpt.ndim != 5 or inpt.shape[-3:] != (257, 8, 2) or min(inpt.shape[:2]) < 1:
            raise ValueError('Expected compressed STFT(B,T,257,8,2)')
        cache = state is not None
        keys = {'base', 'monaural_hidden', 'capacity_hidden'}
        if cache and (not isinstance(state, dict) or (state and set(state) != keys)):
            raise ValueError('Use a complete independently owned capacity stream state')
        memories = {} if state is None else state
        batch, frames = inpt.shape[:2]
        if memories and not isinstance(memories['base'], dict):
            raise ValueError('Use a complete independently owned capacity stream state')
        for name, width in (('monaural_hidden', 128), ('capacity_hidden', self.extra_hidden)):
            hidden = memories.get(name)
            if memories and (not isinstance(hidden, torch.Tensor) or hidden.shape != (1, batch, width)
                    or hidden.dtype != torch.float32 or hidden.device != inpt.device):
                raise ValueError('Recurrent hidden state differs from this stream')
        estimate, base_state = self.base.process(inpt, memories.get('base', {}) if cache else None)
        if estimate.shape != (batch, 2, frames, 257):
            raise ValueError('Causal source returned an incompatible spectrum')
        with torch.autocast(inpt.device.type, enabled=False):
            reference = inpt[..., 0, :].float()
            scale = torch.linalg.vector_norm(reference, dim=-1).mean(-1, keepdim=True).clamp_min(1e-4)
            features = (torch.cat((reference, estimate.float().permute(0, 2, 3, 1)), -1)/scale[..., None]).permute(0, 3, 1, 2)
        enabled = self.training and inpt.device.type == 'cuda'
        with torch.autocast(inpt.device.type, dtype=torch.bfloat16, enabled=enabled):
            value = features
            skips = [features]
            for encoder in self.encoders:
                value = encoder(value)
                skips.append(value)
        with torch.autocast(inpt.device.type, enabled=False):
            normalized = self.recurrent_norm(value.permute(0, 2, 1, 3).reshape(batch, frames, 816).float())
            sequence, hidden = self.recurrent(normalized, memories.get('monaural_hidden'))
            capacity, capacity_hidden = self.capacity_recurrent(normalized, memories.get('capacity_hidden'))
        with torch.autocast(inpt.device.type, dtype=torch.bfloat16, enabled=enabled):
            projected = self.projection(sequence)+self.capacity_projection(capacity)
            value = value+projected.reshape(batch, frames, 48, 17).permute(0, 2, 1, 3)
            for up, decoder, skip in zip(self.upsamplers, self.decoders, reversed(skips[:-1])):
                value = up(value)
                if value.shape[-1] != skip.shape[-1]:
                    raise RuntimeError('Frequency decoder lost alignment')
                value = decoder(torch.cat((value, skip), 1))
            correction = self.output(value).float()*scale[:, None]
        output = estimate.float()+correction
        updated = dict(base=base_state, monaural_hidden=hidden.detach().clone(),
                       capacity_hidden=capacity_hidden.detach().clone()) if cache else None
        return output, updated

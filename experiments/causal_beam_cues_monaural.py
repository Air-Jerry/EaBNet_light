"""Experimental beam-relative cues for the causal monaural encoder."""
import torch
from torch import nn

from experiments.causal_recurrent_adapt_monaural import CausalRecurrentAdaptMonaural


class CausalBeamCuesMonaural(CausalRecurrentAdaptMonaural):
    def __init__(self, wide_source):
        super().__init__(wide_source)
        with torch.random.fork_rng(devices=[]):
            self.beam_cues = nn.Conv2d(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), bias=False)
        nn.init.zeros_(self.beam_cues.weight)

    def load_warm(self, weights):
        if 'beam_cues.weight' in weights:
            raise ValueError('Use the original recurrent model as the warm start')
        self.load_state_dict({**weights, 'beam_cues.weight': torch.zeros_like(self.beam_cues.weight)})

    def process(self, inpt, state=None):
        if inpt.ndim != 5 or inpt.shape[-3:] != (257, 8, 2) or min(inpt.shape[:2]) < 1:
            raise ValueError('Expected compressed STFT(B,T,257,8,2)')
        cache = state is not None
        if cache and (not isinstance(state, dict) or (state and set(state) != {'base', 'monaural_hidden'})):
            raise ValueError('Use a complete independently owned monaural stream state')
        memories = {} if state is None else state
        batch, frames = inpt.shape[:2]
        hidden = memories.get('monaural_hidden')
        if memories and (not isinstance(memories['base'], dict) or not isinstance(hidden, torch.Tensor)):
            raise ValueError('Use a complete independently owned monaural stream state')
        if hidden is not None and (hidden.shape != (1, batch, 128) or hidden.dtype != torch.float32
                                   or hidden.device != inpt.device):
            raise ValueError('Monaural hidden state differs from this stream')

        estimate, base_state = self.base.process(inpt, memories.get('base', {}) if cache else None)
        if estimate.shape != (batch, 2, frames, 257):
            raise ValueError('Causal source returned an incompatible spectrum')
        with torch.autocast(inpt.device.type, enabled=False):
            microphones = inpt.float()
            reference = microphones[..., 0, :]
            scale = torch.linalg.vector_norm(reference, dim=-1).mean(-1, keepdim=True).clamp_min(1e-4)
            beam = estimate.float().permute(0, 2, 3, 1)
            features = (torch.cat((reference, beam), -1) / scale[..., None]).permute(0, 3, 1, 2)
            # Keep cue gradients out of the existing beamforming path.
            unit_mics = microphones / torch.linalg.vector_norm(microphones, dim=-1, keepdim=True).clamp_min(1e-4)
            direction = beam.detach().unsqueeze(-2)
            direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(1e-4)
            real = unit_mics[..., 0] * direction[..., 0] + unit_mics[..., 1] * direction[..., 1]
            imag = unit_mics[..., 1] * direction[..., 0] - unit_mics[..., 0] * direction[..., 1]
            cues = torch.cat((real, imag), dim=-1).permute(0, 3, 1, 2)

        enabled = self.training and inpt.device.type == 'cuda'
        with torch.autocast(inpt.device.type, dtype=torch.bfloat16, enabled=enabled):
            first = self.encoders[0]
            value = first[2](first[1](first[0](features) + self.beam_cues(cues)))
            skips = [features, value]
            for encoder in self.encoders[1:]:
                value = encoder(value)
                skips.append(value)
        with torch.autocast(inpt.device.type, enabled=False):
            sequence = value.permute(0, 2, 1, 3).reshape(batch, frames, 17 * 48).float()
            sequence, hidden = self.recurrent(self.recurrent_norm(sequence), hidden)
        with torch.autocast(inpt.device.type, dtype=torch.bfloat16, enabled=enabled):
            value = value + self.projection(sequence).reshape(batch, frames, 48, 17).permute(0, 2, 1, 3)
            for up, decoder, skip in zip(self.upsamplers, self.decoders, reversed(skips[:-1])):
                value = up(value)
                if value.shape[-1] != skip.shape[-1]:
                    raise RuntimeError('Frequency decoder lost alignment')
                value = decoder(torch.cat((value, skip), 1))
            correction = self.output(value).float() * scale[:, None]
        updated = dict(base=base_state, monaural_hidden=hidden.detach().clone()) if cache else None
        return estimate.float() + correction, updated

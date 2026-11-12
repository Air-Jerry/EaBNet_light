"""Causal full-band spectral envelope adaptation after spatial enhancement."""
import torch
from torch import nn
from torch.nn import functional as F

from experiments.causal_recurrent_adapt_monaural import CausalRecurrentAdaptMonaural


class CausalEnvelopeMonaural(CausalRecurrentAdaptMonaural):
    def __init__(self, wide_source):
        super().__init__(wide_source)
        with torch.random.fork_rng(devices=[]):
            self.envelope = nn.ModuleDict({
                'encode': nn.Conv2d(9, 8, (1, 9), stride=(1, 8), padding=(0, 4)),
                'norm': nn.LayerNorm(8 * 33),
                'memory': nn.GRU(8 * 33, 96, batch_first=True),
                'project': nn.Linear(96, 8 * 33),
                'decode': nn.ConvTranspose2d(8, 1, (1, 9), stride=(1, 8), padding=(0, 4)),
            })
        self._reset_gain()

    def _reset_gain(self):
        nn.init.zeros_(self.envelope['decode'].weight)
        nn.init.zeros_(self.envelope['decode'].bias)

    def load_warm(self, weights):
        state = self.state_dict()
        new = {key for key in state if key.startswith('envelope.')}
        if set(weights) != set(state) - new:
            raise ValueError('Use all weights from the original recurrent model')
        self.load_state_dict({**weights, **{key: state[key] for key in new}})
        self._reset_gain()

    def process(self, inpt, state=None):
        streaming = state is not None
        if streaming and (not isinstance(state, dict) or (state and set(state) != {'speech', 'envelope_hidden'})):
            raise ValueError('Use a complete independently owned envelope stream state')
        memories = {} if state is None else state
        estimate, updated = super().process(inpt, memories.get('speech', {}) if streaming else None)
        batch, frames = inpt.shape[:2]
        hidden = memories.get('envelope_hidden')
        if hidden is not None and (hidden.shape != (1, batch, 96) or hidden.dtype != torch.float32
                                   or hidden.device != inpt.device):
            raise ValueError('Envelope history does not match the stream')

        enabled = self.training and inpt.device.type == 'cuda'
        with torch.autocast(inpt.device.type, enabled=False):
            microphones = torch.linalg.vector_norm(inpt.float(), dim=-1)
            enhanced = torch.linalg.vector_norm(estimate.detach(), dim=1)
            scale = microphones[..., 0].mean(-1, keepdim=True).clamp_min(1e-4)
            cues = torch.cat((microphones, enhanced.unsqueeze(-1)), dim=-1)
            cues = torch.log1p(cues / scale.unsqueeze(-1)).permute(0, 3, 1, 2)
        with torch.autocast(inpt.device.type, dtype=torch.bfloat16, enabled=enabled):
            encoded = F.silu(self.envelope['encode'](cues))
        with torch.autocast(inpt.device.type, enabled=False):
            sequence = encoded.permute(0, 2, 1, 3).reshape(batch, frames, 8 * 33).float()
            sequence, hidden = self.envelope['memory'](self.envelope['norm'](sequence), hidden)
        with torch.autocast(inpt.device.type, dtype=torch.bfloat16, enabled=enabled):
            contextual = self.envelope['project'](sequence).reshape(batch, frames, 8, 33).permute(0, 2, 1, 3)
            log_gain = self.envelope['decode'](encoded + contextual).float()
        gain = (.7 * log_gain.tanh()).exp()
        cache = dict(speech=updated, envelope_hidden=hidden.detach().clone()) if streaming else None
        return estimate * gain, cache

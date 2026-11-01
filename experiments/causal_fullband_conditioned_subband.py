"""Own causal full-band temporal conditioning for the frequency-wise subband GRU.

Current-frame spectra feed a shared temporal speech representation; the
frequency-wise GRU retains independent states. No future frames, external
gain correction or author-equivalence/quality claim is involved.
"""

import torch
from torch import nn
import torch.nn.functional as F

from experiments.causal_subband_gru import CausalSubbandGRU
from experiments.causal_fullband_tcn import CausalFullbandTCN
from light_causal_fast_model import ActivationLayerNorm


class CausalFullbandConditionedSubband(CausalSubbandGRU):
    def __init__(self, base_model, hidden=32, fullband_hidden=128, condition_channels=16):
        super().__init__(base_model, hidden)
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
               for v in (fullband_hidden, condition_channels)):
            raise ValueError('Use positive integer full-band dimensions')
        self.fullband_hidden = fullband_hidden
        self.condition_channels = condition_channels
        with torch.random.fork_rng(devices=[]):
            self.fullband_norm = ActivationLayerNorm(514)
            self.fullband_rnn = nn.GRU(514, fullband_hidden, batch_first=True, bidirectional=False)
            self.fullband_projection = nn.Linear(fullband_hidden, 257*condition_channels)
            self.fullband_mix = nn.Conv2d(condition_channels, hidden, 1)
            nn.init.zeros_(self.fullband_mix.weight)
            nn.init.zeros_(self.fullband_mix.bias)

    def process(self, inpt, state=None):
        if inpt.ndim != 5 or inpt.shape[-3:] != (257, 8, 2) or min(inpt.shape[:2]) < 1:
            raise ValueError('Expected compressed STFT(B,T,257,8,2)')
        cache = state is not None
        memories = {} if state is None else state
        estimate, base_state = self.base.process(inpt, memories.get('base', {}) if cache else None)
        batch, frames = inpt.shape[:2]
        with torch.autocast(inpt.device.type, enabled=False):
            spectra = inpt.float()
            magnitude = torch.linalg.vector_norm(spectra, dim=-1)
            scale = magnitude[..., 0].mean(-1, keepdim=True).unsqueeze(-1).clamp_min(1e-4)
            normalized = (spectra/scale.unsqueeze(-1)).flatten(-2)
            unit = spectra/magnitude.clamp_min(1e-4).unsqueeze(-1)
            real = unit[..., 1:, 0]*unit[..., :1, 0] + unit[..., 1:, 1]*unit[..., :1, 1]
            imaginary = unit[..., 1:, 1]*unit[..., :1, 0] - unit[..., 1:, 0]*unit[..., :1, 1]
            ratio = 2*torch.log((magnitude[..., 1:]+1e-4)/(magnitude[..., :1]+1e-4))
            ratio = 5*torch.tanh(ratio/5)
            envelope = torch.log1p(magnitude[..., :1]/scale)
            base_ri = estimate.float().permute(0, 2, 3, 1)
            features = torch.cat((normalized, real, imaginary, ratio, envelope, base_ri/scale), -1)
            features = self.feature_norm(features).permute(0, 3, 1, 2)
            # The normalization and full-band input use only this current frame.
            fullband = torch.cat((magnitude[..., 0], torch.linalg.vector_norm(base_ri, dim=-1)), -1)
            fullband = self.fullband_norm(torch.log1p(fullband/scale.squeeze(-1)))
            fullband_state = memories.get('fullband_hidden')
            if fullband_state is not None and (
                    fullband_state.shape != (1, batch, self.fullband_hidden)
                    or fullband_state.dtype != torch.float32 or fullband_state.device != inpt.device):
                raise ValueError('Full-band state differs from the input stream')
            fullband, fullband_state = self.fullband_rnn(fullband, fullband_state)
        enabled = self.training and self.use_bfloat16_training and inpt.device.type == 'cuda'
        with torch.autocast(inpt.device.type, dtype=torch.bfloat16, enabled=enabled):
            condition = self.fullband_projection(fullband).reshape(batch, frames, 257, self.condition_channels)
            condition = self.fullband_mix(condition.permute(0, 3, 1, 2))
            features = F.silu(self.input(features)+condition)
        with torch.autocast(inpt.device.type, enabled=False):
            value = features.permute(0, 3, 2, 1).reshape(batch*257, frames, self.hidden).float()
            hidden = memories.get('subband')
            if hidden is not None and (hidden.shape != (1, batch*257, self.hidden)
                                       or hidden.dtype != torch.float32 or hidden.device != inpt.device):
                raise ValueError('Subband GRU state differs from the input stream')
            value, hidden = self.recurrent(self.recurrent_norm(value), hidden)
            value = value.reshape(batch, 257, frames, self.hidden).permute(0, 3, 2, 1)
        with torch.autocast(inpt.device.type, dtype=torch.bfloat16, enabled=enabled):
            extra = self.output(value*self.hidden**-.5).float().permute(0, 2, 3, 1)
        with torch.autocast(inpt.device.type, enabled=False):
            weights = extra[..., :16].reshape(batch, frames, 257, 8, 2)
            raw = CausalFullbandTCN.decompress(spectra)
            real = (weights[..., 0]*raw[..., 0] - weights[..., 1]*raw[..., 1]).sum(-1)
            imaginary = (weights[..., 0]*raw[..., 1] + weights[..., 1]*raw[..., 0]).sum(-1)
            base_raw = CausalFullbandTCN.decompress(base_ri)
            correction = (CausalFullbandTCN.compress(base_raw+torch.stack((real, imaginary), -1))
                          - CausalFullbandTCN.compress(base_raw))
            output = base_ri+correction+extra[..., 16:]*scale
        updated = dict(base=base_state, subband=hidden.detach().clone(),
                       fullband_hidden=fullband_state.detach().clone()) if cache else None
        return output.permute(0, 3, 1, 2), updated

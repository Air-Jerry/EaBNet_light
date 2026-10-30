"""Full-resolution recurrent spatial beamforming trial; not a production candidate.

The entire existing causal global model remains trainable. Each frequency's
unidirectional recurrent state sees all eight microphones and the base estimate.
"""

import torch
from torch import nn
import torch.nn.functional as F

from light_causal_global_model import CausalGlobalSpatial
from light_causal_spatial_model import CausalSpatialEnhancer


class CausalBandBeamformer(CausalGlobalSpatial):
    def __init__(self, M=8, band_hidden=64, band_layers=2, **kwargs):
        super().__init__(M=M, **kwargs)
        with torch.random.fork_rng(devices=[]):
            self.band_input = nn.Linear(2 * M + 2, band_hidden)
            self.band_rnn = nn.GRU(band_hidden, band_hidden, num_layers=band_layers,
                batch_first=True, bidirectional=False)
            self.band_output = nn.Linear(band_hidden, 2 * M)
            nn.init.zeros_(self.band_output.weight)
            nn.init.zeros_(self.band_output.bias)

    def process(self, inpt, state=None):
        output, updated = super().process(inpt, state)
        batch, frames, frequencies = inpt.shape[:3]
        with torch.autocast(inpt.device.type, enabled=False):
            mixture = CausalSpatialEnhancer.decompress(inpt.float())
            estimate_ri = output.permute(0, 2, 3, 1)
            estimated = CausalSpatialEnhancer.decompress(estimate_ri)
            reference = mixture[..., 0, :]
            scale = torch.linalg.vector_norm(reference, dim=-1).mean(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(1e-4)
            features = torch.cat((mixture.flatten(-2), estimated), dim=-1) / scale
            features = features.transpose(1, 2).reshape(batch * frequencies, frames, -1)
            temporal = F.silu(self.band_input(features))
            temporal, hidden = self.band_rnn(temporal, None if state is None else state.get('band_hidden'))
            weights = self.band_output(temporal).reshape(batch, frequencies, frames, self.M, 2).transpose(1, 2)
            real = (weights[..., 0] * mixture[..., 0] - weights[..., 1] * mixture[..., 1]).sum(-1)
            imag = (weights[..., 0] * mixture[..., 1] + weights[..., 1] * mixture[..., 0]).sum(-1)
            refined = estimated + torch.stack((real, imag), dim=-1)
            correction = CausalSpatialEnhancer.compress(refined) - CausalSpatialEnhancer.compress(estimated)
            output = output + correction.permute(0, 3, 1, 2)
        if state is not None:
            updated['band_hidden'] = hidden.detach().clone()
        return output, updated

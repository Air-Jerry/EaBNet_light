"""Screen larger causal global temporal capacity without freezing the source model."""

import torch

from light_causal_global_model import GlobalTemporalLayer
from light_causal_recurrent65_model import CausalRecurrentUNet65


class CausalGlobalCapacity(CausalRecurrentUNet65):
    def __init__(self, M=8, expanded_hidden=512, **kwargs):
        super().__init__(M=M, **kwargs)
        with torch.random.fork_rng(devices=[]):
            self.layers.append(GlobalTemporalLayer(65, self.encoders[-1].conv.out_channels, expanded_hidden))

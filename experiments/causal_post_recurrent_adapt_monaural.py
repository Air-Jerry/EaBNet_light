"""Adapt the existing penultimate temporal FFN without earlier GRU backward passes."""
from experiments.causal_recurrent_adapt_monaural import CausalRecurrentAdaptMonaural
from light_causal_recurrent_model import RecurrentSpatialLayer


class CausalPostRecurrentAdaptMonaural(CausalRecurrentAdaptMonaural):
    def __init__(self, wide_source):
        super().__init__(wide_source)
        layers = self.base.base.layers[:-1]
        if len(layers) != 4 or not isinstance(layers[-2], RecurrentSpatialLayer):
            raise ValueError('Use the original four frequency-group recurrent blocks')
        layers[-2].tconvffn.requires_grad_(True)

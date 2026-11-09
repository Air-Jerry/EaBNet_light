"""Adapt the final frequency-group recurrent block while earlier shared modules stay frozen."""
from experiments.causal_global_adapt_monaural import CausalGlobalAdaptMonaural
from light_causal_recurrent_model import RecurrentSpatialLayer


class CausalRecurrentAdaptMonaural(CausalGlobalAdaptMonaural):
    def __init__(self, wide_source):
        super().__init__(wide_source)
        layers = self.base.base.layers[:-1]
        if len(layers) != 4 or not isinstance(layers[-1], RecurrentSpatialLayer):
            raise ValueError('Use the original four frequency-group recurrent blocks')
        layers[-1].requires_grad_(True)
        # The frequency matrix is shared; freezing earlier blocks also keeps that matrix frozen.
        for layer in layers[:-1]:
            layer.requires_grad_(False)

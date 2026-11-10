"""Unregistered adaptation of the last two existing frequency recurrent blocks."""
from experiments.causal_recurrent_adapt_monaural import CausalRecurrentAdaptMonaural


class CausalTwoBlockAdaptMonaural(CausalRecurrentAdaptMonaural):
    def __init__(self,wide_source):
        super().__init__(wide_source)
        layers=self.base.base.layers[:-1]
        layers[-2].requires_grad_(True)
        # Earlier blocks own the shared matrix; keep it and those blocks frozen.
        for layer in layers[:-2]:layer.requires_grad_(False)

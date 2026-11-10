"""Unregistered last-block frequency adaptation with unchanged causal inference operators."""
import copy

from experiments.causal_recurrent_adapt_monaural import CausalRecurrentAdaptMonaural


class CausalIndependentFrequencyMonaural(CausalRecurrentAdaptMonaural):
    def __init__(self, wide_source):
        super().__init__(wide_source)
        layers = self.base.base.layers[:-1]
        shared = layers[-1].full
        if not all(layer.full is shared for layer in layers[:-1]):
            raise ValueError('Use the original shared frequency matrices')
        # Own the final matrix so its gradients cannot change the frozen earlier blocks.
        layers[-1].full = copy.deepcopy(shared).requires_grad_(True)

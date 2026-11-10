"""Unregistered joint spatial-backbone adaptation; GPU cost and quality require real qualification."""
from experiments.causal_recurrent_adapt_monaural import CausalRecurrentAdaptMonaural


class CausalUnfrozenRootMonaural(CausalRecurrentAdaptMonaural):
    def __init__(self, wide_source):
        super().__init__(wide_source)
        self.base.base.requires_grad_(True)
        self.train(self.training)

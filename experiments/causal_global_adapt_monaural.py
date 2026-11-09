"""Adapt the existing full-band temporal layer while frequency-group recurrence stays frozen."""
from experiments.causal_mapping_monaural import CausalMappingMonaural
from light_causal_global_model import GlobalTemporalLayer


class CausalGlobalAdaptMonaural(CausalMappingMonaural):
    def __init__(self, wide_source):
        super().__init__(wide_source)
        global_layer = self.base.base.layers[-1]
        if not isinstance(global_layer, GlobalTemporalLayer):
            raise ValueError('Use the existing final full-band temporal layer')
        global_layer.requires_grad_(True)

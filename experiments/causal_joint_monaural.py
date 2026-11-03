"""Unregistered end-to-end variant of the existing trained monaural/spatial core.

Inference operators, parameter keys and external states remain identical.
Real-data quality and the complete training budget require new measurement.
"""
from experiments.causal_monaural_spectral_branch import CausalMonauralSpectralBranch
from experiments.causal_wide_conditioned_subband import CausalWideConditionedSubband


class CausalJointMonaural(CausalMonauralSpectralBranch):
    def __init__(self,wide_source):
        if not isinstance(wide_source,CausalWideConditionedSubband):
            raise ValueError('Use the original wide-conditioned causal source')
        super().__init__(wide_source)
        self.requires_grad_(True)
        self.train(self.training)

    def train(self,mode=True):
        super().train(mode)
        # The inherited frozen-source policy must also be lifted for RNN backward.
        self.base.base.train(mode)
        return self

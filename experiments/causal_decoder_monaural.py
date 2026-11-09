"""Unregistered decoder adaptation of the causal monaural/spatial model."""
from experiments.causal_monaural_spectral_branch import CausalMonauralSpectralBranch
from experiments.causal_wide_conditioned_subband import CausalWideConditionedSubband


class CausalDecoderMonaural(CausalMonauralSpectralBranch):
    def __init__(self, wide_source):
        if not isinstance(wide_source, CausalWideConditionedSubband):
            raise ValueError('Use the original wide-conditioned causal source')
        super().__init__(wide_source)
        for module in (self.base.base.upsamplers, self.base.base.decoders, self.base.base.beamformer):
            module.requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        # Match the joint model's training precision; frozen encoder weights stay fixed.
        self.base.base.train(mode)
        return self

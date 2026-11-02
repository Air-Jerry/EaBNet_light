"""Experimental end-to-end training of the existing causal conditioned model.

Inference operators and state layout are inherited unchanged. Quality, full6s
CUDA memory and the complete one-hour training budget are not yet verified.
"""

from experiments.causal_spatial_conditioned_subband import CausalSpatialConditionedSubband


class CausalJointConditionedSubband(CausalSpatialConditionedSubband):
    def __init__(self, base_model, **kwargs):
        super().__init__(base_model, **kwargs)
        self.base.requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        # CuDNN recurrent backward requires the newly trainable base in train mode.
        self.base.train(mode)
        return self

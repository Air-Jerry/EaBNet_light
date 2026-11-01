"""Train the causal conditioned head against a fixed causal speech backbone."""

from experiments.causal_fullband_conditioned_subband import CausalFullbandConditionedSubband


class CausalFrozenBaseConditionedSubband(CausalFullbandConditionedSubband):
    def __init__(self, base_model, **kwargs):
        if getattr(base_model, 'is_causal', None) is not True:
            raise ValueError('A fixed backbone must itself be causal')
        super().__init__(base_model, **kwargs)
        self.base.requires_grad_(False)
        self.base.zero_grad(set_to_none=True)
        self.base.eval()

    def train(self, mode=True):
        super().train(mode)
        # Keep backbone predictions in the same FP32 mode used for inference.
        self.base.eval()
        return self

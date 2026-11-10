"""Jointly adapt existing spatial encoders without adding inference operations.

Earlier recurrent kernels and the shared frequency matrix stay frozen. Real
CUDA memory, complete-epoch cost and full600 quality require qualification.
"""
from experiments.causal_recurrent_adapt_monaural import CausalRecurrentAdaptMonaural


class CausalEncoderAdaptMonaural(CausalRecurrentAdaptMonaural):
    def __init__(self, wide_source):
        super().__init__(wide_source)
        self.base.base.stem.requires_grad_(True)
        self.base.base.encoders.requires_grad_(True)


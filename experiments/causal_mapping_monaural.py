"""Adapt the existing causal frequency/memory mapping without new inference operators."""
from experiments.causal_decoder_monaural import CausalDecoderMonaural


class CausalMappingMonaural(CausalDecoderMonaural):
    mapping_modules = ('mapping_input', 'retention', 'mapping_blocks', 'mapping_output')

    def __init__(self, wide_source):
        super().__init__(wide_source)
        for name in self.mapping_modules:
            getattr(self.base.base, name).requires_grad_(True)

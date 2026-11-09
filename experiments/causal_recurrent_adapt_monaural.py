"""Adapt the final frequency-group recurrent block while earlier shared modules stay frozen."""
from experiments.causal_global_adapt_monaural import CausalGlobalAdaptMonaural
from light_causal_recurrent_model import RecurrentSpatialLayer


def recurrent_root_editable(name):
    prefixes = ('base.base.decoders.', 'base.base.upsamplers.', 'base.base.beamformer.',
                'base.base.mapping_input.', 'base.base.retention.', 'base.base.mapping_blocks.',
                'base.base.mapping_output.', 'base.base.layers.3.', 'base.base.layers.4.')
    return name.startswith(prefixes) and not name.startswith('base.base.layers.3.full.')


class CausalRecurrentAdaptMonaural(CausalGlobalAdaptMonaural):
    def __init__(self, wide_source):
        super().__init__(wide_source)
        layers = self.base.base.layers[:-1]
        if len(layers) != 4 or not isinstance(layers[-1], RecurrentSpatialLayer):
            raise ValueError('Use the original four frequency-group recurrent blocks')
        layers[-1].requires_grad_(True)
        # The frequency matrix is shared; freezing earlier blocks also keeps that matrix frozen.
        for layer in layers[:-1]:
            layer.requires_grad_(False)

"""Train the spatial encoder through the residual paths of frozen GRUs."""
from experiments.causal_encoder_adapt_monaural import CausalEncoderAdaptMonaural


def detach_frozen_gru_input(module, inputs):
    if module.training:
        return (inputs[0].detach(), *inputs[1:])
    return None


class CausalEncoderResidualAdapt(CausalEncoderAdaptMonaural):
    def __init__(self, wide_source):
        super().__init__(wide_source)
        for layer in self.base.base.layers[:3]:
            if any(parameter.requires_grad for parameter in layer.mhsa.parameters()):
                raise ValueError('Residual adaptation requires frozen early GRUs')
            layer.mhsa.register_forward_pre_hook(detach_frozen_gru_input)

    def load_warm(self, weights):
        self.load_state_dict(weights, strict=True)

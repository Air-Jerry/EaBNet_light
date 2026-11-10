"""Joint past-frame attention with the existing recurrent model's trainable scope."""
import torch

from experiments.causal_recurrent_adapt_monaural import recurrent_root_editable
from experiments.causal_temporal_monaural import CausalTemporalMonaural, PastFrameAttention


class TrainingPrecisionPastFrameAttention(PastFrameAttention):
    def process(self, value, state=None):
        enabled = self.training and value.device.type == 'cuda'
        with torch.autocast(value.device.type, dtype=torch.bfloat16, enabled=enabled):
            return super().process(value, state)


class CausalJointPastAttentionMonaural(CausalTemporalMonaural):
    def __init__(self, wide_source):
        super().__init__(wide_source)
        with torch.random.fork_rng(devices=[]):
            memory = TrainingPrecisionPastFrameAttention()
        memory.load_state_dict(self.temporal_memory.state_dict(), strict=True)
        self.temporal_memory = memory
        for name, parameter in self.named_parameters():
            if name.startswith('base.base.'):
                parameter.requires_grad_(recurrent_root_editable(name))

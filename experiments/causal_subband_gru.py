"""Own causal spatial refiner with an independent GRU state per frequency.

FullSubNet (arxiv.org/abs/2010.15508) motivates frequency-wise temporal
modeling. This is an eight-microphone, zero-lookahead residual hypothesis,
not its implementation or a quality claim. No production model is changed.
"""

import torch
from torch import nn
import torch.nn.functional as F

from experiments.causal_fullband_tcn import CausalFullbandTCN
from light_causal_fast_model import ActivationLayerNorm


class CausalSubbandGRU(nn.Module):
    is_causal = True
    refiner_causal = True

    def __init__(self, base_model, hidden=32):
        super().__init__()
        if (getattr(base_model, 'M', None) != 8 or not getattr(base_model, 'is_causal', False)
                or not callable(getattr(base_model, 'process', None)) or hidden < 1):
            raise ValueError('Use a causal eight-microphone base with process() and positive hidden size')
        self.base = base_model
        self.M, self.context, self.hidden = 8, base_model.context, hidden
        self.use_bfloat16_training = True
        # Added parameters must not change the following data-crop random sequence.
        with torch.random.fork_rng(devices=[]):
            self.feature_norm = ActivationLayerNorm(40)
            self.input = nn.Conv2d(40, hidden, (1, 3), padding=(0, 1))
            self.recurrent_norm = ActivationLayerNorm(hidden)
            self.recurrent = nn.GRU(hidden, hidden, batch_first=True, bidirectional=False)
            self.output = nn.Conv2d(hidden, 18, 1)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def process(self, inpt, state=None):
        if inpt.ndim != 5 or inpt.shape[-3:] != (257, 8, 2) or min(inpt.shape[:2]) < 1:
            raise ValueError('Expected compressed STFT(B,T,257,8,2)')
        cache = state is not None
        memories = {} if state is None else state
        estimate, base_state = self.base.process(inpt, memories.get('base', {}) if cache else None)
        batch, frames = inpt.shape[:2]
        with torch.autocast(inpt.device.type, enabled=False):
            spectra = inpt.float()
            magnitude = torch.linalg.vector_norm(spectra, dim=-1)
            scale = magnitude[..., 0].mean(-1, keepdim=True).unsqueeze(-1).clamp_min(1e-4)
            normalized = (spectra / scale.unsqueeze(-1)).flatten(-2)
            unit = spectra / magnitude.clamp_min(1e-4).unsqueeze(-1)
            real = unit[..., 1:, 0]*unit[..., :1, 0] + unit[..., 1:, 1]*unit[..., :1, 1]
            imaginary = unit[..., 1:, 1]*unit[..., :1, 0] - unit[..., 1:, 0]*unit[..., :1, 1]
            ratio = 2*torch.log((magnitude[..., 1:] + 1e-4)/(magnitude[..., :1] + 1e-4))
            ratio = 5*torch.tanh(ratio/5)
            envelope = torch.log1p(magnitude[..., :1]/scale)
            base_ri = estimate.float().permute(0, 2, 3, 1)
            features = torch.cat((normalized, real, imaginary, ratio, envelope, base_ri/scale), -1)
            features = self.feature_norm(features).permute(0, 3, 1, 2)
        enabled = self.training and self.use_bfloat16_training and inpt.device.type == 'cuda'
        with torch.autocast(inpt.device.type, dtype=torch.bfloat16, enabled=enabled):
            features = F.silu(self.input(features))
        with torch.autocast(inpt.device.type, enabled=False):
            value = features.permute(0, 3, 2, 1).reshape(batch*257, frames, self.hidden).float()
            hidden = memories.get('subband')
            if hidden is not None and (hidden.shape != (1, batch*257, self.hidden)
                                       or hidden.dtype != torch.float32 or hidden.device != inpt.device):
                raise ValueError('Subband GRU state differs from the input stream')
            # CuDNN GRU stays FP32; BF16 recurrent execution was slower on the target GPUs.
            value, hidden = self.recurrent(self.recurrent_norm(value), hidden)
            value = value.reshape(batch, 257, frames, self.hidden).permute(0, 3, 2, 1)
        with torch.autocast(inpt.device.type, dtype=torch.bfloat16, enabled=enabled):
            extra = self.output(value*self.hidden**-.5).float().permute(0, 2, 3, 1)
        with torch.autocast(inpt.device.type, enabled=False):
            weights = extra[..., :16].reshape(batch, frames, 257, 8, 2)
            raw = CausalFullbandTCN.decompress(spectra)
            real = (weights[..., 0]*raw[..., 0] - weights[..., 1]*raw[..., 1]).sum(-1)
            imaginary = (weights[..., 0]*raw[..., 1] + weights[..., 1]*raw[..., 0]).sum(-1)
            base_raw = CausalFullbandTCN.decompress(base_ri)
            correction = (CausalFullbandTCN.compress(base_raw + torch.stack((real, imaginary), -1))
                          - CausalFullbandTCN.compress(base_raw))
            output = base_ri + correction + extra[..., 16:]*scale
        updated = dict(base=base_state, subband=hidden.detach().clone()) if cache else None
        return output.permute(0, 3, 1, 2), updated

    def forward(self, inpt, frame_lengths=None):
        if frame_lengths is not None:
            lengths = torch.as_tensor(frame_lengths)
            if lengths.numel() != inpt.shape[0] or torch.any(lengths < 1) or torch.any(lengths > inpt.shape[1]):
                raise ValueError('One valid frame length is required per sample')
        return self.process(inpt)[0]

    def forward_stream(self, inpt, state=None):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError('Streaming requires eval mode and disabled gradients')
        return self.process(inpt, {} if state is None else state)

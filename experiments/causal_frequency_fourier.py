"""Frame-local frequency Fourier mixing for the causal TCN input."""

import torch
from torch import nn
import torch.nn.functional as F

from experiments.causal_tcn_mapping import CausalTCNMapping
from light_causal_fast_model import FrameChannelNorm


class FrequencyFourierInput(nn.Conv2d):
    def __init__(self,incoming,outgoing):
        super().__init__(incoming,outgoing,1)
        self.fourier_in=nn.Conv2d(2*outgoing,2*outgoing,1)
        self.fourier_norm=FrameChannelNorm(2*outgoing)
        self.fourier_out=nn.Conv2d(2*outgoing,2*outgoing,1)
        nn.init.zeros_(self.fourier_out.weight)
        nn.init.zeros_(self.fourier_out.bias)

    def forward(self,features):
        if features.shape[-1]!=257:raise ValueError('Expected257 audio STFT frequency bins')
        value=super().forward(features)
        # Transform latent frequency only; the audio frontend and time axis stay unchanged.
        with torch.autocast(value.device.type,enabled=False):
            spectrum=torch.fft.rfft(value.float(),n=512,dim=-1,norm='ortho')
            pair=torch.cat((spectrum.real,spectrum.imag),dim=1).contiguous(memory_format=torch.channels_last)
        hidden=F.silu(self.fourier_norm(self.fourier_in(pair)))
        mixed=self.fourier_out(hidden)
        with torch.autocast(value.device.type,enabled=False):
            real,imag=mixed.float().chunk(2,dim=1)
            correction=torch.fft.irfft(torch.complex(real,imag),n=512,dim=-1,norm='ortho')[...,:257]
        return value+correction.to(value.dtype).contiguous(memory_format=torch.channels_last)


class CausalFrequencyFourierTCN(CausalTCNMapping):
    def __init__(self,M=8,**kwargs):
        super().__init__(M=M,**kwargs)
        original=self.mapping_input
        with torch.random.fork_rng(devices=[]):
            changed=FrequencyFourierInput(original.in_channels,original.out_channels)
        missing,unexpected=changed.load_state_dict(original.state_dict(),strict=False)
        assert len(missing)==6 and all(name.startswith('fourier_') for name in missing) and not unexpected
        self.mapping_input=changed

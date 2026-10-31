"""Learn current-frame inter-microphone phase and magnitude cues in the network."""

import torch
from torch import nn

from light_causal_fast_model import CausalFrequencyConv
from light_causal_recurrent65_model import CausalRecurrentUNet65


def spatial_cues(value):
    """Input (B,16,T,F) uses mic-major real/imag; output has21 spatial channels."""
    if value.ndim!=4 or value.shape[1]!=16:
        raise ValueError('Spatial cues require eight microphones in original real/imag order')
    with torch.autocast(value.device.type,enabled=False):
        pairs=value.float().reshape(value.shape[0],8,2,*value.shape[2:])
        real,imag=pairs.unbind(2)
        magnitude=torch.linalg.vector_norm(pairs,dim=2).clamp_min(1e-8)
        denominator=(magnitude[:,1:]*magnitude[:,:1]).clamp_min(1e-8)
        cosine=(real[:,1:]*real[:,:1]+imag[:,1:]*imag[:,:1])/denominator
        sine=(imag[:,1:]*real[:,:1]-real[:,1:]*imag[:,:1])/denominator
        log_ratio=(magnitude[:,1:].log()-magnitude[:,:1].log()).clamp(-4,4)
        return torch.cat((cosine,sine,log_ratio),dim=1)


class SpatialCueStem(CausalFrequencyConv):
    def __init__(self):
        super().__init__(16,32,frequency_kernel=5)
        self.spatial_projection=nn.Conv2d(21,32,1)
        nn.init.zeros_(self.spatial_projection.weight)
        nn.init.zeros_(self.spatial_projection.bias)

    def process(self,value,history=None,cache=False):
        output,updated=super().process(value,history,cache)
        return output+self.spatial_projection(spatial_cues(value)),updated


class CausalSpatialCueUNet65(CausalRecurrentUNet65):
    def __init__(self,M=8,**kwargs):
        super().__init__(M=M,**kwargs)
        original=self.stem
        with torch.random.fork_rng(devices=[]):
            changed=SpatialCueStem()
        missing,unexpected=changed.load_state_dict(original.state_dict(),strict=False)
        assert set(missing)=={'spatial_projection.weight','spatial_projection.bias'} and not unexpected
        self.stem=changed

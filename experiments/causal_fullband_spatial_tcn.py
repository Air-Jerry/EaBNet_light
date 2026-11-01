"""Experimental full-band TCN with a direct causal path for each frequency's spatial cues.

The extra output is zero initially, preserving a complete full-band checkpoint.
This is an own architecture hypothesis, without quality or author-equivalence claims.
The immutable source model is left unchanged while its full-data run continues.
"""

import torch
from torch import nn
import torch.nn.functional as F

from experiments.causal_fullband_tcn import CausalFullbandTCN
from light_causal_fast_model import ActivationLayerNorm, ActivationPReLU


class SpatialTemporalBlock(nn.Module):
    def __init__(self, channels, hidden, dilation):
        super().__init__()
        self.history_frames = 2*dilation
        self.norm = ActivationLayerNorm(channels,seq_last=True)
        self.input = nn.Conv2d(channels,2*hidden,1)
        self.temporal = nn.Conv2d(hidden,hidden,(3,3),dilation=(dilation,1),padding=(0,1),groups=hidden)
        self.activation = ActivationPReLU(hidden)
        self.output = nn.Conv2d(hidden,channels,1)

    def process(self, value, history=None, cache=False):
        left,gate = self.input(self.norm(value)).chunk(2,dim=1)
        current = F.silu(left)*torch.sigmoid(gate)
        shape = (value.shape[0],current.shape[1],self.history_frames,value.shape[-1])
        if history is None:
            joined = F.pad(current,(0,0,self.history_frames,0))
        else:
            if history.shape != shape:
                raise ValueError('Spatial TCN history differs from the input stream')
            joined = torch.cat((history,current),dim=2)
        result = value+self.output(self.activation(self.temporal(joined)))
        updated = joined[:,:,-self.history_frames:].detach().clone() if cache else None
        return result,updated


class CausalFullbandSpatialTCN(CausalFullbandTCN):
    def __init__(self, spatial_channels=64, spatial_hidden=96, spatial_dilations=(1,2,4,8), **kwargs):
        super().__init__(**kwargs)
        if min(spatial_channels,spatial_hidden)<1 or not spatial_dilations or min(spatial_dilations)<1:
            raise ValueError('Use positive spatial dimensions and dilations')
        self.spatial_input = nn.Conv2d(38,spatial_channels,1)
        self.spatial_layers = nn.ModuleList([SpatialTemporalBlock(spatial_channels,spatial_hidden,d)
                                            for d in spatial_dilations])
        self.spatial_norm = ActivationLayerNorm(spatial_channels,seq_last=True)
        self.spatial_condition = nn.Conv1d(self.encoder.out_features,2*spatial_channels,1)
        self.spatial_readout = nn.Conv2d(spatial_channels,18,1)
        self.spatial_readout_scale = spatial_channels**-.5
        nn.init.zeros_(self.spatial_readout.weight)
        nn.init.zeros_(self.spatial_readout.bias)
        self.context = max(self.context,sum(layer.history_frames for layer in self.spatial_layers))

    def process(self, inpt, state=None):
        if inpt.ndim!=5 or inpt.shape[-3:]!=(257,8,2) or min(inpt.shape[:2])<1:
            raise ValueError('Expected compressed STFT(B,T,257,8,2)')
        cache = state is not None
        with torch.autocast(inpt.device.type,enabled=False):
            spectra = inpt.float()
            magnitude = torch.linalg.vector_norm(spectra,dim=-1)
            scale = magnitude[...,0].mean(-1,keepdim=True).unsqueeze(-1).clamp_min(1e-4)
            normalized = (spectra/scale.unsqueeze(-1)).flatten(-2)
            unit = spectra/magnitude.clamp_min(1e-4).unsqueeze(-1)
            real = unit[...,1:,0]*unit[...,:1,0]+unit[...,1:,1]*unit[...,:1,1]
            imaginary = unit[...,1:,1]*unit[...,:1,0]-unit[...,1:,0]*unit[...,:1,1]
            ratio = 2*torch.log((magnitude[...,1:]+1e-4)/(magnitude[...,:1]+1e-4))
            ratio = 5*torch.tanh(ratio/5)
            envelope = torch.log1p(magnitude[...,:1]/scale)
            spatial_features = torch.cat((normalized,real,imaginary,ratio,envelope),dim=-1)
            features = spatial_features.flatten(-2)
        enabled = self.training and self.use_bfloat16_training and inpt.device.type=='cuda'
        with torch.autocast(inpt.device.type,dtype=torch.bfloat16,enabled=enabled):
            value = self.encoder(self.input_norm(features)).transpose(1,2)
            value = self.encoder_activation(value)
            histories = []
            for index,layer in enumerate(self.layers):
                history = None if state is None else state.get('layers',[None]*len(self.layers))[index]
                value,updated = layer.process(value,history,cache)
                if cache:histories.append(updated)
            value = self.output_norm(value)*self.readout_scale
            beam = self.beamformer(value).float().transpose(1,2).reshape(*inpt.shape[:2],257,8,2)
            mapping = self.mapping(value).float().transpose(1,2).reshape(*inpt.shape[:2],257,2)
            local = self.spatial_input(spatial_features.permute(0,3,1,2))
            spatial_histories = []
            for index,layer in enumerate(self.spatial_layers):
                history = None if state is None else state.get('spatial_layers',[None]*len(self.spatial_layers))[index]
                local,updated = layer.process(local,history,cache)
                if cache:spatial_histories.append(updated)
            gamma,beta = self.spatial_condition(value/self.readout_scale).chunk(2,dim=1)
            local = self.spatial_norm(local)*(1+torch.tanh(gamma.unsqueeze(-1)))+beta.unsqueeze(-1)
            extra = self.spatial_readout(local*self.spatial_readout_scale).float().permute(0,2,3,1)
            spatial_beam,spatial_mapping = extra[...,:16].reshape(*inpt.shape[:2],257,8,2),extra[...,16:]
        with torch.autocast(inpt.device.type,enabled=False):
            raw = self.decompress(spectra)
            reference = raw[...,0,:]
            real = (beam[...,0]*raw[...,0]-beam[...,1]*raw[...,1]).sum(-1)
            imaginary = (beam[...,0]*raw[...,1]+beam[...,1]*raw[...,0]).sum(-1)
            correction = self.compress(reference+torch.stack((real,imaginary),-1))-self.compress(reference)
            output = spectra[...,0,:]+correction+mapping*scale
            output_raw = self.decompress(output)
            real = (spatial_beam[...,0]*raw[...,0]-spatial_beam[...,1]*raw[...,1]).sum(-1)
            imaginary = (spatial_beam[...,0]*raw[...,1]+spatial_beam[...,1]*raw[...,0]).sum(-1)
            correction = self.compress(output_raw+torch.stack((real,imaginary),-1))-self.compress(output_raw)
            output = output+correction+spatial_mapping*scale
        updated = dict(layers=histories,spatial_layers=spatial_histories) if cache else None
        return output.permute(0,3,1,2),updated

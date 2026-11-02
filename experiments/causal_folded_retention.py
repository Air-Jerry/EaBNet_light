"""Lossless frequency packing before a complete causal spatial/retention core.

Four adjacent bins enter the channel axis together, rather than discarding bins
or pooling microphone spectra. Every original bin is decoded independently.
Frequency modules retain the repository's MIT SpatialNet attribution; this is
an independent causal adaptation, not an author-confirmed implementation.
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from experiments.causal_fullband_retention import FullbandRetentionLayer
from experiments.causal_retention_mapping import CausalRetention
from light_causal_fast_model import ActivationPReLU, activation_norms
from light_causal_spatial_model import CausalSpatialEnhancer, CausalSpatialLayer, left_convolution


class PackedRetentionLayer(FullbandRetentionLayer):
    def __init__(self, hidden, ffn, heads, context, frequencies, full=None):
        CausalSpatialLayer.__init__(self, hidden, ffn, frequencies, heads, context, full=full)
        self.mhsa=CausalRetention(hidden, heads=heads)
        self.mhsa.norm=nn.Identity()
        nn.init.xavier_uniform_(self.mhsa.output.weight, gain=.5)
        self.fconv1[-1]=ActivationPReLU(hidden)
        self.fconv2[-1]=ActivationPReLU(hidden)
        activation_norms(self)


class CausalFoldedRetention(CausalSpatialEnhancer):
    def __init__(self, M=8, *, fold=4, hidden=96, ffn=192, layers=8,
                 heads=4, context=64, **kwargs):
        nn.Module.__init__(self)
        if (isinstance(fold, bool) or fold not in (1, 2, 4) or M != 8
                or kwargs.get('is_causal', True) is False or layers < 1
                or hidden < 8 or ffn < 8 or hidden % 8 or ffn % 8
                or heads < 1 or hidden % heads or context < 2):
            raise ValueError('Use fold1/2/4, eight microphones and compatible causal dimensions')
        self.M,self.context,self.fold=8,context,fold
        self.groups=(257+fold-1)//fold
        self.encoder=nn.Conv1d(16*fold,hidden,5)
        self.layers=nn.ModuleList()
        full=None
        for _ in range(layers):
            layer=PackedRetentionLayer(hidden,ffn,heads,context,self.groups,full)
            self.layers.append(layer);full=layer.full
        self.decoder=nn.Linear(hidden,2*fold)
        nn.init.zeros_(self.decoder.weight);nn.init.zeros_(self.decoder.bias)
        self.gradient_checkpointing=False
        self.use_bfloat16_training=True

    def pack(self, value):
        if value.ndim != 4 or value.shape[2] != 257:
            raise ValueError('Expected (B,T,257,C) spectra')
        batch,frames,_,channels=value.shape
        value=torch.nn.functional.pad(value,(0,0,0,self.groups*self.fold-257))
        return value.reshape(batch,frames,self.groups,self.fold*channels).permute(0,2,1,3)

    def unpack(self, value):
        if value.ndim != 4 or value.shape[1] != self.groups or value.shape[-1] % self.fold:
            raise ValueError('Packed frequency dimensions differ')
        batch,_,frames,width=value.shape
        return value.permute(0,2,1,3).reshape(batch,frames,self.groups*self.fold,width//self.fold)[:,:,:257]

    def process(self, inpt, state=None):
        if inpt.ndim != 5 or inpt.shape[-3:] != (257,8,2) or min(inpt.shape[:2]) < 1:
            raise ValueError('Expected nonempty compressed STFT(B,T,257,8,2)')
        memories={} if state is None else state
        cache=state is not None
        with torch.autocast(inpt.device.type,enabled=False):
            mixture=self.decompress(inpt.float());reference=mixture[...,0,:]
            scale=torch.linalg.vector_norm(reference,dim=-1).mean(-1,keepdim=True).unsqueeze(-1).clamp_min(1e-4)
            features=self.pack(mixture.flatten(-2)/scale)
        batch,frequencies,frames,width=features.shape
        enabled=self.training and self.use_bfloat16_training and inpt.device.type=='cuda'
        with torch.autocast(inpt.device.type,dtype=torch.bfloat16,enabled=enabled):
            encoded=features.reshape(batch*frequencies,frames,width).transpose(1,2)
            encoded,history=left_convolution(self.encoder,encoded,memories.get('encoder'))
            value=encoded.transpose(1,2).reshape(batch,frequencies,frames,-1)
            states=[]
            for index,layer in enumerate(self.layers):
                if cache:
                    value,cached=layer.process(value,memories.get('layers',[{}]*len(self.layers))[index])
                    states.append(cached)
                elif self.training and self.gradient_checkpointing:
                    value=checkpoint(layer,value,use_reentrant=False)
                else:value=layer(value)
            residual=self.unpack(self.decoder(value)).float()
        with torch.autocast(inpt.device.type,enabled=False):
            correction=self.compress(reference+scale*residual)-self.compress(reference)
            output=(inpt[...,0,:].float()+correction).permute(0,3,1,2)
        return output,dict(encoder=history,layers=states) if cache else None

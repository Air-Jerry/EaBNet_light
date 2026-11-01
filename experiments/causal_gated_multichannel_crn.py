"""Independent eight-microphone grouped dual-path CRN, without an inherited base.

Grouped temporal gates and frequency/time recurrent paths are motivated by GTCRN
(https://github.com/Xiaobin-Rong/gtcrn). This independently written adaptation
keeps all257 frequency bins and uses fixed statistics during training/inference.
It does not use GTCRN code, external pretrained weights or future time frames.
"""
import torch
from torch import nn
import torch.nn.functional as F

from light_causal_spatial_model import CausalSpatialEnhancer


class FixedStatisticsAffine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(channels))
        self.bias=nn.Parameter(torch.zeros(channels))
        self.register_buffer('mean',torch.zeros(channels))
        self.register_buffer('variance',torch.ones(channels))

    def forward(self,value):
        return F.batch_norm(value,self.mean,self.variance,self.weight,self.bias,False,0.,1e-5)


class GatedTemporalBlock(nn.Module):
    def __init__(self,channels,dilation):
        super().__init__()
        self.channels=channels
        self.history_frames=2*dilation
        self.input=nn.Conv2d(channels//2*3,channels,1)
        self.input_affine=FixedStatisticsAffine(channels)
        self.input_activation=nn.PReLU(channels)
        self.temporal=nn.Conv2d(channels,channels,(3,3),dilation=(dilation,1),padding=(0,1),groups=channels)
        self.temporal_affine=FixedStatisticsAffine(channels)
        self.temporal_activation=nn.PReLU(channels)
        self.output=nn.Conv2d(channels,channels//2,1)
        self.output_affine=FixedStatisticsAffine(channels//2)
        self.gate_recurrent=nn.GRU(channels//2,channels,batch_first=True,bidirectional=False)
        self.gate_output=nn.Linear(channels,channels//2)

    def process(self,value,state=None,cache=False):
        memories={} if state is None else state
        left,right=value.chunk(2,1)
        batch,channels,frames,frequencies=left.shape
        neighborhood=F.pad(left,(1,1)).unfold(-1,3,1).permute(0,1,4,2,3).reshape(batch,channels*3,frames,frequencies)
        feature=self.input_activation(self.input_affine(self.input(neighborhood)))
        history=memories.get('history')
        shape=(batch,self.channels,self.history_frames,frequencies)
        if history is None:
            history=feature.new_zeros(shape)
        if history.shape!=shape or history.device!=feature.device or history.dtype!=feature.dtype:
            raise ValueError('Temporal convolution state differs from the stream')
        joined=torch.cat((history,feature),2)
        feature=self.temporal_activation(self.temporal_affine(self.temporal(joined)))
        feature=self.output_affine(self.output(feature))
        with torch.autocast(value.device.type,enabled=False):
            pooled=feature.float().square().mean(-1).transpose(1,2)
            hidden=memories.get('gate')
            if hidden is not None and (hidden.shape!=(1,batch,self.channels) or hidden.dtype!=torch.float32 or hidden.device!=value.device):
                raise ValueError('Temporal gate state differs from the stream')
            gate,hidden=self.gate_recurrent(pooled,hidden)
            gate=torch.sigmoid(self.gate_output(gate)).transpose(1,2).unsqueeze(-1)
            feature=(feature.float()*gate).to(value.dtype)
        result=torch.stack((feature,right),2).flatten(1,2)
        updated=dict(history=joined[:,:,-self.history_frames:].detach().clone(),gate=hidden.detach().clone()) if cache else None
        return result,updated


class GroupedDualPath(nn.Module):
    def __init__(self,channels,frequencies):
        super().__init__()
        self.channels,self.frequencies=channels,frequencies
        self.frequency=nn.ModuleList([nn.GRU(channels//2,channels//4,batch_first=True,bidirectional=True) for _ in range(2)])
        self.frequency_output=nn.Linear(channels,channels)
        self.frequency_norm=nn.LayerNorm((frequencies,channels))
        self.temporal=nn.ModuleList([nn.GRU(channels//2,channels//2,batch_first=True,bidirectional=False) for _ in range(2)])
        self.temporal_output=nn.Linear(channels,channels)
        self.temporal_norm=nn.LayerNorm((frequencies,channels))

    def process(self,value,state=None,cache=False):
        batch,channels,frames,frequencies=value.shape
        if (channels,frequencies)!=(self.channels,self.frequencies):
            raise ValueError('Dual-path frequency/channel shape differs')
        dtype=value.dtype
        with torch.autocast(value.device.type,enabled=False):
            current=value.float().permute(0,2,3,1)
            independent=current.reshape(batch*frames,frequencies,channels)
            frequency=torch.cat([rnn(part)[0] for rnn,part in zip(self.frequency,independent.chunk(2,-1))],-1)
            current=current+self.frequency_norm(self.frequency_output(frequency).reshape(batch,frames,frequencies,channels))
            sequence=current.permute(0,2,1,3).reshape(batch*frequencies,frames,channels)
            if state is None:
                states=[None,None]
            else:
                if state.shape!=(1,batch*frequencies,channels) or state.dtype!=torch.float32 or state.device!=value.device:
                    raise ValueError('Dual-path recurrent state differs from the stream')
                states=[part.contiguous() for part in state.chunk(2,-1)]
            results=[rnn(part,hidden) for rnn,part,hidden in zip(self.temporal,sequence.chunk(2,-1),states)]
            temporal=torch.cat([pair[0] for pair in results],-1)
            temporal=self.temporal_output(temporal).reshape(batch,frequencies,frames,channels).permute(0,2,1,3)
            result=current+self.temporal_norm(temporal)
            updated=torch.cat([pair[1] for pair in results],-1).detach().clone() if cache else None
        return result.permute(0,3,1,2).to(dtype),updated


class CausalGatedMultichannelCRN(nn.Module):
    is_causal=True
    refiner_causal=True

    def __init__(self,M=8,*,channels=64,recurrent_layers=4,**kwargs):
        super().__init__()
        if M!=8 or channels<8 or channels%4 or recurrent_layers<1 or kwargs.get('is_causal',True) is False:
            raise ValueError('Use eight microphones and positive compatible causal dimensions')
        self.M,self.context,self.channels=8,32,channels
        self.use_bfloat16_training=True
        self.down=nn.ModuleList([
            nn.Sequential(nn.Conv2d(24,channels,(1,5),stride=(1,2),padding=(0,2)),FixedStatisticsAffine(channels),nn.PReLU(channels)),
            nn.Sequential(nn.Conv2d(channels,channels,(1,5),stride=(1,2),padding=(0,2),groups=2),FixedStatisticsAffine(channels),nn.PReLU(channels))])
        self.encoder=nn.ModuleList([GatedTemporalBlock(channels,d) for d in (1,2,5)])
        self.recurrent=nn.ModuleList([GroupedDualPath(channels,65) for _ in range(recurrent_layers)])
        self.decoder=nn.ModuleList([GatedTemporalBlock(channels,d) for d in (5,2,1)])
        self.up=nn.Sequential(nn.ConvTranspose2d(channels,channels,(1,5),stride=(1,2),padding=(0,2),groups=2),FixedStatisticsAffine(channels),nn.PReLU(channels))
        self.output=nn.ConvTranspose2d(channels,18,(1,5),stride=(1,2),padding=(0,2))
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def process(self,inpt,state=None):
        if inpt.ndim!=5 or inpt.shape[-3:]!=(257,8,2) or min(inpt.shape[:2])<1:
            raise ValueError('Expected compressed STFT(B,T,257,8,2)')
        memories={} if state is None else state
        cache=state is not None
        batch,frames=inpt.shape[:2]
        with torch.autocast(inpt.device.type,enabled=False):
            spectra=inpt.float()
            magnitude=torch.linalg.vector_norm(spectra,dim=-1,keepdim=True)
            # Keep absolute compressed magnitude and phase components before time modeling.
            features=torch.cat((magnitude,spectra),-1).flatten(-2).permute(0,3,1,2).contiguous()
        updated=dict(encoder=[],recurrent=[],decoder=[]) if cache else None
        enabled=self.training and self.use_bfloat16_training and inpt.device.type=='cuda'
        with torch.autocast(inpt.device.type,dtype=torch.bfloat16,enabled=enabled):
            skips=[];value=features
            for layer in self.down:
                value=layer(value);skips.append(value)
            for index,layer in enumerate(self.encoder):
                memory=memories.get('encoder',[None]*len(self.encoder))[index]
                value,memory=layer.process(value,memory,cache);skips.append(value)
                if cache:updated['encoder'].append(memory)
            for index,layer in enumerate(self.recurrent):
                memory=memories.get('recurrent',[None]*len(self.recurrent))[index]
                value,memory=layer.process(value,memory,cache)
                if cache:updated['recurrent'].append(memory)
            for index,layer in enumerate(self.decoder):
                memory=memories.get('decoder',[None]*len(self.decoder))[index]
                value,memory=layer.process(value+skips.pop(),memory,cache)
                if cache:updated['decoder'].append(memory)
            value=self.up(value+skips.pop())
            extra=self.output(value+skips.pop()).float().permute(0,2,3,1)
        assert not skips
        with torch.autocast(inpt.device.type,enabled=False):
            raw=CausalSpatialEnhancer.decompress(spectra)
            weights=extra[...,:16].reshape(batch,frames,257,8,2)
            real=(weights[...,0]*raw[...,0]-weights[...,1]*raw[...,1]).sum(-1)
            imaginary=(weights[...,0]*raw[...,1]+weights[...,1]*raw[...,0]).sum(-1)
            reference=raw[...,0,:]
            correction=CausalSpatialEnhancer.compress(reference+torch.stack((real,imaginary),-1))-CausalSpatialEnhancer.compress(reference)
            scale=magnitude[...,0,:].mean(-2,keepdim=True).clamp_min(1e-4)
            result=spectra[...,0,:]+correction+extra[...,16:]*scale
        return result.permute(0,3,1,2),updated

    def forward(self,inpt,frame_lengths=None):
        if frame_lengths is not None:
            lengths=torch.as_tensor(frame_lengths)
            if lengths.numel()!=inpt.shape[0] or torch.any(lengths<1) or torch.any(lengths>inpt.shape[1]):
                raise ValueError('One valid frame length is required per sample')
        return self.process(inpt)[0]

    def forward_stream(self,inpt,state=None):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError('Streaming requires eval mode and disabled gradients')
        return self.process(inpt,{} if state is None else state)

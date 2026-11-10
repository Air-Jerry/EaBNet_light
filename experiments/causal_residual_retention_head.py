"""Unregistered parallel-trained causal memory head over a frozen eight-mic source.

Every original frequency and microphone, plus the source estimate, enters the
head. Quality, CUDA training cost and all-hop deadlines require real testing.
"""
import torch
from torch import nn

from experiments.causal_folded_retention import PackedRetentionLayer
from light_causal_spatial_model import left_convolution


class CausalResidualRetentionHead(nn.Module):
    is_causal = True
    refiner_causal = True

    def __init__(self, source):
        super().__init__()
        if getattr(source,'is_causal',None) is not True or getattr(source,'M',None)!=8 or not callable(getattr(source,'process',None)):
            raise ValueError('Use an eight-microphone causal source with explicit stream state')
        self.base=source
        self.base.requires_grad_(False)
        self.base.zero_grad(set_to_none=True)
        self.M=8;self.context=int(source.context)
        with torch.random.fork_rng(devices=[]):
            self.encoder=nn.Conv1d(18*4,64,5)
            self.layers=nn.ModuleList()
            full=None
            for _ in range(2):
                layer=PackedRetentionLayer(64,128,4,self.context,65,full)
                self.layers.append(layer);full=layer.full
            self.output=nn.Linear(64,8)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.train(self.training)

    def train(self,mode=True):
        super().train(mode);self.base.eval();return self

    def process(self,inpt,state=None):
        if inpt.ndim!=5 or inpt.shape[-3:]!=(257,8,2) or min(inpt.shape[:2])<1:
            raise ValueError('Expected compressed STFT(B,T,257,8,2)')
        cache=state is not None
        if cache and (not isinstance(state,dict) or (state and set(state)!={'base','encoder','layers'})):
            raise ValueError('Use a complete independently owned residual-retention state')
        memories={} if state is None else state
        if memories and (not isinstance(memories['base'],dict) or not isinstance(memories['layers'],list)
                         or len(memories['layers'])!=2 or any(not isinstance(s,dict) for s in memories['layers'])):
            raise ValueError('Residual-retention layer state differs from this model')
        estimate,base_state=self.base.process(inpt,memories.get('base',{}) if cache else None)
        batch,frames=inpt.shape[:2]
        if estimate.shape!=(batch,2,frames,257):raise ValueError('Causal source spectrum shape differs')
        with torch.autocast(inpt.device.type,enabled=False):
            reference=inpt[...,0,:].float()
            scale=torch.linalg.vector_norm(reference,dim=-1).mean(-1,keepdim=True).unsqueeze(-1).clamp_min(1e-4)
            features=torch.cat((inpt.float().flatten(-2),estimate.float().permute(0,2,3,1)),-1)/scale
            packed=torch.nn.functional.pad(features,(0,0,0,3)).reshape(batch,frames,65,72).permute(0,2,1,3)
        history=memories.get('encoder')
        if history is not None and (not isinstance(history,torch.Tensor) or history.shape!=(batch*65,72,4)
                                    or history.dtype!=torch.float32 or history.device!=inpt.device):
            raise ValueError('Residual-retention encoder state differs from this stream')
        enabled=self.training and inpt.device.type=='cuda'
        with torch.autocast(inpt.device.type,dtype=torch.bfloat16,enabled=enabled):
            value=packed.reshape(batch*65,frames,72).transpose(1,2)
            value,encoder_state=left_convolution(self.encoder,value,history)
            value=value.transpose(1,2).reshape(batch,65,frames,64)
            states=[]
            for index,layer in enumerate(self.layers):
                if cache:
                    value,updated=layer.process(value,memories.get('layers',[{},{}])[index])
                    states.append(updated)
                else:value=layer(value)
            delta=self.output(value).float().permute(0,2,1,3).reshape(batch,frames,260,2)[:,:,:257]
        output=estimate.float()+(delta*scale).permute(0,3,1,2)
        updated=dict(base=base_state,encoder=encoder_state,layers=states) if cache else None
        return output,updated

    def forward(self,inpt,frame_lengths=None):
        if frame_lengths is not None:
            lengths=torch.as_tensor(frame_lengths)
            if lengths.numel()!=inpt.shape[0] or torch.any(lengths<1) or torch.any(lengths>inpt.shape[1]):
                raise ValueError('Use one valid frame length per sample')
        return self.process(inpt)[0]

    def forward_stream(self,inpt,state=None):
        if self.training or torch.is_grad_enabled():raise RuntimeError('Streaming requires eval and disabled gradients')
        return self.process(inpt,{} if state is None else state)

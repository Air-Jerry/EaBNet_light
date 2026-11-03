"""Unregistered inference-only one-frame scan with external states and CUDA replay.

Only tensor-tree caches whose true initial values are zero are supported.
The exact zero-state bootstrap and actual CUDA capture still require checking
for every new core. Training and CPU execution retain the original core.
"""
from threading import RLock

import torch
from torch import nn


def tensor_leaves(value):
    if torch.is_tensor(value):return [value]
    if isinstance(value,dict):return [t for child in value.values() for t in tensor_leaves(child)]
    if isinstance(value,(list,tuple)):return [t for child in value for t in tensor_leaves(child)]
    raise ValueError('The graph core must expose a tensor-only state tree')


def map_state(value,function):
    if torch.is_tensor(value):return function(value)
    if isinstance(value,dict):return {key:map_state(child,function) for key,child in value.items()}
    if isinstance(value,tuple):return tuple(map_state(child,function) for child in value)
    if isinstance(value,list):return [map_state(child,function) for child in value]
    raise ValueError('The graph core must expose a tensor-only state tree')


def checked_pairs(destination,source):
    if torch.is_tensor(destination):
        if (not torch.is_tensor(source) or source.shape!=destination.shape
                or source.dtype!=destination.dtype or source.device!=destination.device):
            raise ValueError('Graph state tensor differs from this stream')
        return [(destination,source)]
    if type(source) is not type(destination):raise ValueError('Graph state tree differs from this stream')
    if isinstance(destination,dict):
        if destination.keys()!=source.keys():raise ValueError('Graph state tree differs from this stream')
        return [pair for key in destination for pair in checked_pairs(destination[key],source[key])]
    if isinstance(destination,(tuple,list)):
        if len(source)!=len(destination):raise ValueError('Graph state tree differs from this stream')
        return [pair for a,b in zip(destination,source) for pair in checked_pairs(a,b)]
    raise ValueError('The graph core must expose a tensor-only state tree')


class _FrameReplay:
    def __init__(self,core,sample):
        self.features=torch.zeros_like(sample[:,:1])
        _,template=core.forward_stream(self.features)
        self.inputs=map_state(template,torch.zeros_like)
        self.last_event=None
        self.key=(sample.shape[0],sample.device,sample.dtype)
        side=torch.cuda.Stream(device=sample.device)
        side.wait_stream(torch.cuda.current_stream(sample.device))
        with torch.cuda.stream(side):
            for _ in range(3):core.forward_stream(self.features,self.inputs)
        torch.cuda.current_stream(sample.device).wait_stream(side)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output,self.updated=core.forward_stream(self.features,self.inputs)

    def step(self,features,state):
        if state is None:
            for value in tensor_leaves(self.inputs):value.zero_()
        else:
            pairs=checked_pairs(self.inputs,state)
            for destination,source in pairs:destination.copy_(source)
        self.features.copy_(features)
        self.graph.replay()
        return self.output,self.updated

    def storage_bytes(self):
        values=[self.features,self.output,*tensor_leaves(self.inputs),*tensor_leaves(self.updated)]
        storages={t.untyped_storage().data_ptr():t.untyped_storage().nbytes() for t in values}
        return sum(storages.values())


class CausalFrameGraph(nn.Module):
    def __init__(self,core):
        super().__init__()
        if (getattr(core,'is_causal',None) is not True or getattr(core,'M',None)!=8
                or not callable(getattr(core,'forward_stream',None))):
            raise ValueError('Use a causal eight-microphone core with external stream states')
        self.core=core;self.M=8;self.is_causal=True;self.context=int(getattr(core,'context',128))
        self._replay=None;self._replay_lock=RLock()

    def _clear_replay(self):
        if self._replay is not None:
            if self._replay.last_event is not None:self._replay.last_event.synchronize()
            self._replay=None

    def train(self,mode=True):
        if hasattr(self,'_replay_lock') and mode:
            with self._replay_lock:self._clear_replay()
        return super().train(mode)

    def _apply(self,function,recurse=True):
        with self._replay_lock:self._clear_replay()
        return super()._apply(function,recurse=recurse)

    def load_state_dict(self,*args,**kwargs):
        with self._replay_lock:self._clear_replay()
        return super().load_state_dict(*args,**kwargs)

    def _scan(self,value,state,return_state):
        if (value.ndim!=5 or value.shape[-3:]!=(257,8,2) or min(value.shape[:2])<1
                or value.dtype!=torch.float32):
            raise ValueError('Expected FP32 compressed STFT(B,T,257,8,2)')
        with self._replay_lock:
            key=(value.shape[0],value.device,value.dtype)
            if self._replay is None or self._replay.key!=key:
                self._clear_replay();self._replay=_FrameReplay(self.core,value)
            replay=self._replay
            current=torch.cuda.current_stream(value.device)
            if replay.last_event is not None:current.wait_event(replay.last_event)
            pieces=[]
            for index in range(value.shape[1]):
                output,state=replay.step(value[:,index:index+1],state)
                pieces.append(output.clone())
            result=torch.cat(pieces,2)
            owned=map_state(state,lambda tensor:tensor.detach().clone()) if return_state else None
            replay.last_event=torch.cuda.Event();replay.last_event.record(current)
            return result,owned

    def forward(self,value,frame_lengths=None):
        if self.training or torch.is_grad_enabled() or value.device.type!='cuda':
            return self.core(value,frame_lengths=frame_lengths)
        if frame_lengths is not None:
            lengths=torch.as_tensor(frame_lengths)
            if lengths.numel()!=value.shape[0] or torch.any(lengths<1) or torch.any(lengths>value.shape[1]):
                raise ValueError('One valid frame length is required per sample')
        return self._scan(value,None,False)[0]

    def forward_stream(self,value,state=None):
        if self.training or torch.is_grad_enabled():raise RuntimeError('Streaming requires eval mode and disabled gradients')
        if value.device.type!='cuda':return self.core.forward_stream(value,state)
        return self._scan(value,state,True)

    @property
    def graph_static_state_storage_bytes(self):
        return 0 if self._replay is None else self._replay.storage_bytes()

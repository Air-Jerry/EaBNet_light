"""Evaluate an explicitly changed causal graph using the old trained weights.

The frequency-only FFT, recurrent beamformer and weight tensors are retained.
This is a causal adaptation, not the paper's original implementation. Streaming
caches and CUDA real-time latency must be verified separately before deployment.
"""

from pathlib import Path
import hashlib
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
import torch.nn.functional as F

import evaluate_light as evaluator
from EaBNet_light import SpatialAttention
from light_reference_variants import CBAMChannelAttention


class PrefixChannelAttention(CBAMChannelAttention):
    def forward(self,x):
        count=torch.arange(1,x.shape[-2]+1,device=x.device,dtype=torch.float64)
        average=x.to(torch.float64).sum(-1).cumsum(-1)/(count*x.shape[-1])
        maximum=x.amax(-1).cummax(-1).values
        weights=torch.sigmoid(self.mlp(average.transpose(1,2).to(x.dtype))+
                              self.mlp(maximum.transpose(1,2)))
        return x*weights.transpose(1,2).unsqueeze(-1)


class LeftSpatialAttention(SpatialAttention):
    def forward(self,x):
        maps=(torch.relu(x.mean(1,keepdim=True)),torch.relu(x.amax(1,keepdim=True)))
        values=[]
        for value,conv in zip(maps,(self.avg_conv,self.max_conv)):
            value=F.pad(value,(0,0,conv.kernel_size[0]-1,0))
            values.append(F.conv2d(value,conv.weight,conv.bias,conv.stride,
                (0,conv.kernel_size[1]//2),conv.dilation,conv.groups))
        return x*torch.sigmoid(values[0]+values[1])


class CausalCBAMConversion(nn.Module):
    is_causal=True

    def __init__(self,source):
        super().__init__()
        self.net=source;self.M=source.M
        self.replacements=dict(channel=0,spatial=0)
        def replace(parent):
            for name,child in tuple(parent.named_children()):
                new=None
                if isinstance(child,CBAMChannelAttention):
                    new=PrefixChannelAttention(child.channels)
                    self.replacements['channel']+=1
                elif isinstance(child,SpatialAttention):
                    new=LeftSpatialAttention(child.avg_conv.kernel_size[0])
                    self.replacements['spatial']+=1
                if new is None:replace(child)
                else:
                    new.to(device=next(child.parameters()).device,dtype=next(child.parameters()).dtype)
                    new.load_state_dict(child.state_dict(),strict=True)
                    setattr(parent,name,new)
        replace(self.net)
        assert self.replacements==dict(channel=15,spatial=25),self.replacements
        assert source.cred.dfsmn.is_causal and source.cred.dfsmn.right_memory_size==0
        assert not source.bf_map.rnn1.bidirectional and not source.bf_map.rnn2.bidirectional
        self.train(False)

    def train(self,mode=True):
        super().train(mode)
        for module in self.net.modules():
            if isinstance(module,nn.modules.batchnorm._BatchNorm):module.eval()
        return self

    def forward(self,inpt,frame_lengths=None):
        return self.net(inpt)


def main(argv=None):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    args=evaluator.parse_args(argv)
    if getattr(args,'checkpoint_dir',None) or args.max_samples!=0 or args.match_estimate_level!='no':
        raise ValueError('This conversion comparison requires one checkpoint, all selected samples, no gain matching')
    load=evaluator.load_model
    def converted_load(path,device,expected=None):
        source,saved=load(path,device,expected or 'cbam_flat_projection64')
        assert saved['structure_candidate']=='cbam_flat_projection64'
        model=CausalCBAMConversion(source).to(device).eval()
        with torch.inference_mode():
            torch.manual_seed(1234)
            value=torch.randn(1,25,257,8,2,device=device)*.05
            changed=value.clone();changed[:,12:]*=7
            prefix=model(value)[:,:,:12];other=model(changed)[:,:,:12]
            error=(prefix-other).abs().max().item()
            assert error<1e-6 and torch.count_nonzero(prefix)>0,error
        saved=dict(saved)
        original={key:saved[key] for key in ('structure_candidate','candidate_implementation_sha256','candidate_evidence')}
        saved['structure_candidate']='causal_cbam_prefix_prototype'
        saved['candidate_implementation_sha256']={**saved['candidate_implementation_sha256'],
            'experiments/probe_causal_cbam_conversion.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        saved['candidate_evidence']=dict(model_causal=True,future_frames=0,author_confirmed=False,
            exact_reproduction_verified=False,conversion='Prefix mean/max channel attention; left spatial kernels; fixed BN statistics',
            original_checkpoint_evidence=original,future_perturbation_max_error=error,
            cuda_streaming_and_latency_verified=False,replacements=model.replacements)
        return model,saved
    evaluator.load_model=converted_load
    try:return evaluator.evaluate(args)
    finally:evaluator.load_model=load


if __name__=='__main__':main()

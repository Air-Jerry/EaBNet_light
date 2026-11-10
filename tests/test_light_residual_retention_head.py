"""The head must learn without changing its source and preserve causal, owned state."""
import copy

import pytest
import torch

from experiments.causal_frame_graph import tensor_leaves
from experiments.causal_residual_retention_head import CausalResidualRetentionHead
from light_candidate_runtime import build_model_candidate,RECURRENT_ADAPT_CANDIDATE


def models():
    torch.set_num_threads(2);torch.manual_seed(1234)
    source=build_model_candidate(RECURRENT_ADAPT_CANDIDATE).eval()
    with torch.no_grad():
        for module in (source.base.base.beamformer,source.base.base.mapping_output,
                       source.base.base.retention.output,source.base.output,source.output):
            module.weight.normal_(std=.01)
    return source,CausalResidualRetentionHead(copy.deepcopy(source))


def test_exact_warm_output_and_all_new_blocks_learn_while_source_stays_fixed():
    source,current=models();value=torch.randn(1,17,257,8,2)*.1
    with torch.inference_mode():
        current.eval();assert torch.equal(source(value),current(value))
    frozen={name:value.clone() for name,value in current.base.state_dict().items()}
    current.train();assert not current.base.training and all(layer.training for layer in current.layers)
    assert all(not p.requires_grad for p in current.base.parameters())
    original=current.encoder.weight.detach().clone()
    retention=current.layers[0].mhsa.qkv.weight.detach().clone()
    optimizer=torch.optim.Adam(current.parameters(),lr=1e-4)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        (current(value)-.1).square().mean().backward()
        assert all(p.grad is None for p in current.base.parameters())
        assert all(torch.isfinite(p.grad).all() for p in current.parameters() if p.grad is not None)
        optimizer.step()
    assert not torch.equal(original,current.encoder.weight)
    assert not torch.equal(retention,current.layers[0].mhsa.qkv.weight)
    assert all(torch.equal(value,current.base.state_dict()[name]) for name,value in frozen.items())
    assert all(torch.isfinite(v).all() for state in optimizer.state.values() for v in state.values())


def test_activated_head_causality_and_chunked_owned_fixed_state():
    _,current=models();current.eval()
    with torch.no_grad():current.output.weight.normal_(std=.01)
    value=(torch.randn(1,131,257,8,2)*.1).requires_grad_()
    output=current(value);output[:,:,:67].square().sum().backward()
    assert torch.count_nonzero(value.grad[:,67:])==0
    with torch.inference_mode():
        expected=current(value.detach());changed=value.detach().clone();changed[:,67:]*=100
        assert (current(changed)[:,:,:67]-expected[:,:,:67]).abs().max()<1e-6
        state=None;parts=[];memory_bytes=[]
        for start,end in ((0,1),(1,17),(17,67),(67,131)):
            before=[x.clone() for x in tensor_leaves(state)] if state else []
            part,updated=current.forward_stream(value.detach()[:,start:end],state)
            if state:
                assert all(torch.equal(a,b) for a,b in zip(before,tensor_leaves(state)))
                assert {x.untyped_storage().data_ptr() for x in tensor_leaves(state)}.isdisjoint(
                    {x.untyped_storage().data_ptr() for x in tensor_leaves(updated)})
            parts.append(part);state=updated
            memory_bytes.append(sum(x.numel()*x.element_size() for x in tensor_leaves(state)))
        assert len(set(memory_bytes))==1
        assert (expected-torch.cat(parts,2)).abs().max()<1e-5


def test_bad_stream_state_rejected():
    _,current=models();current.eval();value=torch.randn(1,1,257,8,2)*.1
    with torch.inference_mode():
        for state in ({'base':{}},{'base':{},'encoder':None,'layers':[]},
                      {'base':{},'encoder':torch.zeros(1),'layers':[{},{}]}):
            with pytest.raises(ValueError):current.forward_stream(value,state)

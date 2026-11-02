"""Nontrivial warm parity, extra-state learning and causality of the128-state head."""
import copy

import pytest
import torch

from experiments.causal_wide_conditioned_subband import CausalWideConditionedSubband
from light_candidate_runtime import JOINT_CONDITIONED_CANDIDATE,build_model_candidate


def models():
    torch.set_num_threads(2);torch.manual_seed(1234)
    source=build_model_candidate(JOINT_CONDITIONED_CANDIDATE).eval()
    with torch.no_grad():
        for module in (source.base.beamformer,source.base.mapping_output,source.base.retention.output,
                       source.output,source.fullband_mix,source.fullband_projection,source.spatial_projection):
            module.weight.normal_(std=.003)
    target=CausalWideConditionedSubband(copy.deepcopy(source.base))
    target.load_warm_32(source.state_dict())
    return target,source


def learn(model):
    model.train();optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    original=model.recurrent.weight_ih_l0.detach().clone()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        value=torch.randn(1,17,257,8,2)*.02
        loss=(model(value)-torch.randn(1,2,17,257)*.02).square().mean()
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        assert torch.count_nonzero(model.output.weight.grad[:,32:])>0
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
    assert not torch.equal(model.recurrent.weight_ih_l0[32:128],original[32:128])
    assert {int(s['step']) for s in optimizer.state.values()}=={2}
    assert all(torch.isfinite(v).all() for s in optimizer.state.values() for v in s.values())


@pytest.mark.parametrize('frames',[17,65,131])
def test_wide_head_keeps_nonzero_warm_output_and_source_ownership(frames):
    model,source=models();model.eval()
    value=torch.randn(1,frames,257,8,2)*.02
    with torch.no_grad():
        expected=source(value);actual=model(value)
        torch.testing.assert_close(actual,expected,rtol=1e-4,atol=2e-6)
        assert not torch.equal(expected,value[...,0,:].permute(0,3,1,2))
    assert sum(p.numel() for p in model.parameters())==5166584
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==2586886
    assert all(not p.requires_grad for p in model.base.parameters())
    assert all(p.requires_grad for p in source.base.parameters())
    assert not ({p.data_ptr() for p in source.parameters()} & {p.data_ptr() for p in model.parameters()})


def test_new_temporal_states_learn_while_the_causal_backbone_stays_fixed():
    model,_=models();base={k:v.detach().clone() for k,v in model.base.state_dict().items()}
    learn(model)
    assert not model.base.training
    for key,value in model.base.state_dict().items():torch.testing.assert_close(value,base[key],rtol=0,atol=0)


def tensors(value):
    if torch.is_tensor(value):return [value]
    if isinstance(value,dict):return [v for item in value.values() for v in tensors(item)]
    if isinstance(value,(tuple,list)):return [v for item in value for v in tensors(item)]
    return []


def test_trained_wide_head_has_no_future_gradient_and_independent_bounded_state():
    model,_=models();learn(model);model.eval();model.zero_grad(set_to_none=True)
    value=(torch.randn(1,131,257,8,2)*.02).requires_grad_()
    full=model(value);changed=value.detach().clone();changed[:,65:]*=11
    torch.testing.assert_close(full[:,:,:65],model(changed)[:,:,:65],rtol=0,atol=0)
    full[:,:,:65].square().mean().backward();assert torch.count_nonzero(value.grad[:,65:])==0
    with torch.no_grad():
        state=None;parts=[]
        for start,end in ((0,1),(1,17),(17,64),(64,65),(65,131)):
            part,state=model.forward_stream(value[:,start:end],state);parts.append(part)
        torch.testing.assert_close(torch.cat(parts,2),full,rtol=1e-4,atol=2e-6)
        before=[v.clone() for v in tensors(state)]
        assert sum(v.untyped_storage().nbytes() for v in tensors(state))==6218512
        _,later=model.forward_stream(value[:,:1],state)
        assert sum(v.untyped_storage().nbytes() for v in tensors(later))==6218512
        for old,current in zip(before,tensors(state)):torch.testing.assert_close(old,current,rtol=0,atol=0)
        _,other=model.forward_stream(value[:,:1])
        assert not ({v.data_ptr() for v in tensors(state)} & {v.data_ptr() for v in tensors(other)})


def test_wide_warm_load_rejects_incompatible_or_nonfinite_source():
    model,source=models();state=source.state_dict();bad=dict(state)
    bad['input.weight']=bad['input.weight'][:16]
    with pytest.raises(ValueError,match='32-state'):model.load_warm_32(bad)
    bad={k:v.clone() for k,v in state.items()};bad['output.bias'][0]=float('nan')
    with pytest.raises(ValueError,match='Nonfinite'):model.load_warm_32(bad)

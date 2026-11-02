"""Nonzero warm-output preservation, joint gradients and unchanged causal state."""

import copy

import pytest
import torch

from experiments.causal_joint_conditioned_subband import CausalJointConditionedSubband
from light_candidate_runtime import (SPATIAL_CONDITIONED_CANDIDATE,JOINT_CONDITIONED_CANDIDATE,
    build_model_candidate,candidate_metadata,read_candidate_checkpoint)


def models():
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    source=build_model_candidate(SPATIAL_CONDITIONED_CANDIDATE).eval()
    with torch.no_grad():
        for module in (source.base.beamformer,source.base.mapping_output,source.base.retention.output,
                       source.output,source.fullband_projection,source.spatial_projection):
            module.weight.normal_(std=.003)
    model=CausalJointConditionedSubband(copy.deepcopy(source.base))
    model.load_state_dict(source.state_dict(),strict=True)
    return model,source


def train_once(model, value):
    model.train();model.zero_grad(set_to_none=True)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    loss=(model(value)-torch.randn(value.shape[0],2,value.shape[1],257)*.02).square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
    optimizer.step()
    assert {int(state['step']) for state in optimizer.state.values()}=={1}
    assert all(torch.isfinite(v).all() for state in optimizer.state.values() for v in state.values())
    return optimizer


def tensors(value):
    if torch.is_tensor(value):return [value]
    if isinstance(value,dict):return [t for child in value.values() for t in tensors(child)]
    if isinstance(value,(list,tuple)):return [t for child in value for t in tensors(child)]
    return []


def test_joint_model_preserves_nonzero_source_output_and_parameter_keys():
    model,source=models();model.eval()
    value=torch.randn(1,17,257,8,2)*.02
    with torch.no_grad():
        expected=source(value)
        torch.testing.assert_close(model(value),expected,rtol=0,atol=0)
        assert not torch.equal(expected,value[...,0,:].permute(0,3,1,2))
    assert model.state_dict().keys()==source.state_dict().keys()
    assert sum(p.numel() for p in model.parameters())==5058680
    assert all(p.requires_grad for p in model.parameters())
    assert all(not p.requires_grad for p in source.base.parameters())
    for name,value in model.state_dict().items():
        torch.testing.assert_close(value,source.state_dict()[name],rtol=0,atol=0)


def test_joint_train_mode_enables_recurrent_backward_and_updates_the_base():
    model,source=models()
    assert model.training and model.base.training
    assert all(module.training for module in model.base.modules() if isinstance(module,torch.nn.RNNBase))
    original={name:p.detach().clone() for name,p in model.named_parameters()}
    train_once(model,torch.randn(1,17,257,8,2)*.02)
    for name in ('base.beamformer.weight','base.mapping_output.weight','base.retention.qkv.weight','output.weight'):
        parameter=dict(model.named_parameters())[name]
        assert torch.count_nonzero(parameter.grad)>0
        assert not torch.equal(parameter,original[name])
    assert any(not torch.equal(parameter,original[name]) for name,parameter in model.named_parameters() if name.startswith('base.stem.'))
    assert all(not p.requires_grad for p in source.base.parameters())
    model.eval()
    assert not model.base.training
    assert all(not module.training for module in model.base.modules() if isinstance(module,torch.nn.RNNBase))


def test_trained_joint_model_has_no_future_gradient_and_owns_bounded_state():
    model,_=models();train_once(model,torch.randn(1,17,257,8,2)*.02);model.eval();model.zero_grad(set_to_none=True)
    value=(torch.randn(1,131,257,8,2)*.02).requires_grad_()
    full=model(value);changed=value.detach().clone();changed[:,65:]*=11
    torch.testing.assert_close(full[:,:,:65],model(changed)[:,:,:65],rtol=0,atol=0)
    full[:,:,:65].square().mean().backward()
    assert torch.count_nonzero(value.grad[:,65:])==0
    with torch.no_grad():
        parts=[];state=None
        for start,end in ((0,1),(1,17),(17,64),(64,65),(65,131)):
            part,state=model.forward_stream(value[:,start:end],state);parts.append(part)
        torch.testing.assert_close(torch.cat(parts,2),full,rtol=1e-4,atol=2e-6)
        original=[v.clone() for v in tensors(state)]
        size=sum(v.untyped_storage().nbytes() for v in tensors(state))
        _,later=model.forward_stream(value[:,:1],state)
        assert sum(v.untyped_storage().nbytes() for v in tensors(later))==size==6119824
        for saved,current in zip(original,tensors(state)):
            torch.testing.assert_close(saved,current,rtol=0,atol=0)
        _,separate=model.forward_stream(value[:,:1])
        assert not ({v.data_ptr() for v in tensors(state)} & {v.data_ptr() for v in tensors(separate)})


def test_joint_stream_rejects_training_mode_and_bad_base():
    model,_=models()
    with torch.no_grad(),pytest.raises(RuntimeError,match='eval mode'):
        model.forward_stream(torch.randn(1,1,257,8,2))
    with pytest.raises(ValueError,match='causal'):
        CausalJointConditionedSubband(torch.nn.Linear(2,2))


def test_joint_checkpoint_has_its_own_identity_and_never_claims_a_frozen_base(tmp_path):
    import train_light
    model=build_model_candidate(JOINT_CONDITIONED_CANDIDATE)
    metadata=candidate_metadata(JOINT_CONDITIONED_CANDIDATE)
    evidence=metadata['candidate_evidence']
    assert metadata['candidate_schema']==23 and evidence['author_confirmed'] is False
    assert evidence['frozen_backbone'] is False and evidence['model_config']['frozen_causal_parameters']==0
    assert evidence['model_config']['trainable_parameters']==sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert candidate_metadata(SPATIAL_CONDITIONED_CANDIDATE)['candidate_evidence']['frozen_backbone'] is True
    args=vars(train_light.parse_args([]));args.update(metadata)
    checkpoint=tmp_path/'joint.pt';torch.save(dict(args=args,model_state_dict=model.state_dict()),checkpoint)
    state,name=read_candidate_checkpoint(checkpoint,JOINT_CONDITIONED_CANDIDATE)
    assert name==JOINT_CONDITIONED_CANDIDATE
    with pytest.raises(ValueError,match='Candidate mismatch'):
        read_candidate_checkpoint(checkpoint,SPATIAL_CONDITIONED_CANDIDATE)
    state['args']['candidate_evidence']['frozen_backbone']=True;torch.save(state,checkpoint)
    with pytest.raises(ValueError,match='candidate_evidence'):
        read_candidate_checkpoint(checkpoint,JOINT_CONDITIONED_CANDIDATE)

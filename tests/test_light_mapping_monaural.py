"""Nonzero warm identity,actual mapping updates,and unchanged causal streaming equations."""
import copy

import pytest
import torch

from experiments.causal_mapping_monaural import CausalMappingMonaural
from experiments.causal_global_adapt_monaural import CausalGlobalAdaptMonaural
from experiments.causal_frame_graph import tensor_leaves
from light_candidate_runtime import build_model_candidate,candidate_metadata


def models(adapter):
    torch.set_num_threads(2)
    torch.manual_seed(314)
    source=build_model_candidate('causal_decoder_monaural_spectral_v1')
    with torch.no_grad():
        for module in (source.base.base.beamformer,source.base.base.mapping_output,
                       source.base.base.retention.output,source.base.output,source.output):
            module.weight.normal_(std=.01)
    model=adapter(copy.deepcopy(source.base))
    model.load_state_dict(source.state_dict(),strict=True)
    return source,model


@pytest.mark.parametrize('adapter',[CausalMappingMonaural,CausalGlobalAdaptMonaural])
def test_complete_nonzero_warm_output_and_old_registered_metadata_are_unchanged(adapter):
    metadata=candidate_metadata('causal_decoder_monaural_spectral_v1')
    source,model=models(adapter)
    source.eval();model.eval()
    value=torch.randn(1,131,257,8,2)*.1
    with torch.inference_mode():
        assert torch.equal(source(value),model(value))
    assert model.state_dict().keys()==source.state_dict().keys()
    assert sum(p.numel() for p in model.parameters())==5686130
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)>3191840
    assert not any(p.requires_grad for p in model.base.base.encoders.parameters())
    assert not any(p.requires_grad for layer in model.base.base.layers[:-1] for p in layer.parameters())
    expected=4951426 if adapter is CausalGlobalAdaptMonaural else 3236066
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==expected
    assert all(p.requires_grad for name in model.mapping_modules for p in getattr(model.base.base,name).parameters())
    assert metadata==candidate_metadata('causal_decoder_monaural_spectral_v1')


@pytest.mark.parametrize('adapter',[CausalMappingMonaural,CausalGlobalAdaptMonaural])
def test_actual_adam_updates_mapping_with_finite_states_and_unchanged_frozen_encoder(adapter):
    source,model=models(adapter)
    frozen={name:p.detach().clone() for name,p in model.named_parameters() if not p.requires_grad}
    old=model.base.base.mapping_output.weight.detach().clone()
    old_memory=model.base.base.retention.qkv.weight.detach().clone()
    global_rnn=next(module for module in model.base.base.layers[-1].modules() if isinstance(module,torch.nn.GRU))
    old_global=global_rnn.weight_ih_l0.detach().clone()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    model.train()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        value=torch.randn(1,17,257,8,2)*.1
        loss=(model(value)-torch.randn(1,2,17,257)*.05).square().mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        optimizer.step()
    assert not torch.equal(old,model.base.base.mapping_output.weight)
    assert not torch.equal(old_memory,model.base.base.retention.qkv.weight)
    if adapter is CausalGlobalAdaptMonaural:
        assert not torch.equal(old_global,global_rnn.weight_ih_l0)
    assert all(torch.equal(value,dict(model.named_parameters())[name]) for name,value in frozen.items())
    assert {int(item['step']) for item in optimizer.state.values()}=={2}
    assert all(torch.isfinite(value).all() for item in optimizer.state.values() for value in item.values())
    assert sum(p.requires_grad for p in source.parameters())<sum(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize('adapter',[CausalMappingMonaural,CausalGlobalAdaptMonaural])
def test_future_gradient_zero_and_bounded_independent_stream_state_remain(adapter):
    _,model=models(adapter)
    model.eval()
    value=(torch.randn(1,131,257,8,2)*.1).requires_grad_()
    full=model(value)
    derivative,=torch.autograd.grad(full[:,:,:65].square().mean(),value)
    assert torch.count_nonzero(derivative[:,65:])==0
    with torch.no_grad():
        altered=value.clone();altered[:,65:]*=11
        assert torch.equal(full[:,:,:65],model(altered)[:,:,:65])
        parts=[];state=None;sizes=set()
        for start,end in ((0,1),(1,17),(17,64),(64,65),(65,131)):
            part,state=model.forward_stream(value[:,start:end],state)
            parts.append(part)
            sizes.add(sum(t.untyped_storage().nbytes() for t in tensor_leaves(state)))
        assert sizes=={6219024}
        torch.testing.assert_close(torch.cat(parts,2),full,rtol=1e-4,atol=2e-6)
        old=[t.clone() for t in tensor_leaves(state)]
        _,later=model.forward_stream(value[:,:1],state)
        assert all(torch.equal(a,b) for a,b in zip(old,tensor_leaves(state)))
        pointers=lambda s:{t.untyped_storage().data_ptr() for t in tensor_leaves(s)}
        assert pointers(state).isdisjoint(pointers(later))

import copy

import torch

from experiments.causal_frame_graph import tensor_leaves
from experiments.causal_expanded_temporal_monaural import CausalExpandedTemporalMonaural, expand_temporal_tensor
from light_candidate_runtime import build_model_candidate, RECURRENT_ADAPT_CANDIDATE


def models():
    torch.set_num_threads(2);torch.manual_seed(1234)
    source=build_model_candidate(RECURRENT_ADAPT_CANDIDATE)
    with torch.no_grad():
        for module in (source.base.base.beamformer,source.base.base.mapping_output,
                       source.base.base.retention.output,source.base.output,source.output):module.weight.normal_(std=.01)
    model=CausalExpandedTemporalMonaural(copy.deepcopy(source.base));model.load_warm(source.state_dict())
    return source,model


def test_exact_old_gate_embedding_zero_cross_connections_and_preserved_warm_output():
    source,model=models();source.eval();model.eval()
    assert sum(p.numel() for p in model.parameters())==6252146
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==5667402
    for gate in range(3):
        old=slice(gate*128,(gate+1)*128);new=slice(gate*256,gate*256+128)
        assert torch.equal(model.recurrent.weight_ih_l0[new],source.recurrent.weight_ih_l0[old])
        assert torch.equal(model.recurrent.weight_hh_l0[new,:128],source.recurrent.weight_hh_l0[old])
        assert torch.count_nonzero(model.recurrent.weight_hh_l0[new,128:])==0
    assert torch.equal(model.projection.weight[:,:128],source.projection.weight)
    assert torch.count_nonzero(model.projection.weight[:,128:])==0
    with torch.inference_mode():
        features=torch.randn(1,131,257,8,2)*.1
        assert (source(features)-model(features)).abs().max()<1e-6
    moment=torch.randn_like(source.recurrent.weight_hh_l0)
    expanded=expand_temporal_tensor('recurrent.weight_hh_l0',moment,model.recurrent.weight_hh_l0,zero=True)
    for gate in range(3):
        assert torch.equal(expanded[gate*256:gate*256+128,:128],moment[gate*128:(gate+1)*128])
        assert torch.count_nonzero(expanded[gate*256+128:(gate+1)*256])==0


def test_new_neurons_learn_with_finite_adam_and_old_frozen_parameters_stay_fixed():
    _,model=models();model.train();opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    old=model.recurrent.weight_ih_l0[128:256].detach().clone()
    features=torch.randn(1,17,257,8,2)*.1
    for _ in range(3):
        opt.zero_grad(set_to_none=True);(model(features)-.1).square().mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None);opt.step()
    assert not torch.equal(old,model.recurrent.weight_ih_l0[128:256])
    assert all(torch.equal(v,dict(model.named_parameters())[n]) for n,v in frozen.items())
    assert all(torch.isfinite(v).all() for item in opt.state.values() for v in item.values())


def test_activated_expansion_has_no_future_time_gradient_and_fixed_independent_cache():
    _,model=models();model.eval()
    with torch.no_grad():model.projection.weight[:,128:].normal_(std=.001)
    features=(torch.randn(1,131,257,8,2)*.1).requires_grad_(True)
    model(features)[:,:,:67].square().sum().backward();assert torch.count_nonzero(features.grad[:,67:])==0
    with torch.inference_mode():
        original=model(features.detach());changed=features.detach().clone();changed[:,67:]*=100
        assert (original[:,:,:67]-model(changed)[:,:,:67]).abs().max()<1e-6
        state=None;pieces=[]
        for start,end in ((0,1),(1,17),(17,67),(67,131)):
            incoming=[t.clone() for t in tensor_leaves(state)] if state is not None else []
            part,updated=model.forward_stream(features.detach()[:,start:end],state)
            if state is not None:
                assert all(torch.equal(a,b) for a,b in zip(incoming,tensor_leaves(state)))
                assert {t.untyped_storage().data_ptr() for t in tensor_leaves(state)}.isdisjoint(
                    {t.untyped_storage().data_ptr() for t in tensor_leaves(updated)})
            state=updated;pieces.append(part)
            assert sum(t.numel()*t.element_size() for t in tensor_leaves(state))==6219536
        assert (original-torch.cat(pieces,2)).abs().max()<1e-5

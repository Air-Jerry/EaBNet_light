import copy

import torch

from experiments.causal_frame_graph import tensor_leaves
from experiments.causal_two_block_adapt_monaural import CausalTwoBlockAdaptMonaural
from light_candidate_runtime import build_model_candidate, RECURRENT_ADAPT_CANDIDATE


def models():
    torch.set_num_threads(2);torch.manual_seed(1234)
    source=build_model_candidate(RECURRENT_ADAPT_CANDIDATE)
    with torch.no_grad():
        for module in (source.base.base.beamformer,source.base.base.mapping_output,
                       source.base.base.retention.output,source.base.output,source.output):module.weight.normal_(std=.01)
    model=CausalTwoBlockAdaptMonaural(copy.deepcopy(source.base));model.load_state_dict(source.state_dict(),strict=True)
    return source,model


def test_original_weights_and_function_are_exact_with_only_two_frequency_blocks_trainable():
    source,model=models();source.eval();model.eval();layers=model.base.base.layers
    assert sum(p.numel() for p in model.parameters())==5686130
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==5251346
    assert not any(p.requires_grad for l in layers[:2] for p in l.parameters())
    assert layers[0].full is layers[1].full is layers[2].full is layers[3].full
    assert not any(p.requires_grad for p in layers[2].full.parameters())
    for layer in layers[2:4]:
        assert all(p.requires_grad for name,p in layer.named_parameters() if not name.startswith('full.'))
    with torch.inference_mode():
        features=torch.randn(1,131,257,8,2)*.1
        torch.testing.assert_close(model(features),source(features),rtol=0,atol=0)


def test_penultimate_recurrence_learns_with_finite_adam_and_all_frozen_weights_stay_fixed():
    _,model=models();model.train();opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    old=model.base.base.layers[2].mhsa.weight_ih_l0.detach().clone()
    features=torch.randn(1,17,257,8,2)*.1
    for _ in range(3):
        opt.zero_grad(set_to_none=True);(model(features)-.1).square().mean().backward()
        assert model.base.base.layers[2].mhsa.weight_ih_l0.grad.abs().sum()>0
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None);opt.step()
    assert not torch.equal(old,model.base.base.layers[2].mhsa.weight_ih_l0)
    assert all(torch.equal(v,dict(model.named_parameters())[n]) for n,v in frozen.items())
    assert all(torch.isfinite(v).all() for item in opt.state.values() for v in item.values())


def test_activated_penultimate_weights_keep_temporal_causality_and_owned_fixed_stream_state():
    _,model=models();model.eval()
    with torch.no_grad():model.base.base.layers[2].mhsa.weight_ih_l0.add_(torch.randn_like(model.base.base.layers[2].mhsa.weight_ih_l0)*.001)
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
            assert sum(t.numel()*t.element_size() for t in tensor_leaves(state))==6219024
        assert (original-torch.cat(pieces,2)).abs().max()<1e-5

import copy

import torch

from experiments.causal_frame_graph import tensor_leaves
from experiments.causal_frequency_mix_monaural import CurrentFrameFrequencyMix, CausalFrequencyMixMonaural
from light_candidate_runtime import build_model_candidate, RECURRENT_ADAPT_CANDIDATE


def models():
    torch.set_num_threads(2);torch.manual_seed(1234)
    source=build_model_candidate(RECURRENT_ADAPT_CANDIDATE)
    with torch.no_grad():
        for module in (source.base.base.beamformer,source.base.base.mapping_output,
                       source.base.base.retention.output,source.base.output,source.output):
            module.weight.normal_(std=.01)
    model=CausalFrequencyMixMonaural(copy.deepcopy(source.base));model.load_warm(source.state_dict())
    return source,model


def test_frequency_sequence_never_connects_different_time_frames():
    torch.set_num_threads(2);torch.manual_seed(12)
    mixer=CurrentFrameFrequencyMix()
    with torch.no_grad():mixer.projection.weight.normal_(std=.01)
    value=torch.randn(2,16,5,257,requires_grad=True)
    output=mixer(value);output[:,:,:2].square().sum().backward()
    assert torch.count_nonzero(value.grad[:,:,2:])==0
    with torch.no_grad():
        torch.testing.assert_close(output,torch.cat([mixer(value[:,:,i:i+1]) for i in range(5)],2),rtol=1e-5,atol=1e-6)
    assert mixer.frequency_gru.bidirectional


def test_exact_warm_identity_and_actual_finite_learning():
    source,model=models();source.eval();model.eval()
    features=torch.randn(1,17,257,8,2)*.1
    assert sum(p.numel() for p in model.parameters())==5692994
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==5108250
    with torch.inference_mode():torch.testing.assert_close(model(features),source(features),rtol=0,atol=0)
    model.train();optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    recurrent=model.frequency_mix.frequency_gru.weight_ih_l0.detach().clone()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True);(model(features)-.1).square().mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
    assert not torch.equal(recurrent,model.frequency_mix.frequency_gru.weight_ih_l0)
    assert all(torch.equal(v,dict(model.named_parameters())[n]) for n,v in frozen.items())
    assert all(torch.isfinite(v).all() for item in optimizer.state.values() for v in item.values())


def test_activated_branch_is_temporally_causal_and_keeps_fixed_independent_states():
    _,model=models();model.eval()
    with torch.no_grad():model.frequency_mix.projection.weight.normal_(std=.001)
    features=(torch.randn(1,131,257,8,2)*.1).requires_grad_(True)
    output=model(features);output[:,:,:67].square().sum().backward()
    assert torch.count_nonzero(features.grad[:,67:])==0
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

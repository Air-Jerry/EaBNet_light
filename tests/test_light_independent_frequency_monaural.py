"""Verify exact inheritance, genuinely independent frequency learning and causal states."""
import copy

import torch

from experiments.causal_frame_graph import tensor_leaves
from experiments.causal_independent_frequency_monaural import CausalIndependentFrequencyMonaural
from light_candidate_runtime import RECURRENT_ADAPT_CANDIDATE, build_model_candidate, candidate_metadata


def models():
    torch.set_num_threads(2); torch.manual_seed(1234)
    source = build_model_candidate(RECURRENT_ADAPT_CANDIDATE)
    with torch.no_grad():
        for module in (source.base.base.beamformer, source.base.base.mapping_output,
                       source.base.base.retention.output, source.base.output, source.output):
            module.weight.normal_(std=.01)
    model = CausalIndependentFrequencyMonaural(copy.deepcopy(source.base))
    model.load_state_dict(source.state_dict(), strict=True)
    return source, model


def test_exact_nonzero_warm_identity_with_independent_final_frequency_parameters():
    source, model = models(); source.eval(); model.eval()
    assert set(source.state_dict()) == set(model.state_dict())
    assert sum(p.numel() for p in model.parameters()) == 5720450
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 5135706
    layers = model.base.base.layers
    assert layers[0].full is layers[1].full is layers[2].full
    assert layers[3].full is not layers[0].full
    assert not any(p.requires_grad for layer in layers[:3] for p in layer.parameters())
    assert all(p.requires_grad for p in layers[3].full.parameters())
    features = torch.randn(1, 131, 257, 8, 2)*.1
    with torch.inference_mode():
        torch.testing.assert_close(model(features), source(features), rtol=0, atol=0)
    assert candidate_metadata(RECURRENT_ADAPT_CANDIDATE)['candidate_schema'] == 31


def test_actual_learning_keeps_shared_early_matrix_and_all_frozen_parameters_unchanged():
    source, model = models(); model.train()
    frozen = {n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    before = model.base.base.layers[3].full.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    features = torch.randn(1, 13, 257, 8, 2)*.1
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        output = model(features)
        (output-.1).square().mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert model.base.base.layers[3].full.weight.grad.abs().sum() > 0
        optimizer.step()
    assert not torch.equal(before, model.base.base.layers[3].full.weight)
    assert all(torch.equal(p, frozen[n]) for n,p in model.named_parameters() if n in frozen)
    assert all(torch.isfinite(v).all() for item in optimizer.state.values() for v in item.values())
    torch.testing.assert_close(model.base.base.layers[0].full.weight, source.base.base.layers[0].full.weight, rtol=0, atol=0)


def test_activated_frequency_matrix_has_zero_future_gradient_and_fixed_owned_stream_states():
    _, model = models(); model.eval()
    with torch.no_grad(): model.base.base.layers[3].full.weight.add_(torch.randn_like(model.base.base.layers[3].full.weight)*.001)
    features = (torch.randn(1, 131, 257, 8, 2)*.1).requires_grad_(True)
    output = model(features)
    output[:, :, :67].square().sum().backward()
    assert torch.count_nonzero(features.grad[:, 67:]) == 0
    with torch.inference_mode():
        original = model(features.detach()); changed = features.detach().clone(); changed[:, 67:] *= 100
        assert (original[:, :, :67]-model(changed)[:, :, :67]).abs().max() < 1e-6
        pieces = []; state = None
        for start, end in ((0,1),(1,17),(17,67),(67,131)):
            incoming = [t.clone() for t in tensor_leaves(state)] if state is not None else []
            result, updated = model.forward_stream(features.detach()[:, start:end], state)
            if state is not None:
                assert all(torch.equal(a,b) for a,b in zip(incoming,tensor_leaves(state)))
                assert {t.untyped_storage().data_ptr() for t in tensor_leaves(state)}.isdisjoint(
                    {t.untyped_storage().data_ptr() for t in tensor_leaves(updated)})
            state = updated; pieces.append(result)
            assert sum(t.numel()*t.element_size() for t in tensor_leaves(state)) == 6219024
        assert (original-torch.cat(pieces,2)).abs().max() < 1e-5
        _, independent = model.forward_stream(features.detach()[:, :1])
        assert {t.untyped_storage().data_ptr() for t in tensor_leaves(state)}.isdisjoint(
            {t.untyped_storage().data_ptr() for t in tensor_leaves(independent)})

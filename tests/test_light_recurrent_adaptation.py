"""The shared frequency matrix must stay frozen when adapting the final temporal block."""
import torch
import pytest

from experiments.causal_recurrent_adapt_monaural import CausalRecurrentAdaptMonaural
from experiments.graphed_model_training import graphed_model_training
from experiments.causal_frame_graph import tensor_leaves
from test_light_mapping_monaural import models


def test_recurrent_adaptation_preserves_warm_output_and_earlier_shared_parameters():
    source, model = models(CausalRecurrentAdaptMonaural)
    source.eval(); model.eval()
    value = torch.randn(1, 131, 257, 8, 2) * .1
    with torch.no_grad():
        assert torch.equal(source(value), model(value))
    assert source.state_dict().keys() == model.state_dict().keys()
    layers = model.base.base.layers
    assert layers[0].full is layers[3].full
    assert not any(p.requires_grad for layer in layers[:3] for p in layer.parameters())
    assert not any(p.requires_grad for p in layers[3].full.parameters())
    assert all(p.requires_grad for p in layers[3].mhsa.parameters())
    assert sum(p.numel() for p in model.parameters()) == 5686130
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 5101386


def test_recurrent_adaptation_actually_learns_without_updating_shared_frozen_weights():
    _, model = models(CausalRecurrentAdaptMonaural)
    frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    recurrent = model.base.base.layers[3].mhsa
    old = recurrent.weight_ih_l0.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    model.train()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        value = torch.randn(1, 17, 257, 8, 2) * .1
        estimate = model(value)
        loss = (estimate - torch.randn_like(estimate) * .05).square().mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        optimizer.step()
    assert not torch.equal(old, recurrent.weight_ih_l0)
    assert all(torch.equal(value, dict(model.named_parameters())[name]) for name, value in frozen.items())
    assert {int(values['step']) for values in optimizer.state.values()} == {2}
    assert all(torch.isfinite(value).all() for values in optimizer.state.values() for value in values.values())
    model.eval()
    value = (torch.randn(1, 65, 257, 8, 2) * .1).requires_grad_()
    derivative, = torch.autograd.grad(model(value)[:, :, :33].square().mean(), value)
    assert torch.count_nonzero(derivative[:, 33:]) == 0
    with torch.no_grad():
        complete = model(value)
        changed = value.clone(); changed[:, 33:] *= 11
        assert torch.equal(complete[:, :, :33], model(changed)[:, :, :33])
        state = None; parts = []; sizes = set()
        for start, end in ((0, 1), (1, 17), (17, 33), (33, 65)):
            part, state = model.forward_stream(value[:, start:end], state)
            parts.append(part)
            sizes.add(sum(t.untyped_storage().nbytes() for t in tensor_leaves(state)))
        torch.testing.assert_close(torch.cat(parts, 2), complete, rtol=1e-4, atol=2e-6)
        assert sizes == {6219024}
        before = [t.clone() for t in tensor_leaves(state)]
        _, next_state = model.forward_stream(value[:, :1], state)
        assert all(torch.equal(a, b) for a, b in zip(before, tensor_leaves(state)))
        pointers = lambda values: {t.untyped_storage().data_ptr() for t in tensor_leaves(values)}
        assert pointers(state).isdisjoint(pointers(next_state))


def test_graph_capture_rejects_cpu_without_changing_the_model():
    _, model = models(CausalRecurrentAdaptMonaural)
    state = {name: value.clone() for name, value in model.state_dict().items()}
    original = model.process
    with pytest.raises(ValueError, match='CUDA'):
        with graphed_model_training(model, torch.randn(1, 17, 257, 8, 2)):
            pytest.fail('CPU capture must fail')
    assert model.process == original and 'process' not in model.__dict__
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in state.items())

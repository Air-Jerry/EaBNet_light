"""Verify that head training preserves the causal backbone and inference graph."""

import copy

import pytest
import torch

from experiments.causal_frozen_conditioned_subband import CausalFrozenBaseConditionedSubband
from experiments.causal_fullband_conditioned_subband import CausalFullbandConditionedSubband
from test_light_subband_gru import model as source_model


def test_fixed_backbone_survives_real_adam_updates_and_mode_changes():
    old = source_model(nonzero=True)
    model = CausalFrozenBaseConditionedSubband(copy.deepcopy(old.base))
    missing, unexpected = model.load_state_dict(old.state_dict(), strict=False)
    assert len(missing) == 10 and all(key.startswith('fullband_') for key in missing) and not unexpected
    value = torch.randn(2, 21, 257, 8, 2)*.02
    with torch.no_grad():
        torch.testing.assert_close(model(value), old(value), rtol=0, atol=0)
    before = {key: value.clone() for key, value in model.base.state_dict().items()}
    trained_before = model.fullband_rnn.weight_hh_l0.detach().clone()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    frozen = set(model.base.parameters())
    assert all(not parameter.requires_grad for parameter in frozen)
    assert all(parameter.requires_grad for name, parameter in model.named_parameters() if not name.startswith('base.'))
    for _ in range(3):
        model.eval()
        model.train()
        assert model.training and model.fullband_rnn.training and not model.base.training
        assert all(not module.training for module in model.base.modules())
        optimizer.zero_grad(set_to_none=True)
        model(value).square().mean().backward()
        assert all(p.grad is None for p in frozen)
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters() if p.requires_grad)
        optimizer.step()
    assert torch.count_nonzero(model.fullband_rnn.weight_hh_l0.grad) > 0
    assert not torch.equal(trained_before, model.fullband_rnn.weight_hh_l0)
    assert not frozen.intersection(optimizer.state)
    assert all(torch.isfinite(value).all() for state in optimizer.state.values() for value in state.values())
    for key, base_value in model.base.state_dict().items():
        torch.testing.assert_close(base_value, before[key], rtol=0, atol=0)
    reference = CausalFullbandConditionedSubband(copy.deepcopy(old.base)).eval()
    reference.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(value), reference(value), rtol=0, atol=0)


def test_fixed_causal_backbone_keeps_nonzero_condition_causality_and_owned_state():
    old = source_model(nonzero=True)
    model = CausalFrozenBaseConditionedSubband(copy.deepcopy(old.base)).eval()
    model.load_state_dict(old.state_dict(), strict=False)
    model.fullband_mix.weight.data.normal_(std=.01)
    value = (torch.randn(1, 31, 257, 8, 2)*.02).requires_grad_()
    full = model(value)
    changed = value.detach().clone()
    changed[:, 13:] *= 11
    torch.testing.assert_close(full[:, :, :13], model(changed)[:, :, :13], rtol=0, atol=0)
    full[:, :, :13].square().mean().backward()
    assert torch.count_nonzero(value.grad[:, 13:]) == 0
    assert all(p.grad is None for p in model.base.parameters())
    with torch.no_grad():
        first, state = model.forward_stream(value[:, :7])
        snapshot = state['fullband_hidden'].clone()
        second, updated = model.forward_stream(value[:, 7:], state)
        torch.testing.assert_close(torch.cat((first, second), 2), full, rtol=1e-4, atol=2e-6)
        torch.testing.assert_close(state['fullband_hidden'], snapshot, rtol=0, atol=0)
        assert updated['fullband_hidden'].untyped_storage().nbytes() == 512
    old.base.is_causal = False
    with pytest.raises(ValueError, match='must itself be causal'):
        CausalFrozenBaseConditionedSubband(old.base)

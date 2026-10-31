"""Physical power, padding, silence and the exact upstream PMSQE convention."""

import pytest
import torch

from experiments.causal_pmsqe_objective import PMSQEObjective, physical_power
from light_quality_loss import QualityObjective


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def features(objective, lengths=(1601, 1921), silent=False):
    generator = torch.Generator().manual_seed(79)
    target = torch.randn(len(lengths), max(lengths), generator=generator) * (0 if silent else .03)
    mixture = target[..., None].expand(-1, -1, 8).clone()
    mixture += torch.randn(mixture.shape, generator=generator) * .01
    mix, ref = objective.build_features(mixture, target, 512, 160, 320, .5, 'cpu', lengths)
    return mix[..., 0, :].permute(0, 3, 1, 2), ref, [n // 160 + 1 for n in lengths]


@pytest.mark.parametrize('power', [.3, .5, .7])
def test_physical_power_exactly_inverts_frontend_including_epsilon(power):
    values = torch.tensor([0., 1e-12, 1e-9, 1e-8, 1e-4, .2, 2.])
    phase = torch.linspace(-2, 2, len(values))
    spectrum = torch.polar(values, phase)
    compressed = spectrum * spectrum.abs().clamp_min(1e-8).pow(power - 1)
    ri = torch.view_as_real(compressed).T[None, :, None, :]
    torch.testing.assert_close(physical_power(ri, power)[0, 0], values.square(), rtol=2e-5, atol=1e-30)


def test_matches_upstream_power_window_and_257_frame_axis_convention():
    objective = PMSQEObjective()
    estimate, ref, frames = features(objective, lengths=(40960,))
    assert frames == [257]
    combined = objective(estimate, ref, frames)
    raw = QualityObjective()
    features(raw, lengths=(40960,))
    physical = raw(estimate, ref, frames)
    expected = objective.perceptual((physical_power(estimate) + 1e-12).transpose(1, 2),
                                    (physical_power(ref) + 1e-12).transpose(1, 2)).mean()
    torch.testing.assert_close(combined, physical + .1 * expected)
    assert objective.perceptual.pow_correc_factor == pytest.approx(512 / torch.hann_window(320).square().sum())


def test_padded_values_cannot_change_loss_or_gradients():
    first, second = PMSQEObjective(), PMSQEObjective()
    estimate, ref, frames = features(first)
    other, other_ref, _ = features(second)
    other[0, :, frames[0]:] = 1e20
    other_ref[0, :, frames[0]:] = -1e20
    estimate.requires_grad_(); other.requires_grad_()
    loss, other_loss = first(estimate, ref, frames), second(other, other_ref, frames)
    torch.testing.assert_close(loss, other_loss, rtol=0, atol=0)
    loss.backward(); other_loss.backward()
    torch.testing.assert_close(estimate.grad, other.grad, rtol=0, atol=0)
    assert torch.count_nonzero(other.grad[0, :, frames[0]:]) == 0
    assert torch.isfinite(other.grad).all()


@pytest.mark.parametrize('silent', [False, True])
def test_finite_gradients_and_silent_references_use_physical_supervision(silent):
    objective = PMSQEObjective()
    estimate, ref, frames = features(objective, silent=silent)
    estimate.requires_grad_()
    loss = objective(estimate, ref, frames)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(estimate.grad).all()
    assert estimate.grad.abs().max() > 0
    assert objective.stats()['combined_total'] == pytest.approx(float(loss.detach()), rel=1e-6)
    if silent:
        assert objective.stats()['pmsqe'] == 0


def test_all_zero_estimate_and_reference_have_zero_loss_and_finite_gradient():
    objective = PMSQEObjective()
    estimate, ref, frames = features(objective, silent=True)
    estimate = torch.zeros_like(estimate, requires_grad=True)
    loss = objective(estimate, ref, frames)
    loss.backward()
    assert loss == 0 and torch.isfinite(estimate.grad).all()
    assert torch.count_nonzero(estimate.grad) == 0


def test_zero_weight_is_exact_original_loss_without_changing_estimates():
    objective, original = PMSQEObjective(weight=0), QualityObjective()
    estimate, ref, frames = features(objective)
    features(original)
    before = estimate.clone()
    torch.testing.assert_close(objective(estimate, ref, frames), original(estimate, ref, frames), rtol=0, atol=0)
    torch.testing.assert_close(estimate, before, rtol=0, atol=0)
    with pytest.raises(ValueError): PMSQEObjective(weight=float('nan'))

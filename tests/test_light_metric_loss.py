"""PESQ labels, differentiable G supervision, and independent D updates."""

import copy

import numpy as np
import pytest
import torch

from light_metric_loss import MetricDiscriminator, MetricObjective, real_pesq
from light_quality_loss import QualityObjective


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def features(objective):
    time = torch.arange(8000) / 16000
    target = (0.1 * torch.sin(2 * torch.pi * 149 * time))[None]
    mixture = target[..., None].repeat(1, 1, 8)
    _, reference = objective.build_features(mixture, target, 512, 160, 320, 0.5,
                                           torch.device("cpu"), [8000])
    estimate = (reference * 0.8).detach().requires_grad_()
    return estimate, reference, [51]


def test_discriminator_architecture_output_and_input_gradient():
    model = MetricDiscriminator()
    value = torch.rand(2, 51, 257, requires_grad=True)
    score = model(value, value * 0.7)
    assert score.shape == (2,) and torch.isfinite(score).all()
    assert ((score > 0) & (score < 1)).all()
    score.sum().backward()
    assert torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0


def test_real_labels_are_detached_and_metric_gradient_reaches_generator():
    received = []

    def label(clean, enhanced):
        assert isinstance(clean, np.ndarray) and isinstance(enhanced, np.ndarray)
        received.append((clean.copy(), enhanced.copy()))
        return 3.0

    objective = MetricObjective("cpu", score_fn=label)
    objective.training = True
    predictions = []
    hook = objective.discriminator.register_forward_hook(
        lambda module, inputs, output: predictions.append(output.detach().clone()))
    estimate, target, frames = features(objective)
    total = objective(estimate, target, frames)
    hook.remove()
    base = QualityObjective.__call__(objective, estimate, target, frames)
    torch.testing.assert_close(total - base, 0.05 * (predictions[2] - 1).square().mean(), atol=1e-6, rtol=1e-5)
    metric_gradient = torch.autograd.grad(total - base, estimate, retain_graph=True)[0]
    assert torch.isfinite(metric_gradient).all() and metric_gradient.abs().sum() > 0
    d_gradients = [p.grad.clone() for p in objective.discriminator.parameters()]
    total.backward()
    for parameter, gradient in zip(objective.discriminator.parameters(), d_gradients):
        torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
    assert torch.isfinite(estimate.grad).all()
    assert len(received) == 1 and received[0][0].shape == (8000,)
    assert objective.updates == 1
    stats = objective.metric_stats()
    assert stats["mean_training_pesq"] == 3.0 and stats["valid_pesq_labels"] == 1
    assert stats["discriminator_calibration_rmse"] == pytest.approx(
        abs(predictions[1].item() - (3.0 - 1) / 3.5), abs=1e-7)


def test_invalid_labels_do_not_fabricate_pesq_or_update_discriminator():
    objective = MetricObjective("cpu", score_fn=lambda *args: None)
    objective.training = True
    before = [p.clone() for p in objective.discriminator.parameters()]
    estimate, target, frames = features(objective)
    loss = objective(estimate, target, frames)
    loss.backward()
    assert objective.updates == 0 and not objective.optimizer.state
    for parameter, previous in zip(objective.discriminator.parameters(), before):
        torch.testing.assert_close(parameter, previous, rtol=0, atol=0)
    assert torch.isfinite(estimate.grad).all()
    stats = objective.metric_stats()
    assert stats["valid_pesq_labels"] == 0 and stats["invalid_pesq_labels"] == 1


def test_validation_uses_base_loss_without_labels_or_d_updates():
    objective = MetricObjective("cpu", score_fn=lambda *args: pytest.fail("Validation must not label or update D"))
    estimate, target, frames = features(objective)
    before = copy.deepcopy(objective.state_dict())
    with torch.no_grad():
        actual = objective(estimate, target, frames)
        expected = QualityObjective.__call__(objective, estimate, target, frames)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name, value in before["discriminator"].items():
        torch.testing.assert_close(objective.state_dict()["discriminator"][name], value, rtol=0, atol=0)
    assert objective.updates == 0


def test_discriminator_adam_and_buffers_resume_the_next_update_exactly():
    first = MetricObjective("cpu", score_fn=lambda *args: 3.0)
    first.training = True
    estimate, target, frames = features(first)
    first(estimate, target, frames).backward()
    state = copy.deepcopy(first.state_dict())
    second = MetricObjective("cpu", discriminator_lr=1e-2, score_fn=lambda *args: 3.0)
    second.training = True
    second.load_state_dict(copy.deepcopy(state))
    assert second.optimizer.param_groups[0]["lr"] == 1e-4
    for objective in (first, second):
        x, y, lengths = features(objective)
        torch.manual_seed(731)
        objective(x, y, lengths).backward()
    assert first.updates == second.updates == 2
    for name, value in first.state_dict()["discriminator"].items():
        torch.testing.assert_close(value, second.state_dict()["discriminator"][name], rtol=0, atol=0)
    for one, two in zip(first.optimizer.state.values(), second.optimizer.state.values()):
        for key in one:
            torch.testing.assert_close(one[key], two[key], rtol=0, atol=0)


def test_nonfinite_labels_abort_before_discriminator_step():
    objective = MetricObjective("cpu", score_fn=lambda *args: float("nan"))
    objective.training = True
    with pytest.raises(FloatingPointError, match="Non-finite"):
        objective(*features(objective))
    assert objective.updates == 0


def test_actual_wideband_pesq_and_silence_failure():
    time = np.arange(32000) / 16000
    clean = (0.1 * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * time)) *
             (np.sin(2 * np.pi * 143 * time) + 0.4 * np.sin(2 * np.pi * 286 * time))).astype(np.float32)
    degraded = clean + np.random.default_rng(4).normal(0, 0.002, len(time)).astype(np.float32)
    assert real_pesq(clean, degraded) is not None
    assert real_pesq(np.zeros_like(clean), np.zeros_like(clean)) is None

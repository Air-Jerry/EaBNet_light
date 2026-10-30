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


def test_accelerated_runtime_preserves_loss_gradients_and_discriminator_adam():
    standard = MetricObjective("cpu", score_fn=lambda *args: 3.0)
    fast = MetricObjective("cpu", score_fn=lambda *args: 3.0, accelerated=True)
    fast.load_state_dict(copy.deepcopy(standard.state_dict()))
    for _ in range(2):
        outputs = []
        for objective in (standard, fast):
            objective.training = True
            estimate, target, frames = features(objective)
            torch.manual_seed(197)
            loss = objective(estimate, target, frames)
            loss.backward()
            outputs.append((loss.detach(), estimate.grad.clone(), copy.deepcopy(objective.state_dict())))
        torch.testing.assert_close(outputs[0][0], outputs[1][0], rtol=0, atol=0)
        torch.testing.assert_close(outputs[0][1], outputs[1][1], rtol=1e-5, atol=1e-6)
        # Shared ISTFT backward sums the two FP32 branches before its transform.
        assert torch.linalg.vector_norm(outputs[0][1] - outputs[1][1]) / torch.linalg.vector_norm(outputs[0][1]) < 1e-4
        for key, value in outputs[0][2]["discriminator"].items():
            torch.testing.assert_close(value, outputs[1][2]["discriminator"][key], rtol=0, atol=0)
        for one, two in zip(outputs[0][2]["optimizer"]["state"].values(),
                            outputs[1][2]["optimizer"]["state"].values()):
            for key in one:
                torch.testing.assert_close(one[key], two[key], rtol=0, atol=0)


def test_four_merged_examples_preserve_four_d_steps_and_accumulated_g_gradient():
    standard = MetricObjective("cpu", score_fn=lambda *args: 3.0)
    merged = MetricObjective("cpu", score_fn=lambda *args: 3.0, accelerated=True)
    merged.load_state_dict(copy.deepcopy(standard.state_dict()))
    standard.training = merged.training = True
    references, targets, gradients, losses = [], [], [], []
    torch.manual_seed(71)
    for index in range(4):
        estimate, target, frames = features(standard)
        estimate = (estimate.detach() * (1 + 0.02 * index)).requires_grad_()
        loss = standard(estimate, target, frames)
        (loss / 4).backward()
        gradients.append(estimate.grad)
        references.append(target)
        targets.append(standard._target)
        losses.append(loss.detach())
    merged._target = torch.cat(targets)
    merged._lengths = [8000] * 4
    estimate = torch.cat([reference * 0.8 * (1 + 0.02 * i)
                          for i, reference in enumerate(references)]).requires_grad_()
    torch.manual_seed(71)
    loss = merged(estimate, torch.cat(references), [51] * 4)
    loss.backward()
    torch.testing.assert_close(loss, torch.stack(losses).mean(), rtol=1e-6, atol=1e-7)
    expected = torch.cat(gradients)
    assert torch.linalg.vector_norm(expected - estimate.grad) / torch.linalg.vector_norm(expected) < 1e-4
    assert standard.updates == merged.updates == 4
    for name, value in standard.discriminator.state_dict().items():
        torch.testing.assert_close(value, merged.discriminator.state_dict()[name], rtol=0, atol=0)
    for one, two in zip(standard.optimizer.state.values(), merged.optimizer.state.values()):
        for key in one:
            torch.testing.assert_close(one[key], two[key], rtol=0, atol=0)

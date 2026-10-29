"""Checks of the isolated quality objective, including silence and padding."""

import math

import pytest
import torch

import evaluate_light
import train_light
from light_quality_loss import QualityObjective


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_features(objective, lengths=(1600, 1921), amplitude=0.08, padded_value=0.0):
    generator = torch.Generator().manual_seed(902)
    maximum = max(lengths)
    target = torch.randn(len(lengths), maximum, generator=generator) * amplitude
    mixture = target.unsqueeze(-1).repeat(1, 1, 8)
    mixture += torch.randn(mixture.shape, generator=generator) * amplitude * 0.3
    for index, length in enumerate(lengths):
        target[index, length:] = padded_value
        mixture[index, length:] = padded_value
    mix_stft, target_stft = objective.build_features(
        mixture, target, n_fft=512, hop_length=160, win_length=320,
        power=0.5, device=torch.device("cpu"), lengths=torch.tensor(lengths))
    return mixture, target, mix_stft, target_stft, [length // 160 + 1 for length in lengths]


@pytest.mark.parametrize("lengths", [(1600, 1600), (1600, 1921)])
def test_features_equal_independent_original_frontend_and_preserve_gain(lengths):
    objective = QualityObjective()
    mixture, target, mix_stft, target_stft, frames = make_features(objective, lengths=lengths,
                                                                amplitude=2.0, padded_value=700.0)
    for index, length in enumerate(lengths):
        expected_mix, expected_target = train_light.build_stft_batch(
            mixture[index:index + 1, :length], target[index:index + 1, :length],
            n_fft=512, hop_length=160, win_length=320, power=0.5, device=torch.device("cpu"))
        torch.testing.assert_close(mix_stft[index:index + 1, :frames[index]], expected_mix)
        torch.testing.assert_close(target_stft[index:index + 1, :, :frames[index]], expected_target)
        assert not mix_stft[index, frames[index]:].count_nonzero()
        assert not target_stft[index, :, frames[index]:].count_nonzero()


def test_identical_estimate_has_nearly_zero_loss_and_detached_stats():
    objective = QualityObjective()
    _, _, _, target, frames = make_features(objective)
    estimate = target.detach().clone().requires_grad_(True)
    loss = objective(estimate, target, frames)
    assert 0 <= loss.item() < 2e-5
    loss.backward()
    assert torch.isfinite(estimate.grad).all()
    stats = objective.stats()
    assert stats["samples"] == 2 and stats["batches"] == 1
    assert stats["total"] == pytest.approx(loss.item())
    assert not objective._component_totals.requires_grad
    objective.reset_stats()
    assert objective.stats() == dict(samples=0, batches=0, spectral=0.0, mrstft=0.0,
                                     waveform=0.0, total=0.0)


@pytest.mark.parametrize("amplitude", [0.0, 1e-12, 0.08, 2.0])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_silent_tiny_and_normal_targets_have_finite_loss_and_gradients(amplitude, sign):
    objective = QualityObjective()
    _, _, _, target, frames = make_features(objective, lengths=(1601,), amplitude=amplitude)
    generator = torch.Generator().manual_seed(43)
    estimate = (target * sign + torch.randn(target.shape, generator=generator) * 0.001).requires_grad_(True)
    loss = objective(estimate, target, frames)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert torch.isfinite(estimate.grad).all()
    assert estimate.grad.count_nonzero() > 0
    stats = objective.stats()
    assert stats["total"] == pytest.approx(stats["spectral"] + 0.1 * stats["mrstft"]
                                          + 0.1 * stats["waveform"], rel=1e-6)


def test_zero_estimate_of_silence_has_zero_loss_and_finite_zero_gradient():
    objective = QualityObjective()
    _, _, _, target, frames = make_features(objective, lengths=(1600,), amplitude=0.0)
    estimate = torch.zeros_like(target, requires_grad=True)
    loss = objective(estimate, target, frames)
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(estimate.grad).all()
    assert not estimate.grad.count_nonzero()


def test_padding_values_and_padding_gradients_are_ignored():
    first, second = QualityObjective(), QualityObjective()
    _, _, _, reference, frames = make_features(first, padded_value=0.0)
    _, _, _, other_reference, other_frames = make_features(second, padded_value=990.0)
    perturbed = (reference * 0.8).detach().requires_grad_(True)
    other = (other_reference * 0.8).detach()
    other[0, :, frames[0]:] = -1000
    other_reference[0, :, frames[0]:] = 1000
    other.requires_grad_(True)
    loss = first(perturbed, reference, frames)
    other_loss = second(other, other_reference, other_frames)
    torch.testing.assert_close(loss, other_loss)
    loss.backward()
    other_loss.backward()
    torch.testing.assert_close(perturbed.grad, other.grad)
    assert not other.grad[0, :, frames[0]:].count_nonzero()


@pytest.mark.parametrize("amplitude", [0.0, 1e-7, 0.1, 2.0])
def test_inverse_agrees_with_evaluator_and_has_finite_gradients(amplitude):
    objective = QualityObjective()
    generator = torch.Generator().manual_seed(151)
    estimate = (torch.randn(2, 2, 11, 257, generator=generator) * amplitude).requires_grad_(True)
    actual = objective.reconstruct(estimate, length=1601)
    expected = evaluate_light.reconstruct_waveform(estimate, length=1601, n_fft=512,
                                                  hop_length=160, win_length=320, power=0.5,
                                                  device=torch.device("cpu"))
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=5e-7)
    actual.square().mean().backward()
    assert torch.isfinite(estimate.grad).all()


def test_original_short_valid_utterance_supports_larger_auxiliary_fft():
    objective = QualityObjective()
    _, _, _, target, frames = make_features(objective, lengths=(321,))
    estimate = (target * 0.7).detach().requires_grad_(True)
    loss = objective(estimate, target, frames)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(estimate.grad).all()


def test_loss_and_features_stay_float32_under_autocast():
    objective = QualityObjective()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        _, _, mix_stft, target, frames = make_features(objective, lengths=(1600,))
        estimate = (target * 0.8).detach().requires_grad_(True)
        loss = objective(estimate, target, frames)
    assert mix_stft.dtype == target.dtype == loss.dtype == torch.float32
    loss.backward()
    assert torch.isfinite(estimate.grad).all()


def test_stats_weight_samples_instead_of_batches():
    objective = QualityObjective()
    _, _, _, first_target, first_frames = make_features(objective, lengths=(1600, 1600))
    first_loss = objective(first_target * 0.8, first_target, first_frames).item()
    _, _, _, second_target, second_frames = make_features(objective, lengths=(1600,))
    second_loss = objective(second_target * 0.4, second_target, second_frames).item()
    stats = objective.stats()
    assert stats["samples"] == 3 and stats["batches"] == 2
    assert stats["total"] == pytest.approx((2 * first_loss + second_loss) / 3)


def test_original_epoch_hooks_train_and_validate_without_changing_the_trainer():
    class ReferenceGain(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = torch.nn.Parameter(torch.tensor(0.7))

        def forward(self, features):
            return features[:, :, :, 0].permute(0, 3, 1, 2) * self.gain

    objective = QualityObjective()
    mixture, target, _, _, _ = make_features(objective)
    batch = dict(mixture=mixture, target=target, lengths=torch.tensor([1600, 1921]),
                 sample_ids=torch.tensor([1, 2]))
    args = train_light.parse_args([])
    args.use_amp = args.model_amp = "no"
    args.grad_accum_steps = 1
    args.grad_clip = 0
    args.mem_log_every_batches = args.malloc_trim_every_batches = args.gc_every_batches = 0
    args.log_perf = "no"
    model = ReferenceGain()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    before = model.gain.detach().clone()
    loss = train_light.run_epoch(
        model, [batch], optimizer, scaler, torch.device("cpu"), [], args, training=True,
        loss_fn=objective, feature_builder=objective.build_features, fail_nonfinite_grad=True)
    assert math.isfinite(loss) and not torch.equal(model.gain.detach(), before)
    assert objective.stats()["samples"] == 2
    objective.reset_stats()
    after = model.gain.detach().clone()
    val_loss = train_light.run_epoch(
        model, [batch], optimizer, scaler, torch.device("cpu"), [], args, training=False,
        loss_fn=objective, feature_builder=objective.build_features, fail_nonfinite_grad=True)
    assert math.isfinite(val_loss)
    torch.testing.assert_close(model.gain.detach(), after, rtol=0, atol=0)
    assert objective.stats()["samples"] == 2


def test_context_and_frontend_mismatches_fail_explicitly():
    objective = QualityObjective()
    with pytest.raises(RuntimeError, match="build_features"):
        objective(torch.zeros(1, 2, 11, 257), torch.zeros(1, 2, 11, 257), [11])
    mixture, target, _, target_stft, frames = make_features(objective, lengths=(1600,))
    with pytest.raises(ValueError, match="frame counts"):
        objective(target_stft, target_stft, [frames[0] - 1])
    with pytest.raises(ValueError, match="frontend differ"):
        objective.build_features(mixture, target, n_fft=512, hop_length=160, win_length=320,
                                 power=0.3, device="cpu", lengths=[1600])
    with pytest.raises(ValueError, match="reflect padding"):
        objective.build_features(mixture, target, n_fft=512, hop_length=160, win_length=320,
                                 power=0.5, device="cpu", lengths=[256])

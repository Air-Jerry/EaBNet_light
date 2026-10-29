"""Frozen-base, padding and gradient checks for the quality enhancement model."""

import io

import pytest
import torch

from light_quality_model import (
    BASE_CANDIDATE, QUALITY_CANDIDATE, EaBNetQuality, build_quality_model,
)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(2)
    torch.manual_seed(71)


def inputs(batch_size=2, frames=7):
    return torch.randn(batch_size, frames, 257, 8, 2) * 0.2


def test_initial_output_exactly_matches_base():
    model = build_quality_model().train()
    mixture = inputs()
    with torch.no_grad():
        expected = model.base(mixture)
        actual = model(mixture)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (2, 2, 7, 257)
    assert not model.base.training
    assert model.refiner.training
    assert not model.refiner_causal
    assert QUALITY_CANDIDATE != BASE_CANDIDATE
    assert 1_500_000 < sum(p.numel() for p in model.refiner.parameters()) < 2_500_000


def test_training_updates_refiner_only_and_gradient_reaches_trunk():
    model = EaBNetQuality().train()
    mixture = inputs(batch_size=1)
    target = torch.randn(1, 2, 7, 257) * 0.1
    original_base = {name: value.clone() for name, value in model.base.state_dict().items()}
    original_refiner = {name: value.clone() for name, value in model.refiner.state_dict().items()}
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=0.001)

    first_loss = (model(mixture) - target).square().mean()
    first_loss.backward()
    assert model.refiner.head.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.base.parameters())
    assert all(not parameter.requires_grad for parameter in model.base.parameters())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    second_loss = (model(mixture) - target).square().mean()
    second_loss.backward()
    trunk_gradient = model.refiner.encoder[0].conv.weight.grad
    assert trunk_gradient is not None
    assert torch.isfinite(trunk_gradient).all()
    assert trunk_gradient.abs().sum() > 0
    recurrent_gradient = model.refiner.context.weight_ih_l0_reverse.grad
    assert recurrent_gradient is not None
    assert torch.isfinite(recurrent_gradient).all()
    assert recurrent_gradient.abs().sum() > 0
    optimizer.step()

    for name, value in model.base.state_dict().items():
        torch.testing.assert_close(value, original_base[name], rtol=0, atol=0)
    assert any(not torch.equal(value, original_refiner[name])
               for name, value in model.refiner.state_dict().items())
    assert not model.base.training
    assert all(not module.training for module in model.base.modules())


def test_input_amplitude_scales_residual_and_silence_remains_zero():
    model = EaBNetQuality().eval()
    with torch.no_grad():
        model.refiner.head.weight.zero_()
        model.refiner.head.bias.fill_(0.5)
        mixture = inputs(batch_size=1, frames=5)
        for gain in (0.0, 0.01, 1.0, 10.0):
            scaled = mixture * gain
            baseline = model.base(scaled)
            actual = model(scaled)
            expected_residual = 0.5 * torch.linalg.vector_norm(scaled, dim=-1).mean(dim=-1).unsqueeze(1)
            torch.testing.assert_close(actual - baseline, expected_residual.expand_as(actual), rtol=1e-5, atol=1e-7)
            if gain == 0:
                assert torch.count_nonzero(actual) == 0


def test_variable_length_outputs_ignore_padded_input():
    model = EaBNetQuality().eval()
    with torch.no_grad():
        model.refiner.head.weight.normal_(0, 0.02)
        model.refiner.head.bias.normal_(0, 0.02)
    mixture = inputs(frames=9)
    contaminated = mixture.clone()
    contaminated[0, 5:] = torch.randn_like(contaminated[0, 5:]) * 100
    with torch.no_grad():
        expected_short = model(mixture[:1, :5])
        expected_long = model(mixture[1:])
        actual = model(contaminated, frame_lengths=torch.tensor([5, 9]))
    torch.testing.assert_close(actual[:1, :, :5], expected_short, rtol=0, atol=0)
    torch.testing.assert_close(actual[1:], expected_long, rtol=0, atol=0)
    assert torch.count_nonzero(actual[0, :, 5:]) == 0


def test_equal_short_lengths_use_true_frames_and_keep_gradients():
    model = EaBNetQuality().train()
    mixture = inputs(frames=8)
    expected = model(mixture[:, :5])
    actual = model(mixture, frame_lengths=[5, 5])
    torch.testing.assert_close(actual[:, :, :5], expected, rtol=0, atol=0)
    assert torch.count_nonzero(actual[:, :, 5:]) == 0
    actual.square().mean().backward()
    assert model.refiner.head.weight.grad is not None
    assert torch.isfinite(model.refiner.head.weight.grad).all()


def test_variable_length_training_keeps_gradients_and_excludes_padding():
    model = EaBNetQuality().train()
    with torch.no_grad():
        model.refiner.head.weight.normal_(0, 0.02)
    mixture = inputs(frames=8).requires_grad_()
    estimate = model(mixture, frame_lengths=[5, 8])
    loss = estimate[0, :, :5].square().mean() + estimate[1].square().mean()
    loss.backward()
    assert torch.isfinite(model.refiner.context.weight_ih_l0_reverse.grad).all()
    assert mixture.grad[0, :5].abs().sum() > 0
    assert mixture.grad[1].abs().sum() > 0
    assert torch.count_nonzero(mixture.grad[0, 5:]) == 0


def test_full_state_dict_round_trip_includes_frozen_base():
    model = EaBNetQuality().eval()
    with torch.no_grad():
        model.refiner.head.weight.normal_(0, 0.02)
    mixture = inputs(batch_size=1, frames=5)
    with torch.no_grad():
        expected = model(mixture)
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    state = torch.load(checkpoint, weights_only=True)
    assert any(name.startswith("base.") for name in state)
    assert any(name.startswith("refiner.") for name in state)
    restored = EaBNetQuality().eval()
    restored.load_state_dict(state, strict=True)
    with torch.no_grad():
        actual = restored(mixture)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_outer_autocast_keeps_frozen_base_fp32():
    model = EaBNetQuality().train()
    observed_dtypes = []
    handle = model.base.register_forward_hook(lambda module, args, output: observed_dtypes.append(output.dtype))
    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            estimate = model(inputs(batch_size=1, frames=5))
    finally:
        handle.remove()
    assert observed_dtypes == [torch.float32]
    assert estimate.dtype == torch.float32
    assert torch.isfinite(estimate).all()


@pytest.mark.parametrize("lengths", [[0, 7], [8, 7], [5], [[5, 7]], [5.0, 7.0], [True, True]])
def test_invalid_frame_lengths_rejected(lengths):
    with pytest.raises(ValueError, match="frame_lengths"):
        EaBNetQuality()(inputs(), frame_lengths=lengths)


@pytest.mark.parametrize("mixture", [
    torch.randn(1, 5, 257, 2),
    torch.randn(1, 5, 256, 8, 2),
    torch.randn(1, 5, 257, 7, 2),
    torch.zeros(1, 5, 257, 8, 2, dtype=torch.int64),
    torch.zeros(0, 5, 257, 8, 2),
    torch.zeros(1, 0, 257, 8, 2),
])
def test_invalid_stft_shape_or_dtype_rejected(mixture):
    with pytest.raises(ValueError):
        EaBNetQuality()(mixture)

"""Causality, finite gradients and full waveform streaming for the fast model."""

import pytest
import torch
import torch.nn.functional as F

from light_causal_fast_model import CausalUNetSpatial, ActivationLayerNorm
from light_causal_stream import CausalWaveformStream
from light_quality_loss import QualityObjective
from train_light import build_stft_batch


@pytest.fixture
def model():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    result = CausalUNetSpatial(hidden=32, ffn=64, layers=2, context=5)
    with torch.no_grad():
        result.beamformer.weight.normal_(std=0.01)
        result.beamformer.bias.normal_(std=0.01)
    return result


@pytest.mark.parametrize("training", [True, False])
def test_future_independence_with_nonzero_beamformer(model, training):
    model.train(training)
    value = torch.randn(1, 12, 257, 8, 2) * 0.1
    changed = value.clone()
    changed[:, 7:] = torch.randn_like(changed[:, 7:]) * 3
    torch.testing.assert_close(model(value)[:, :, :7], model(changed)[:, :, :7], rtol=0, atol=0)
    value.requires_grad_()
    model(value)[:, :, :7].square().mean().backward()
    assert torch.count_nonzero(value.grad[:, 7:]) == 0
    assert torch.count_nonzero(value.grad[:, :7]) > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_arbitrary_frame_chunks_and_bounded_state(model):
    model.eval()
    value = torch.randn(1, 31, 257, 8, 2) * 0.1
    with torch.inference_mode():
        expected = model(value)
        outputs, state = [], None
        for first, last in zip((0, 1, 8, 13), (1, 8, 13, 31)):
            output, state = model.forward_stream(value[:, first:last], state)
            outputs.append(output)
            for layer in state["middle"]:
                assert layer["k"].shape[-2] <= 4 and layer["v"].shape[-2] <= 4
            for memory in (*state["encoder"], *state["decoder"], state["stem"]):
                assert memory.shape[2] == 2
        torch.testing.assert_close(torch.cat(outputs, dim=2), expected, rtol=3e-5, atol=2e-6)


@pytest.mark.parametrize("length", [257, 320, 1327])
def test_full_audio_stream_matches_training_frontend(model, length):
    model.eval()
    mixture = torch.randn(1, length, 8) * 0.01
    features, _ = build_stft_batch(mixture, mixture[..., 0], 512, 160, 320, 0.5, torch.device("cpu"))
    with torch.inference_mode():
        expected = QualityObjective().reconstruct(model(features), length)
        stream = CausalWaveformStream(model)
        stream.warmup()
        assert stream.state is None and stream.total_samples == 0
        actual = torch.cat([stream.push(mixture[:, :31]), stream.push(mixture[:, 31:410]),
                            stream.push(mixture[:, 410:]), stream.flush()], dim=1)
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-6)


def test_zero_beamformer_is_exact_current_mixture(model):
    model.beamformer.weight.data.zero_()
    model.beamformer.bias.data.zero_()
    value = torch.randn(1, 7, 257, 8, 2) * 0.1
    torch.testing.assert_close(model(value), value[..., 0, :].permute(0, 3, 1, 2), rtol=0, atol=0)


def test_normalization_preserves_bfloat16_and_uses_fp32_parameters():
    layer = ActivationLayerNorm(32)
    with torch.no_grad():
        layer.weight.uniform_(0.5, 1.5)
        layer.bias.normal_(std=0.1)
    value = torch.randn(2, 3, 32).bfloat16().requires_grad_()
    result = layer(value)
    assert result.dtype == torch.bfloat16 and layer.weight.dtype == torch.float32
    reference = F.layer_norm(value.float(), (32,), layer.weight, layer.bias).bfloat16()
    torch.testing.assert_close(result, reference, rtol=0, atol=0)
    result.float().square().mean().backward()
    assert torch.isfinite(value.grad).all() and torch.isfinite(layer.weight.grad).all()

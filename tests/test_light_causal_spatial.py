"""Nonzero-head future independence and actual bounded waveform streaming."""

import pytest
import torch

from light_causal_spatial_model import CausalSpatialEnhancer
from light_causal_stream import CausalWaveformStream
from light_quality_loss import QualityObjective
from train_light import build_stft_batch


@pytest.fixture
def model():
    torch.set_num_threads(2)
    torch.manual_seed(51)
    result = CausalSpatialEnhancer(hidden=16, ffn=32, layers=2, context=5)
    with torch.no_grad():
        result.decoder.weight.normal_(std=0.01)
        result.decoder.bias.normal_(std=0.01)
    return result


@pytest.mark.parametrize("training", [False, True])
def test_no_future_frames_even_with_nonzero_head(model, training):
    model.train(training)
    value = torch.randn(1, 11, 257, 8, 2) * 0.1
    changed = value.clone()
    changed[:, 6:] = torch.randn_like(changed[:, 6:]) * 7
    first, second = model(value), model(changed)
    torch.testing.assert_close(first[:, :, :6], second[:, :, :6], rtol=0, atol=0)
    differentiable = value.requires_grad_()
    model(differentiable)[:, :, :6].square().sum().backward()
    assert torch.count_nonzero(differentiable.grad[:, 6:]) == 0
    assert torch.count_nonzero(differentiable.grad[:, :6]) > 0


def test_streaming_frames_match_full_and_caches_do_not_grow(model):
    model.eval()
    value = torch.randn(1, 37, 257, 8, 2) * 0.1
    with torch.inference_mode():
        expected = model(value)
        state, outputs = None, []
        for start, end in zip((0, 1, 8, 10, 23), (1, 8, 10, 23, 37)):
            output, state = model.forward_stream(value[:, start:end], state)
            outputs.append(output)
            for layer in state["layers"]:
                assert layer["k"].shape[-2] <= 4
                assert layer["v"].shape[-2] <= 4
                assert all(history.shape[-1] == 2 for history in layer["convolutions"].values())
        torch.testing.assert_close(torch.cat(outputs, dim=2), expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("length", [257, 319, 320, 321, 1280, 1327])
def test_waveform_stream_matches_original_frontend_and_full_model(model, length):
    model.eval()
    mixture = torch.randn(1, length, 8) * 0.01
    features, _ = build_stft_batch(mixture, mixture[..., 0], n_fft=512,
        hop_length=160, win_length=320, power=0.5, device=torch.device("cpu"))
    with torch.inference_mode():
        expected = QualityObjective().reconstruct(model(features), length)
        stream = CausalWaveformStream(model)
        outputs = [stream.push(mixture[:, :31]), stream.push(mixture[:, 31:417]),
                   stream.push(mixture[:, 417:909]), stream.push(mixture[:, 909:]), stream.flush()]
    actual = torch.cat(outputs, dim=1)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=3e-6)
    assert stream.tail.shape[1] <= 320 and stream.pending.shape[1] < 160
    with pytest.raises(RuntimeError, match="flushed"):
        stream.push(mixture[:, :160])


def test_zero_decoder_initializes_to_current_mixture_without_old_backbone():
    model = CausalSpatialEnhancer(hidden=16, ffn=32, layers=1, context=4).eval()
    assert not any(name.startswith("base.") for name in model.state_dict())
    value = torch.randn(1, 5, 257, 8, 2)
    with torch.no_grad():
        torch.testing.assert_close(model(value), value[..., 0, :].permute(0, 3, 1, 2), rtol=0, atol=0)


def test_warmup_does_not_contaminate_live_stream_state_or_weights(model):
    model.eval()
    original = {k: v.clone() for k, v in model.state_dict().items()}
    stream = CausalWaveformStream(model)
    stream.warmup()
    assert stream.state is None and stream.pending is None and stream.total_samples == 0
    assert all(torch.equal(value, original[name]) for name, value in model.state_dict().items())
    stream.push(torch.randn(1, 160, 8) * 0.01)
    with pytest.raises(RuntimeError, match="before receiving"):
        stream.warmup()

"""Causality, exact warm-start and bounded state of the experimental band model."""

import pytest
import torch

from experiments.causal_band_beamformer import CausalBandBeamformer
from light_causal_global_model import CausalGlobalSpatial
from light_causal_stream import CausalWaveformStream
from light_quality_loss import QualityObjective
from train_light import build_stft_batch


SMALL = dict(hidden=32, ffn=64, layers=2, context=5, global_hidden=16)


@pytest.fixture
def model():
    torch.set_num_threads(2)
    torch.manual_seed(23)
    model = CausalBandBeamformer(band_hidden=16, **SMALL)
    model.beamformer.weight.data.normal_(std=.01)
    model.layers[-1].output.weight.data.normal_(std=.01)
    model.band_output.weight.data.normal_(std=.01)
    return model


def test_band_zero_residual_preserves_source_weights_output_and_crop_rng():
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    source = CausalGlobalSpatial(**SMALL)
    source.beamformer.weight.data.normal_(std=.01)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(1234)
    model = CausalBandBeamformer(band_hidden=16, **SMALL)
    model.beamformer.weight.data.normal_(std=.01)
    torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
    target = model.state_dict()
    target.update(source.state_dict())
    model.load_state_dict(target, strict=True)
    assert all(parameter.requires_grad for parameter in model.parameters())
    source.eval(); model.eval()
    with torch.inference_mode():
        value = torch.randn(1, 9, 257, 8, 2) * .02
        torch.testing.assert_close(model(value), source(value), rtol=0, atol=0)


@pytest.mark.parametrize('training', [False, True])
def test_band_future_frames_have_zero_effect_and_zero_gradient(model, training):
    model.train(training)
    value = (torch.randn(1, 15, 257, 8, 2) * .02).requires_grad_()
    changed = value.detach().clone(); changed[:, 9:] *= 11
    output = model(value)
    torch.testing.assert_close(output[:, :, :9], model(changed)[:, :, :9], rtol=0, atol=0)
    output[:, :, :9].square().mean().backward()
    assert torch.count_nonzero(value.grad[:, 9:]) == 0
    assert torch.count_nonzero(value.grad[:, :9]) > 0
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_band_nonzero_recurrence_chunking_and_fixed_cache(model):
    model.eval()
    value = torch.randn(1, 23, 257, 8, 2) * .02
    with torch.inference_mode():
        expected = model(value)
        state = None; parts = []
        for start, end in ((0, 1), (1, 7), (7, 9), (9, 23)):
            part, state = model.forward_stream(value[:, start:end], state)
            parts.append(part)
            assert state['band_hidden'].shape == (2, 257, 16)
        torch.testing.assert_close(torch.cat(parts, 2), expected, rtol=1e-4, atol=3e-6)


def test_band_waveform_stream_preserves_frontend(model):
    model.eval()
    mixture = torch.randn(1, 1481, 8) * .02
    with torch.inference_mode():
        features, _ = build_stft_batch(mixture, mixture[:, :, 0], 512, 160, 320, .5, torch.device('cpu'))
        expected = QualityObjective().reconstruct(model(features), mixture.shape[1])
        stream = CausalWaveformStream(model)
        stream.warmup()
        actual = torch.cat((stream.push(mixture[:, :311]), stream.push(mixture[:, 311:]), stream.flush()), 1)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-5)

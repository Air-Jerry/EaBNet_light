"""Nonzero full-frequency mapping: causality, fixed caches and waveform identity."""

import pytest
import torch

from experiments.causal_tcn_mapping import CausalTCNMapping
from light_causal_recurrent65_model import CausalRecurrentUNet65
from light_causal_stream import CausalWaveformStream
from light_quality_loss import QualityObjective
from train_light import build_stft_batch


SMALL = dict(hidden=32, ffn=64, layers=2, global_hidden=16, context=5)


@pytest.fixture
def model():
    torch.set_num_threads(2)
    torch.manual_seed(82)
    model = CausalTCNMapping(mapping_channels=8, **SMALL)
    model.beamformer.weight.data.normal_(std=.01)
    model.mapping_output.weight.data.normal_(std=.01)
    return model


def test_zero_mapping_preserves_source_and_crop_rng():
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    source = CausalRecurrentUNet65(**SMALL)
    source.beamformer.weight.data.normal_(std=.01)
    expected = torch.get_rng_state()
    torch.manual_seed(1234)
    changed = CausalTCNMapping(mapping_channels=8, **SMALL)
    changed.beamformer.weight.data.normal_(std=.01)
    torch.testing.assert_close(torch.get_rng_state(), expected, rtol=0, atol=0)
    values = changed.state_dict(); values.update(source.state_dict())
    changed.load_state_dict(values, strict=True)
    assert all(p.requires_grad for p in changed.parameters())
    source.eval(); changed.eval()
    with torch.inference_mode():
        x = torch.randn(1, 11, 257, 8, 2) * .02
        torch.testing.assert_close(changed(x), source(x), rtol=0, atol=0)


@pytest.mark.parametrize('training', [False, True])
def test_nonzero_mapping_has_no_future_gradient(model, training):
    model.train(training)
    x = (torch.randn(1, 17, 257, 8, 2) * .02).requires_grad_()
    altered = x.detach().clone(); altered[:, 9:] *= 17
    y = model(x)
    torch.testing.assert_close(y[:, :, :9], model(altered)[:, :, :9], rtol=0, atol=0)
    y[:, :, :9].square().mean().backward()
    assert torch.count_nonzero(x.grad[:, 9:]) == 0
    assert torch.count_nonzero(x.grad[:, :9]) > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert torch.count_nonzero(model.mapping_output.weight.grad) > 0


def test_mapping_stream_chunks_and_owned_caches_are_bounded(model):
    model.eval()
    with torch.inference_mode():
        x = torch.randn(1, 97, 257, 8, 2) * .02
        full = model(x); state = None; outputs = []
        for start, stop in [(0, 1), (1, 3), (3, 35), (35, 63), (63, 97)]:
            y, state = model.forward_stream(x[:, start:stop], state); outputs.append(y)
            for layer, history in zip(model.mapping_blocks, state['mapping']):
                assert history.shape == (1, 8, layer.history_frames, 257)
                assert history.untyped_storage().nbytes() == 8 * layer.history_frames * 257 * 4
        torch.testing.assert_close(torch.cat(outputs, 2), full, rtol=1e-4, atol=1e-6)
        replay, _ = model.forward_stream(x, None)
        torch.testing.assert_close(replay, full, rtol=1e-4, atol=1e-6)


def test_waveform_stream_matches_nonzero_mapping(model):
    model.eval()
    with torch.inference_mode():
        waveform = torch.randn(1, 1617, 8) * .02
        features, _ = build_stft_batch(waveform, waveform[:, :, 0], 512, 160, 320, .5, torch.device('cpu'))
        expected = QualityObjective().reconstruct(model(features), waveform.shape[1])
        stream = CausalWaveformStream(model); stream.warmup()
        parts = [stream.push(waveform[:, i:i+160]) for i in range(0, waveform.shape[1], 160)]
        actual = torch.cat([*parts, stream.flush()], dim=1)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)

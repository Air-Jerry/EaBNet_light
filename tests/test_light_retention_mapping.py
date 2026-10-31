"""Canonical batch/stream equations, causal gradients and owned bounded matrix state."""

import pytest
import torch

from experiments.causal_frequency_attention import CausalFrequencyAttentionTCN
from experiments.causal_retention_mapping import CausalAttentionRetentionTCN, CausalRetention
from light_causal_stream import CausalWaveformStream
from light_quality_loss import QualityObjective
from train_light import build_stft_batch


SMALL = dict(hidden=32, ffn=64, layers=2, global_hidden=16, context=5, mapping_channels=8)


def test_canonical_retention_matches_single_frames_and_chunk_boundaries():
    torch.set_num_threads(2)
    torch.manual_seed(88)
    layer = CausalRetention(16).eval()
    layer.output.weight.data.normal_(std=.03)
    value = torch.randn(2, 16, 201, 9)
    with torch.inference_mode():
        expected = layer(value)
        state = None
        parts = []
        for start, end in ((0, 1), (1, 8), (8, 72), (72, 73), (73, 201)):
            part, state = layer.process(value[:, :, start:end], state, cache=True)
            parts.append(part)
            assert state['matrix'].shape == (18, 4, 4, 4)
            assert state['matrix'].untyped_storage().nbytes() == 18 * 4 * 4 * 4 * 4
            assert state['count'].untyped_storage().nbytes() == 16
        torch.testing.assert_close(torch.cat(parts, 2), expected, rtol=1e-4, atol=2e-6)
        state = None
        frames = []
        for index in range(201):
            part, state = layer.process(value[:, :, index:index + 1], state, cache=True)
            frames.append(part)
        torch.testing.assert_close(torch.cat(frames, 2), expected, rtol=1e-4, atol=2e-6)
        with pytest.raises(ValueError, match='state'):
            layer.process(value[:1, :, :1], state, cache=True)


def test_canonical_retention_has_no_future_dependency_or_nonfinite_gradients():
    torch.set_num_threads(2)
    torch.manual_seed(89)
    layer = CausalRetention(16)
    layer.output.weight.data.normal_(std=.03)
    value = torch.randn(1, 16, 145, 7, requires_grad=True)
    changed = value.detach().clone()
    changed[:, :, 71:] *= 19
    expected = layer(value)
    torch.testing.assert_close(expected[:, :, :71], layer(changed)[:, :, :71], rtol=0, atol=0)
    expected[:, :, :71].square().mean().backward()
    assert torch.count_nonzero(value.grad[:, :, 71:]) == 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
    assert torch.count_nonzero(layer.qkv.weight.grad) > 0


def test_zero_retention_preserves_attention_source_and_crop_rng():
    torch.set_num_threads(2)
    torch.manual_seed(90)
    old = CausalFrequencyAttentionTCN(**SMALL)
    rng = torch.get_rng_state()
    torch.manual_seed(90)
    changed = CausalAttentionRetentionTCN(**SMALL)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    old.beamformer.weight.data.normal_(std=.01)
    old.mapping_output.weight.data.normal_(std=.01)
    weights = changed.state_dict()
    weights.update(old.state_dict())
    changed.load_state_dict(weights, strict=True)
    with torch.inference_mode():
        value = torch.randn(1, 17, 257, 8, 2) * .02
        torch.testing.assert_close(old.eval()(value), changed.eval()(value), rtol=0, atol=0)


def test_nonzero_retention_model_matches_streaming_and_waveform_reconstruction():
    torch.set_num_threads(2)
    torch.manual_seed(91)
    model = CausalAttentionRetentionTCN(**SMALL).eval()
    for weight in (model.beamformer.weight, model.mapping_output.weight,
                   model.mapping_input.frequency_output.weight, model.retention.output.weight):
        weight.data.normal_(std=.01)
    with torch.inference_mode():
        value = torch.randn(1, 145, 257, 8, 2) * .02
        expected = model(value)
        pieces = []
        state = None
        for start, end in ((0, 1), (1, 31), (31, 95), (95, 145)):
            part, state = model.forward_stream(value[:, start:end], state)
            pieces.append(part)
            assert state['retention']['matrix'].shape == (257, 4, 2, 2)
            assert state['retention']['matrix'].untyped_storage().nbytes() == 257 * 4 * 2 * 2 * 4
        torch.testing.assert_close(torch.cat(pieces, 2), expected, rtol=1e-4, atol=2e-6)
        changed = value.clone()
        changed[:, 71:] *= 13
        torch.testing.assert_close(expected[:, :, :71], model(changed)[:, :, :71], rtol=0, atol=0)
        wave = torch.randn(1, 22137, 8) * .02
        features, _ = build_stft_batch(wave, wave[:, :, 0], 512, 160, 320, .5, torch.device('cpu'))
        full = QualityObjective().reconstruct(model(features), wave.shape[1])
        stream = CausalWaveformStream(model)
        stream.warmup()
        outputs = [stream.push(wave[:, start:start + 160]) for start in range(0, wave.shape[1], 160)]
        torch.testing.assert_close(torch.cat([*outputs, stream.flush()], 1), full, rtol=1e-4, atol=2e-6)

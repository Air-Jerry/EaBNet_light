"""Frame-local global frequency support, preserved source and bounded streaming."""

import torch

from experiments.causal_frequency_attention import CausalFrequencyAttentionTCN, FrequencyAttentionInput
from experiments.causal_tcn_mapping import CausalTCNMapping
from light_causal_stream import CausalWaveformStream
from light_quality_loss import QualityObjective
from train_light import build_stft_batch


SMALL = dict(hidden=32, ffn=64, layers=2, global_hidden=16, context=5, mapping_channels=8)


def test_global_frequency_support_cannot_attend_to_other_time_frames():
    torch.set_num_threads(2); torch.manual_seed(89)
    layer = FrequencyAttentionInput(3, 4)
    layer.frequency_output.weight.data.normal_(std=.01)
    value = torch.randn(1, 3, 3, 257, requires_grad=True)
    layer(value)[:, :, 0, 0].sum().backward()
    assert value.grad[:, :, 0, 100:].abs().max() > 1e-9
    assert torch.count_nonzero(value.grad[:, :, 1:]) == 0


def test_zero_attention_preserves_source_and_crop_rng():
    torch.set_num_threads(2); torch.manual_seed(1234)
    old = CausalTCNMapping(**SMALL); rng = torch.get_rng_state()
    torch.manual_seed(1234); changed = CausalFrequencyAttentionTCN(**SMALL)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    old.beamformer.weight.data.normal_(std=.01); old.mapping_output.weight.data.normal_(std=.01)
    state = changed.state_dict(); state.update(old.state_dict()); changed.load_state_dict(state, strict=True)
    with torch.inference_mode():
        old.eval(); changed.eval(); value = torch.randn(1, 17, 257, 8, 2) * .02
        torch.testing.assert_close(changed(value), old(value), rtol=0, atol=0)


def test_nonzero_attention_and_source_have_zero_future_gradients():
    torch.set_num_threads(2); torch.manual_seed(97)
    model = CausalFrequencyAttentionTCN(**SMALL)
    model.beamformer.weight.data.normal_(std=.01); model.mapping_output.weight.data.normal_(std=.01)
    model.mapping_input.frequency_output.weight.data.normal_(std=.01)
    value = (torch.randn(1, 17, 257, 8, 2) * .02).requires_grad_()
    other = value.detach().clone(); other[:, 9:] *= 11
    estimate = model(value)
    torch.testing.assert_close(estimate[:, :, :9], model(other)[:, :, :9], rtol=0, atol=0)
    estimate[:, :, :9].square().mean().backward()
    assert torch.count_nonzero(value.grad[:, 9:]) == 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert torch.count_nonzero(model.mapping_input.frequency_qkv.weight.grad) > 0


def test_nonzero_attention_matches_waveform_stream_and_fixed_histories():
    torch.set_num_threads(2); torch.manual_seed(101)
    model = CausalFrequencyAttentionTCN(**SMALL).eval()
    model.beamformer.weight.data.normal_(std=.01); model.mapping_output.weight.data.normal_(std=.01)
    model.mapping_input.frequency_output.weight.data.normal_(std=.01)
    with torch.inference_mode():
        value = torch.randn(1, 97, 257, 8, 2) * .02
        full = model(value); parts = []; state = None
        for start, end in ((0, 1), (1, 3), (3, 35), (35, 63), (63, 97)):
            part, state = model.forward_stream(value[:, start:end], state); parts.append(part)
            for block, history in zip(model.mapping_blocks, state['mapping']):
                assert history.shape == (1, 8, block.history_frames, 257)
                assert history.untyped_storage().nbytes() == 8 * block.history_frames * 257 * 4
        torch.testing.assert_close(torch.cat(parts, 2), full, rtol=1e-4, atol=1e-6)
        wave = torch.randn(1, 1617, 8) * .02
        features, _ = build_stft_batch(wave, wave[:, :, 0], 512, 160, 320, .5, torch.device('cpu'))
        expected = QualityObjective().reconstruct(model(features), 1617)
        stream = CausalWaveformStream(model); stream.warmup()
        outputs = [stream.push(wave[:, index:index + 160]) for index in range(0, 1617, 160)]
        torch.testing.assert_close(torch.cat([*outputs, stream.flush()], 1), expected, rtol=1e-4, atol=1e-6)

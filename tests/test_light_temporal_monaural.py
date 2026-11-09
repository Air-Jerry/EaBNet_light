"""Nonzero warm inheritance, learned causal attention and fixed tensor-only stream memory."""
import pytest
import torch

from experiments.causal_frame_graph import tensor_leaves
from experiments.causal_temporal_monaural import CausalTemporalMonaural, PastFrameAttention
from light_candidate_runtime import build_model_candidate, candidate_metadata


def models():
    torch.set_num_threads(2); torch.manual_seed(1234)
    source = build_model_candidate('causal_decoder_monaural_spectral_v1')
    with torch.no_grad():
        for module in (source.base.base.beamformer, source.base.base.mapping_output,
                       source.base.output, source.output):
            module.weight.normal_(std=.01)
    model = CausalTemporalMonaural(build_model_candidate('causal_wide_conditioned_subband_v1'))
    model.load_warm(source.state_dict())
    return source, model


def test_complete_warm_output_and_registered_metadata_remain_unchanged():
    metadata = candidate_metadata('causal_decoder_monaural_spectral_v1')
    source, model = models(); source.eval(); model.eval()
    with torch.no_grad():
        value = torch.randn(1, 131, 257, 8, 2) * .1
        assert torch.equal(source(value), model(value))
    assert sum(p.numel() for p in model.parameters()) == 6105122
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 3610832
    assert metadata == candidate_metadata('causal_decoder_monaural_spectral_v1')
    incomplete = dict(source.state_dict()); incomplete.pop('output.bias')
    with pytest.raises(ValueError, match='complete'): model.load_warm(incomplete)


def test_attention_actual_adam_learning_and_no_future_gradient():
    _, model = models(); model.train()
    frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    before = model.temporal_memory.qkv.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        estimate = model(torch.randn(1, 17, 257, 8, 2) * .1)
        (estimate - torch.randn_like(estimate) * .05).square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        optimizer.step()
    assert not torch.equal(before, model.temporal_memory.qkv.weight)
    assert all(torch.equal(value, dict(model.named_parameters())[name]) for name, value in frozen.items())
    assert {int(values['step']) for values in optimizer.state.values()} == {3}
    assert all(torch.isfinite(value).all() for values in optimizer.state.values() for value in values.values())
    model.eval()
    value = (torch.randn(1, 131, 257, 8, 2) * .1).requires_grad_()
    full = model(value)
    derivative, = torch.autograd.grad(full[:, :, :65].square().mean(), value)
    assert torch.count_nonzero(derivative[:, 65:]) == 0
    with torch.no_grad():
        changed = value.clone(); changed[:, 65:] *= 11
        assert torch.equal(full[:, :, :65], model(changed)[:, :, :65])


def test_nonzero_attention_matches_chunks_with_fixed_independent_tensor_caches():
    _, model = models(); model.eval()
    with torch.no_grad():
        model.temporal_memory.output.weight.normal_(std=.01)
        value = torch.randn(1, 259, 257, 8, 2) * .1
        complete = model(value)
        state = None; parts = []; sizes = set()
        for start, end in ((0, 1), (1, 17), (17, 128), (128, 129), (129, 259)):
            part, state = model.forward_stream(value[:, start:end], state)
            parts.append(part)
            sizes.add(sum(t.untyped_storage().nbytes() for t in tensor_leaves(state)))
        assert sizes == {6349199}
        torch.testing.assert_close(torch.cat(parts, 2), complete, rtol=1e-4, atol=2e-6)
        assert torch.all(state['temporal_memory']['valid'])
        before = [t.clone() for t in tensor_leaves(state)]
        _, later = model.forward_stream(value[:, :1], state)
        assert all(torch.equal(a, b) for a, b in zip(before, tensor_leaves(state)))
        pointers = lambda s: {t.untyped_storage().data_ptr() for t in tensor_leaves(s)}
        assert pointers(state).isdisjoint(pointers(later))
        _, independent = model.forward_stream(value[:, :1])
        assert pointers(state).isdisjoint(pointers(independent))
        bad = dict(state, temporal_memory=dict(state['temporal_memory'], valid=torch.zeros(1)))
        with pytest.raises(ValueError, match='validity'): model.forward_stream(value[:, :1], bad)


def test_attention_context_excludes_frames_older_than_the_fixed_window():
    torch.manual_seed(1234); memory = PastFrameAttention().eval()
    with torch.no_grad():
        memory.output.weight.normal_(std=.01)
        value = torch.randn(1, 260, 816)
        expected, _ = memory.process(value)
        changed = value.clone(); changed[:, :132] *= 10
        actual, _ = memory.process(changed)
        assert torch.equal(actual[:, 259:], expected[:, 259:])


def test_nonfinite_warm_state_is_rejected_before_parameters_change():
    source, model = models()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    invalid = dict(source.state_dict())
    invalid['output.weight'] = torch.full_like(invalid['output.weight'], float('nan'))
    with pytest.raises(ValueError, match='finite'): model.load_warm(invalid)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())


def test_active_attention_matches_the_original_full6s_waveform_frontend_and_stream():
    from evaluate_light import build_stft_batch, reconstruct_waveform
    from light_causal_stream import CausalWaveformStream
    _, model = models(); model.eval()
    with torch.no_grad(): model.temporal_memory.output.weight.normal_(std=.01)
    t = torch.arange(96000, dtype=torch.float32) / 16000
    speech = .1 * (torch.sin(2 * torch.pi * 143 * t) + .2 * torch.sin(2 * torch.pi * 286 * t))
    wave = speech[None, :, None] + torch.randn(1, 96000, 8) * .02
    frontend = dict(n_fft=512, hop_length=160, win_length=320, power=.5, device=torch.device('cpu'))
    with torch.inference_mode():
        expected = reconstruct_waveform(model(build_stft_batch(wave, **frontend)), length=96000, **frontend)
        stream = CausalWaveformStream(model); stream.warmup()
        assert stream.state is None and stream.total_samples == stream.emitted_samples == 0
        parts = [stream.push(wave[:, start:start+160]) for start in range(0, 96000, 160)]
        parts.append(stream.flush())
        actual = torch.cat(parts, 1)
        assert actual.shape == expected.shape == (1, 96000)
        assert (actual - expected).abs().max() < 1e-5
        assert sum(t.untyped_storage().nbytes() for t in tensor_leaves(stream.state)) == 6349199

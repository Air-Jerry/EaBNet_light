"""Nonzero warm inheritance, effective new capacity and causal owned states."""
import copy

import pytest
import torch

from experiments.causal_capacity_monaural import CausalCapacityMonaural
from experiments.causal_frame_graph import tensor_leaves
from light_candidate_runtime import build_model_candidate, candidate_metadata


def models():
    torch.set_num_threads(2)
    torch.manual_seed(381)
    source = build_model_candidate('causal_decoder_monaural_spectral_v1')
    with torch.no_grad():
        for module in (source.base.base.beamformer, source.base.output, source.output):
            module.weight.normal_(std=.01)
    capacity = CausalCapacityMonaural(copy.deepcopy(source.base))
    capacity.load_warm(source.state_dict())
    return source, capacity


def test_nonzero_complete_source_and_registered_metadata_are_preserved():
    metadata = candidate_metadata('causal_decoder_monaural_spectral_v1')
    source, capacity = models()
    source.eval()
    capacity.eval()
    value = torch.randn(1, 131, 257, 8, 2)*.1
    with torch.no_grad():
        assert torch.equal(source(value), capacity(value))
    assert sum(p.numel() for p in capacity.parameters()) == 6720674
    assert sum(p.numel() for p in capacity.parameters() if p.requires_grad) == 4226384
    assert capacity.capacity_recurrent.bidirectional is False
    assert metadata == candidate_metadata('causal_decoder_monaural_spectral_v1')
    assert all(torch.equal(weight, capacity.state_dict()[name]) for name, weight in source.state_dict().items())


def test_capacity_learns_with_finite_adam_and_unchanged_frozen_encoder():
    _, capacity = models()
    frozen = {name: value.clone() for name, value in capacity.named_parameters() if not value.requires_grad}
    previous = capacity.capacity_recurrent.weight_ih_l0.detach().clone()
    optimizer = torch.optim.Adam(capacity.parameters(), lr=1e-3)
    capacity.train()
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        value = torch.randn(1, 17, 257, 8, 2)*.1
        loss = (capacity(value)-torch.randn(1, 2, 17, 257)*.05).square().mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in capacity.parameters() if p.requires_grad)
        derivative = capacity.capacity_recurrent.weight_ih_l0.grad
        assert (torch.count_nonzero(derivative) == 0) if step == 0 else derivative.abs().sum() > 0
        optimizer.step()
    assert not torch.equal(previous, capacity.capacity_recurrent.weight_ih_l0)
    assert all(torch.equal(value, dict(capacity.named_parameters())[name]) for name, value in frozen.items())
    assert {int(item['step']) for item in optimizer.state.values()} == {3}
    assert all(torch.isfinite(value).all() for item in optimizer.state.values() for value in item.values())


def test_nonzero_new_readout_is_causal_with_independent_bounded_states():
    _, capacity = models()
    with torch.no_grad():
        capacity.capacity_projection.weight.normal_(std=.01)
    capacity.eval()
    value = (torch.randn(1, 131, 257, 8, 2)*.1).requires_grad_()
    full = capacity(value)
    derivative, = torch.autograd.grad(full[:, :, :65].square().mean(), value)
    assert torch.count_nonzero(derivative[:, 65:]) == 0
    with torch.no_grad():
        changed = value.clone()
        changed[:, 65:] *= 11
        assert torch.equal(full[:, :, :65], capacity(changed)[:, :, :65])
        parts = []
        state = None
        sizes = set()
        for start, end in ((0, 1), (1, 17), (17, 64), (64, 65), (65, 131)):
            part, state = capacity.forward_stream(value[:, start:end], state)
            parts.append(part)
            sizes.add(sum(t.untyped_storage().nbytes() for t in tensor_leaves(state)))
        assert sizes == {6220048}
        torch.testing.assert_close(torch.cat(parts, 2), full, rtol=1e-4, atol=2e-6)
        before = [t.clone() for t in tensor_leaves(state)]
        _, later = capacity.forward_stream(value[:, :1], state)
        assert all(torch.equal(t, old) for t, old in zip(tensor_leaves(state), before))
        _, other = capacity.forward_stream(value[:, :1])
        pointer = lambda s: {t.untyped_storage().data_ptr() for t in tensor_leaves(s)}
        assert pointer(state).isdisjoint(pointer(later)) and pointer(state).isdisjoint(pointer(other))
        broken = dict(state, capacity_hidden=state['capacity_hidden'].double())
        with pytest.raises(ValueError, match='differs'):
            capacity.forward_stream(value[:, :1], broken)


def test_invalid_warm_state_is_rejected_before_any_parameter_changes():
    source, capacity = models()
    before = {name: value.clone() for name, value in capacity.state_dict().items()}
    bad = dict(source.state_dict())
    bad['output.weight'] = torch.full_like(bad['output.weight'], float('nan'))
    with pytest.raises(ValueError, match='finite'):
        capacity.load_warm(bad)
    assert all(torch.equal(value, capacity.state_dict()[name]) for name, value in before.items())


def test_full6s_frontend_and_waveform_stream_match_with_active_extra_readout():
    from evaluate_light import build_stft_batch, reconstruct_waveform
    from light_causal_stream import CausalWaveformStream
    _, capacity = models()
    with torch.no_grad():
        capacity.capacity_projection.weight.normal_(std=.01)
    capacity.eval()
    t = torch.arange(96000, dtype=torch.float32)/16000
    speech = .1*(torch.sin(2*torch.pi*143*t)+.2*torch.sin(2*torch.pi*286*t))
    wave = speech[None, :, None]+torch.randn(1, 96000, 8)*.02
    frontend = dict(n_fft=512, hop_length=160, win_length=320, power=.5, device=torch.device('cpu'))
    with torch.inference_mode():
        spectrum = build_stft_batch(wave, **frontend)
        expected = reconstruct_waveform(capacity(spectrum), length=96000, **frontend)
        stream = CausalWaveformStream(capacity)
        stream.warmup()
        assert stream.state is None and stream.total_samples == stream.emitted_samples == 0
        parts = [stream.push(wave[:, start:start+160]) for start in range(0, 96000, 160)]
        parts.append(stream.flush())
        actual = torch.cat(parts, 1)
        assert actual.shape == expected.shape == (1, 96000)
        assert (actual-expected).abs().max() < 1e-5
        assert sum(t.untyped_storage().nbytes() for t in tensor_leaves(stream.state)) == 6220048

"""Independent prototype: actual learning, future independence and owned fixed-size state."""
import copy

import pytest
import torch

from experiments.causal_frequency_state_memory import CausalFrequencyStateMemory


def network():
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    return CausalFrequencyStateMemory()


def tensors(value):
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in tensors(item)]
    if isinstance(value, tuple):
        return [tensor for item in value for tensor in tensors(item)]
    return []


@pytest.mark.parametrize('frames', [1, 9, 33])
def test_zero_readout_preserves_reference(frames):
    model = network().eval()
    value = torch.randn(1, frames, 257, 8, 2)*.1
    with torch.no_grad():
        assert torch.equal(model(value), value[..., 0, :].permute(0, 3, 1, 2))


def test_natural_zero_readout_two_adam_updates_reach_frequency_and_temporal_memory():
    model = network().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        value = torch.randn(1, 9, 257, 8, 2)*.1
        loss = (model(value)-torch.randn(1, 2, 9, 257)*.05).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        if step:
            for name in ('blocks.0.frequency_rnn.weight_ih_l0', 'blocks.0.hidden_time.weight_ih_l0', 'blocks.0.cell_time.weight_ih_l0'):
                assert dict(model.named_parameters())[name].grad.abs().sum() > 0
        optimizer.step()
    for name in ('encoder.conv.weight', 'blocks.0.frequency_rnn.weight_ih_l0', 'blocks.0.hidden_time.weight_ih_l0', 'blocks.0.cell_time.weight_ih_l0'):
        assert not torch.equal(before[name], dict(model.named_parameters())[name])
    assert {int(item['step']) for item in optimizer.state.values()} == {2}
    assert all(torch.isfinite(value).all() for item in optimizer.state.values() for value in item.values())


def test_nonzero_future_gradient_and_chunk_equivalence_with_owned_bounded_state():
    model = network().eval()
    with torch.no_grad():
        model.decoder.conv.weight.normal_(std=.01)
    value = (torch.randn(1, 19, 257, 8, 2)*.1).requires_grad_()
    result = model(value)
    gradient, = torch.autograd.grad(result[:, :, :8].square().sum(), value)
    assert gradient[:, :8].abs().sum() > 0
    assert torch.count_nonzero(gradient[:, 8:]) == 0
    with torch.no_grad():
        result = model(value.detach())
        perturbed = value.detach().clone()
        perturbed[:, 8:] = torch.randn_like(perturbed[:, 8:])*10
        assert torch.equal(model(perturbed)[:, :, :8], result[:, :, :8])
        state = None
        pieces, sizes = [], []
        for start, stop in ((0, 1), (1, 8), (8, 10), (10, 19)):
            old = copy.deepcopy(state)
            saved = [item.clone() for item in tensors(state)]
            piece, state = model.forward_stream(value.detach()[:, start:stop], state)
            assert all(torch.equal(a, b) for a, b in zip(saved, tensors(old)))
            pieces.append(piece)
            sizes.append(sum(item.numel()*item.element_size() for item in tensors(state)))
        assert max(sizes) == min(sizes) == 996112
        assert torch.allclose(torch.cat(pieces, 2), result, atol=1e-5, rtol=1e-5)
        again, _ = model.forward_stream(value.detach())
        assert torch.allclose(again, result, atol=1e-5, rtol=1e-5)


def test_caller_state_is_not_mutated_or_shared_between_streams():
    model = network().eval()
    with torch.no_grad():
        model.decoder.conv.weight.normal_(std=.01)
        value = torch.randn(1, 7, 257, 8, 2)*.1
        _, state = model.forward_stream(value[:, :3])
        old = [item.clone() for item in tensors(state)]
        first, next_state = model.forward_stream(value[:, 3:], state)
        assert all(torch.equal(a, b) for a, b in zip(old, tensors(state)))
        model.forward_stream(torch.randn_like(value))
        second, other_state = model.forward_stream(value[:, 3:], state)
        assert torch.equal(first, second)
        assert all(a.data_ptr() != b.data_ptr() for a, b in zip(tensors(next_state), tensors(other_state)))
        with pytest.raises(ValueError, match='complete independently owned'):
            model.forward_stream(value, {'encoder': None})
        with pytest.raises(ValueError, match='differs from this stream'):
            model.forward_stream(value.expand(2, -1, -1, -1, -1), state)
        partial = copy.deepcopy(state)
        partial['blocks'][0]['hidden'] = None
        with pytest.raises(ValueError, match='complete independently owned'):
            model.forward_stream(value, partial)


def test_model_retains_historical_frame_gain_information():
    model = network().eval()
    with torch.no_grad():
        model.decoder.conv.weight.normal_(std=.01)
    value = torch.randn(1, 13, 257, 8, 2)*.1
    gain = torch.ones(1, 13, 1, 1, 1, requires_grad=True)
    result = model(value*gain)
    derivative, = torch.autograd.grad(result[:, :, -1].square().sum(), gain)
    assert torch.isfinite(derivative).all()
    assert derivative[:, :-1].abs().sum() > 0


def test_updated_model_matches_original_frontend_full6s_and600_waveform_hops():
    from light_causal_stream import CausalWaveformStream
    from light_quality_loss import QualityObjective
    from train_light import build_stft_batch

    model = network().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        value = torch.randn(1, 9, 257, 8, 2)*.1
        loss = (model(value)-torch.randn(1, 2, 9, 257)*.05).square().mean()
        loss.backward()
        optimizer.step()
    assert model.decoder.conv.weight.abs().sum() > 0
    model.eval()
    waveform = torch.randn(1, 96000, 8)*.02
    with torch.no_grad():
        features, _ = build_stft_batch(waveform, waveform[:, :, 0], 512, 160, 320, .5, torch.device('cpu'))
        expected = QualityObjective().reconstruct(model(features), 96000)
        streaming = CausalWaveformStream(model)
        streaming.warmup()
        pieces = [streaming.push(waveform[:, offset:offset+160]) for offset in range(0, 96000, 160)]
        pieces.append(streaming.flush())
        actual = torch.cat(pieces, -1)
    assert actual.shape == expected.shape == (1, 96000)
    assert (actual-expected).abs().max() < 1e-5
    assert sum(item.numel()*item.element_size() for item in tensors(streaming.state)) == 996112

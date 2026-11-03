"""Standalone global model: actual learning, no future dependence and owned bounded state."""
import copy

import pytest
import torch

from experiments.causal_global_spectral_unet import CausalGlobalSpectralUNet


def model():
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    return CausalGlobalSpectralUNet()


@pytest.mark.parametrize('frames', [1, 9, 33])
def test_global_zero_readout_preserves_reference(frames):
    network = model().eval()
    value = torch.randn(1, frames, 257, 8, 2)*.1
    with torch.no_grad():
        assert torch.equal(network(value), value[..., 0, :].permute(0, 3, 1, 2))


def learned():
    network = model().train()
    before = {name: p.detach().clone() for name, p in network.named_parameters()}
    optimizer = torch.optim.Adam(network.parameters(), lr=.001)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        value = torch.randn(1, 9, 257, 8, 2)*.1
        loss = (network(value)-torch.randn(1, 2, 9, 257)*.05).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in network.parameters())
        if step:
            for name in ('encoders.0.0.weight', 'recurrent.weight_ih_l0', 'recurrent.weight_ih_l2'):
                assert dict(network.named_parameters())[name].grad.abs().sum() > 0
        optimizer.step()
    for name in ('encoders.0.0.weight', 'recurrent.weight_ih_l0', 'recurrent.weight_ih_l2'):
        assert not torch.equal(before[name], dict(network.named_parameters())[name])
    assert {int(state['step']) for state in optimizer.state.values()} == {2}
    assert all(torch.isfinite(value).all() for state in optimizer.state.values() for value in state.values())
    return network.eval()


def test_global_actual_updates_future_independence_and_chunk_equivalence():
    network = learned()
    value = (torch.randn(1, 19, 257, 8, 2)*.1).requires_grad_()
    gradient, = torch.autograd.grad(network(value)[:, :, :8].square().sum(), value)
    assert gradient[:, :8].abs().sum() > 0 and torch.count_nonzero(gradient[:, 8:]) == 0
    with torch.no_grad():
        expected = network(value.detach())
        future = value.detach().clone(); future[:, 8:] = torch.randn_like(future[:, 8:])*10
        assert torch.equal(network(future)[:, :, :8], expected[:, :, :8])
        pieces, state = [], None
        for start, stop in ((0, 1), (1, 8), (8, 10), (10, 19)):
            old = copy.deepcopy(state)
            piece, next_state = network.forward_stream(value.detach()[:, start:stop], state)
            if state:
                assert torch.equal(state['hidden'], old['hidden'])
                assert next_state['hidden'].data_ptr() != state['hidden'].data_ptr()
            assert next_state['hidden'].numel()*next_state['hidden'].element_size() == 4608
            state = next_state; pieces.append(piece)
        assert torch.allclose(torch.cat(pieces, 2), expected, rtol=1e-5, atol=1e-5)


def test_global_independent_streams_and_invalid_state():
    network = model().eval()
    with torch.no_grad():
        network.output.weight.normal_(std=.01)
        value = torch.randn(1, 9, 257, 8, 2)*.1
        _, state = network.forward_stream(value[:, :3])
        old = state['hidden'].clone()
        first, a = network.forward_stream(value[:, 3:], state)
        network.forward_stream(torch.randn_like(value))
        second, b = network.forward_stream(value[:, 3:], state)
        assert torch.equal(first, second) and torch.equal(state['hidden'], old)
        assert a['hidden'].data_ptr() != b['hidden'].data_ptr()
        for bad in ({'wrong': old}, {'hidden': None}, {'hidden': old.half()}):
            with pytest.raises(ValueError): network.forward_stream(value, bad)
        with pytest.raises(ValueError): network.forward_stream(value.expand(2, -1, -1, -1, -1), state)


def test_global_historical_gain_information_is_available():
    network = model().eval()
    with torch.no_grad(): network.output.weight.normal_(std=.01)
    gain = torch.ones(1, 13, 1, 1, 1, requires_grad=True)
    output = network(torch.randn(1, 13, 257, 8, 2)*.1*gain)
    derivative, = torch.autograd.grad(output[:, :, -1].square().sum(), gain)
    assert torch.isfinite(derivative).all() and derivative[:, :-1].abs().sum() > 0


def test_global_updated_model_full6s_waveform_and600_hops_match():
    from light_causal_stream import CausalWaveformStream
    from light_quality_loss import QualityObjective
    from train_light import build_stft_batch

    network = learned()
    waveform = torch.randn(1, 96000, 8)*.02
    with torch.no_grad():
        features, _ = build_stft_batch(waveform, waveform[:, :, 0], 512, 160, 320, .5, torch.device('cpu'))
        expected = QualityObjective().reconstruct(network(features), 96000)
        stream = CausalWaveformStream(network); stream.warmup()
        pieces = [stream.push(waveform[:, start:start+160]) for start in range(0, 96000, 160)]
        pieces.append(stream.flush()); actual = torch.cat(pieces, 1)
    assert actual.shape == expected.shape == (1, 96000)
    assert (actual-expected).abs().max() < 1e-5
    assert stream.state['hidden'].numel()*stream.state['hidden'].element_size() == 4608

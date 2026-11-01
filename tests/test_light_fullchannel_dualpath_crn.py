"""Nonzero full-channel LSTMs, strict temporal causality and owned h/c state."""
import pytest
import torch

from experiments.causal_fullchannel_dualpath_crn import CausalFullChannelDualPathCRN
from light_causal_stream import CausalWaveformStream
from light_quality_loss import QualityObjective
from train_light import build_stft_batch


@pytest.fixture(scope='module')
def trained():
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    model = CausalFullChannelDualPathCRN()
    value = torch.randn(1, 23, 257, 8, 2) * .02
    torch.testing.assert_close(model(value), value[..., 0, :].permute(0, 3, 1, 2), rtol=0, atol=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        (model(value) - value[..., 7, :].permute(0, 3, 1, 2) * .6).square().mean().backward()
        assert all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
    assert torch.count_nonzero(model.recurrent[0].frequency.weight_hh_l0_reverse.grad) > 0
    assert torch.count_nonzero(model.recurrent[0].temporal.weight_hh_l0.grad) > 0
    assert sum(p.numel() for p in model.parameters()) == 1783826
    assert all(torch.isfinite(value).all() for state in optimizer.state.values() for value in state.values())
    return model


def test_fullchannel_nonzero_train_and_eval_causality(trained):
    value = (torch.randn(1, 47, 257, 8, 2) * .02).requires_grad_()
    changed = value.detach().clone()
    changed[:, 19:] *= 13
    for mode in (True, False):
        trained.train(mode)
        result = trained(value)
        torch.testing.assert_close(result[:, :, :19], trained(changed)[:, :, :19], rtol=0, atol=0)
    result[:, :, :19].square().mean().backward()
    assert torch.count_nonzero(value.grad[:, 19:]) == 0
    assert torch.count_nonzero(value.grad[:, :19, :, 7]) > 0


def test_fullchannel_partition_state_ownership_and_shape(trained):
    trained.eval()
    value = torch.randn(2, 47, 257, 8, 2) * .02
    with torch.inference_mode():
        result = trained(value)
        state, parts = None, []
        for begin, end in ((0, 1), (1, 17), (17, 32), (32, 47)):
            piece, state = trained.forward_stream(value[:, begin:end], state)
            parts.append(piece)
        torch.testing.assert_close(torch.cat(parts, 2), result, rtol=1e-4, atol=2e-6)
        saved = state['recurrent'][0][0].clone()
        saved_cell = state['recurrent'][0][1].clone()
        trained.forward_stream(value[:, :1], state)
        torch.testing.assert_close(state['recurrent'][0][0], saved, rtol=0, atol=0)
        torch.testing.assert_close(state['recurrent'][0][1], saved_cell, rtol=0, atol=0)
        with pytest.raises(ValueError, match='state differs'):
            trained.forward_stream(value[:1, :1], state)


def test_fullchannel_600_hops_and_constant_owned_cache(trained):
    trained.eval()
    waveform = torch.randn(1, 96000, 8) * .02
    with torch.inference_mode():
        features, _ = build_stft_batch(waveform, waveform[:, :, 0], 512, 160, 320, .5, torch.device('cpu'))
        full = QualityObjective().reconstruct(trained(features), 96000)
        stream = CausalWaveformStream(trained)
        stream.warmup()
        pieces = [stream.push(waveform[:, begin:begin + 160]) for begin in range(0, 96000, 160)]
        pieces.append(stream.flush())
        torch.testing.assert_close(torch.cat(pieces, 1), full, rtol=1e-4, atol=2e-6)
        def byte_count(value):
            if torch.is_tensor(value):
                assert value.untyped_storage().nbytes() == value.numel() * value.element_size()
                return value.untyped_storage().nbytes()
            if isinstance(value, dict):
                return sum(byte_count(item) for item in value.values())
            if isinstance(value, (list, tuple)):
                return sum(byte_count(item) for item in value)
            return 0
        assert byte_count(stream.state) == 933376


@pytest.mark.parametrize('settings', [dict(M=9), dict(layers=0), dict(frequency_hidden=0), dict(temporal_hidden=True), dict(is_causal=False)])
def test_fullchannel_invalid_dimensions(settings):
    with pytest.raises(ValueError):
        CausalFullChannelDualPathCRN(**settings)

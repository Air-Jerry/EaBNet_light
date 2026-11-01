"""Nonzero warm-output inheritance, full-band causality and real waveform caches."""

import copy

import pytest
import torch

from experiments.causal_fullband_conditioned_subband import CausalFullbandConditionedSubband
from light_causal_stream import CausalWaveformStream
from light_quality_loss import QualityObjective
from test_light_subband_gru import model as source_model
from train_light import build_stft_batch


def model(nonzero=False):
    old = source_model(nonzero=True)
    rng = torch.get_rng_state().clone()
    net = CausalFullbandConditionedSubband(copy.deepcopy(old.base))
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    missing, unexpected = net.load_state_dict(old.state_dict(), strict=False)
    assert len(missing) == 10 and all(name.startswith('fullband_') for name in missing) and not unexpected
    if nonzero:
        net.fullband_mix.weight.data.normal_(std=.01)
    return net, old


def test_fullband_condition_preserves_active_subband_then_trains():
    net, old = model()
    value = torch.randn(2, 21, 257, 8, 2)*.02
    torch.testing.assert_close(net(value), old(value), rtol=0, atol=0)
    assert sum(p.numel() for p in net.parameters())-sum(p.numel() for p in old.parameters()) == 779316
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        net(value).square().mean().backward()
        assert all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all()
                   for p in net.parameters())
        optimizer.step()
    assert torch.count_nonzero(net.fullband_rnn.weight_hh_l0.grad) > 0
    assert torch.count_nonzero(net.fullband_projection.weight.grad) > 0
    assert all(torch.isfinite(v).all() for s in optimizer.state.values() for v in s.values())


def test_fullband_condition_nonzero_causality_partition_and_owned_cache():
    net, _ = model(nonzero=True); net.eval()
    value = (torch.randn(2, 57, 257, 8, 2)*.02).requires_grad_()
    full = net(value)
    changed = value.detach().clone(); changed[:, 23:] *= 13
    torch.testing.assert_close(full[:, :, :23], net(changed)[:, :, :23], rtol=0, atol=0)
    full[:, :, :23].square().mean().backward()
    assert torch.count_nonzero(value.grad[:, 23:]) == 0
    assert torch.count_nonzero(net.fullband_rnn.weight_hh_l0.grad) > 0
    with torch.inference_mode():
        separate = torch.cat([net(value[i:i+1]) for i in range(2)])
        torch.testing.assert_close(separate, full, rtol=1e-4, atol=2e-6)
        parts, state = [], None
        for start, end in ((0, 1), (1, 11), (11, 32), (32, 57)):
            part, state = net.forward_stream(value[:, start:end], state)
            parts.append(part)
            assert state['fullband_hidden'].shape == (1, 2, 128)
            assert state['fullband_hidden'].untyped_storage().nbytes() == 2*128*4
        torch.testing.assert_close(torch.cat(parts, 2), full, rtol=1e-4, atol=2e-6)
        snapshot = {k:state[k].clone() for k in ('fullband_hidden', 'subband')}
        net.forward_stream(value[:, :1], state)
        for key, saved in snapshot.items():torch.testing.assert_close(state[key], saved, rtol=0, atol=0)
        with pytest.raises(ValueError, match='Full-band state'):
            net.forward_stream(value[:1, :1], dict(fullband_hidden=state['fullband_hidden']))


def test_fullband_condition_600_actual_waveform_hops():
    net, _ = model(nonzero=True); net.eval()
    wave = torch.randn(1, 96000, 8)*.02
    with torch.inference_mode():
        features, _ = build_stft_batch(wave, wave[:, :, 0], 512, 160, 320, .5, torch.device('cpu'))
        full = QualityObjective().reconstruct(net(features), wave.shape[1])
        stream = CausalWaveformStream(net); stream.warmup()
        parts = [stream.push(wave[:, start:start+160]) for start in range(0, 96000, 160)]
        actual = torch.cat([*parts, stream.flush()], 1)
        torch.testing.assert_close(actual, full, rtol=1e-4, atol=2e-6)
        assert stream.state['fullband_hidden'].untyped_storage().nbytes() == 512

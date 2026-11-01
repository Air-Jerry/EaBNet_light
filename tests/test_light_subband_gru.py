"""Warm-source identity, nonzero recurrent gradients and causal waveform streaming."""

import pytest
import torch

from experiments.causal_retention_mapping import CausalAttentionRetentionTCN
from experiments.causal_subband_gru import CausalSubbandGRU
from experiments.causal_pmsqe_objective import PMSQEObjective
from light_causal_stream import CausalWaveformStream
from light_quality_loss import QualityObjective
from train_light import build_stft_batch


def model(nonzero=False):
    torch.set_num_threads(2)
    torch.manual_seed(7363)
    base = CausalAttentionRetentionTCN(hidden=32, ffn=64, layers=2, global_hidden=16,
                                     context=5, mapping_channels=8)
    for p in (base.beamformer.weight, base.mapping_output.weight,
              base.mapping_input.frequency_output.weight, base.retention.output.weight):
        p.data.normal_(std=.001)
    result = CausalSubbandGRU(base)
    if nonzero:
        result.output.weight.data.normal_(std=.001)
    return result


def test_subband_gru_preserves_nonzero_base_and_rng_then_trains():
    net = model()
    original = {k: v.clone() for k, v in net.base.state_dict().items()}
    rng = torch.get_rng_state()
    CausalSubbandGRU(net.base)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    value = torch.randn(2, 21, 257, 8, 2)*.02
    torch.testing.assert_close(net(value), net.base(value), rtol=0, atol=0)
    for k, v in original.items():
        torch.testing.assert_close(v, net.base.state_dict()[k], rtol=0, atol=0)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        net(value).square().mean().backward()
        assert all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all()
                   for p in net.parameters())
        optimizer.step()
    assert torch.count_nonzero(net.recurrent.weight_ih_l0.grad) > 0
    assert torch.count_nonzero(net.input.weight.grad) > 0
    assert all(torch.isfinite(v).all() for s in optimizer.state.values() for v in s.values())


def test_subband_gru_nonzero_causality_chunk_equivalence_and_owned_state():
    net = model(nonzero=True).eval()
    value = (torch.randn(2, 57, 257, 8, 2)*.02).requires_grad_()
    full = net(value)
    changed = value.detach().clone(); changed[:, 23:] *= 13
    torch.testing.assert_close(full[:, :, :23], net(changed)[:, :, :23], rtol=0, atol=0)
    full[:, :, :23].square().mean().backward()
    assert torch.count_nonzero(value.grad[:, 23:]) == 0
    assert torch.count_nonzero(net.recurrent.weight_hh_l0.grad) > 0
    with torch.inference_mode():
        pieces, state = [], None
        for start, end in ((0, 1), (1, 11), (11, 32), (32, 57)):
            part, state = net.forward_stream(value[:, start:end], state)
            pieces.append(part)
            assert state['subband'].shape == (1, 2*257, 32)
            assert state['subband'].untyped_storage().nbytes() == 2*257*32*4
        torch.testing.assert_close(torch.cat(pieces, 2), full, rtol=1e-4, atol=2e-6)
        snapshot = state['subband'].clone()
        net.forward_stream(value[:, :1], state)
        torch.testing.assert_close(state['subband'], snapshot, rtol=0, atol=0)
        with pytest.raises(ValueError, match='state'):
            net.forward_stream(value[:1, :1], dict(subband=state['subband']))


def test_subband_gru_original_frontend_waveform_streaming():
    net = model(nonzero=True).eval()
    wave = torch.randn(1, 11513, 8)*.02
    with torch.inference_mode():
        features, _ = build_stft_batch(wave, wave[:, :, 0], 512, 160, 320, .5, torch.device('cpu'))
        full = QualityObjective().reconstruct(net(features), wave.shape[1])
        stream = CausalWaveformStream(net); stream.warmup()
        parts = [stream.push(wave[:, start:start+160]) for start in range(0, wave.shape[1], 160)]
        actual = torch.cat([*parts, stream.flush()], 1)
        torch.testing.assert_close(actual, full, rtol=1e-4, atol=2e-6)


def test_subband_gru_microbatch_loss_weighting_preserves_prediction_gradients():
    net = model(nonzero=True).eval()
    wave = torch.randn(4, 2413, 8)*.02
    target = wave[:, :, 0]*.7 + torch.randn(4, 2413)*.003
    objective = PMSQEObjective()
    features, reference = objective.build_features(wave, target, 512, 160, 320, .5, 'cpu', [2413]*4)
    with torch.no_grad():
        predictions = net(features)
        micro_predictions = torch.cat([net(features[start:start+2]) for start in (0, 2)])
        torch.testing.assert_close(micro_predictions, predictions, rtol=1e-4, atol=2e-6)
    # Fixed predictions isolate loss weighting from batch-dependent GEMM roundoff.
    full_estimate = predictions.detach().clone().requires_grad_()
    micro_estimate = predictions.detach().clone().requires_grad_()
    full_loss = objective(full_estimate, reference, [features.shape[1]]*4)
    full_loss.backward()
    micro_objective = PMSQEObjective()
    micro_losses = []
    for start in (0, 2):
        micro_objective.build_features(wave[start:start+2], target[start:start+2],
                                        512, 160, 320, .5, 'cpu', [2413]*2)
        loss = micro_objective(micro_estimate[start:start+2], reference[start:start+2],
                               [features.shape[1]]*2)/2
        loss.backward(); micro_losses.append(loss.detach())
    torch.testing.assert_close(sum(micro_losses), full_loss.detach(), rtol=1e-6, atol=1e-6)
    expected = full_estimate.grad
    actual = micro_estimate.grad
    assert torch.isfinite(actual).all()
    assert torch.linalg.vector_norm(actual-expected)/torch.linalg.vector_norm(expected) < 5e-5

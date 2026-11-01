"""Reference initialization, causality and bounded waveform streaming."""

import torch

from experiments.causal_fullband_tcn import CausalFullbandTCN
from light_causal_stream import CausalWaveformStream
from light_quality_loss import QualityObjective
from train_light import build_stft_batch


def model():
    torch.set_num_threads(2)
    torch.manual_seed(7351)
    net = CausalFullbandTCN(fullband_channels=16, temporal_hidden=32, repeats=2, dilations=(1,2,4))
    net.beamformer.weight.data.normal_(std=.001)
    net.mapping.weight.data.normal_(std=.001)
    return net


def test_fullband_initialization_is_exact_reference_and_gradients_are_finite():
    net = model()
    net.beamformer.weight.data.zero_()
    net.mapping.weight.data.zero_()
    value = torch.randn(2, 31, 257, 8, 2) * .02
    result = net(value)
    torch.testing.assert_close(result, value[..., 0, :].permute(0,3,1,2), rtol=0, atol=0)
    result.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def test_fullband_has_no_future_dependency_and_matches_arbitrary_chunks():
    net = model().eval()
    value = (torch.randn(1, 145, 257, 8, 2) * .02).requires_grad_()
    expected = net(value)
    changed = value.detach().clone()
    changed[:, 71:] *= 19
    torch.testing.assert_close(expected[:, :, :71], net(changed)[:, :, :71], rtol=0, atol=0)
    expected[:, :, :71].square().mean().backward()
    assert torch.count_nonzero(value.grad[:, 71:]) == 0
    assert torch.isfinite(value.grad).all()
    with torch.inference_mode():
        pieces, state = [], None
        for start,end in ((0,1),(1,31),(31,95),(95,145)):
            part,state = net.forward_stream(value[:,start:end], state)
            pieces.append(part)
            for layer,history in zip(net.layers,state['layers']):
                assert history.shape == (1,32,layer.history_frames)
                assert history.untyped_storage().nbytes() == 32 * layer.history_frames * 4
        torch.testing.assert_close(torch.cat(pieces,2),expected,rtol=1e-4,atol=2e-6)


def test_fullband_streamed_waveform_matches_original_frontend():
    net = model().eval()
    wave = torch.randn(1, 22137, 8) * .02
    with torch.inference_mode():
        features,_ = build_stft_batch(wave,wave[:,:,0],512,160,320,.5,torch.device('cpu'))
        full = QualityObjective().reconstruct(net(features),wave.shape[1])
        stream = CausalWaveformStream(net)
        stream.warmup()
        pieces = [stream.push(wave[:,start:start+160]) for start in range(0,wave.shape[1],160)]
        actual = torch.cat([*pieces,stream.flush()],1)
        torch.testing.assert_close(actual,full,rtol=1e-4,atol=2e-6)

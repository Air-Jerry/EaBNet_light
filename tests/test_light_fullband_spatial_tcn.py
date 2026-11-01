"""Warm-source identity, direct spatial gradients, causality and bounded streaming."""

import torch

from experiments.causal_fullband_tcn import CausalFullbandTCN
from experiments.causal_fullband_spatial_tcn import CausalFullbandSpatialTCN
from light_causal_stream import CausalWaveformStream
from light_quality_loss import QualityObjective
from train_light import build_stft_batch


BASE = dict(fullband_channels=16,temporal_hidden=32,repeats=2,dilations=(1,2,4))
LOCAL = dict(spatial_channels=8,spatial_hidden=12,spatial_dilations=(1,2,4))


def model(nonzero=True):
    torch.set_num_threads(2)
    torch.manual_seed(7354)
    net = CausalFullbandSpatialTCN(**BASE,**LOCAL)
    if nonzero:net.spatial_readout.weight.data.normal_(std=.001)
    net.beamformer.weight.data.normal_(std=.001)
    net.mapping.weight.data.normal_(std=.001)
    return net


def test_fullband_spatial_zero_output_preserves_nonzero_source_exactly():
    net = model(nonzero=False)
    source = CausalFullbandTCN(**BASE)
    source.load_state_dict({k:v for k,v in net.state_dict().items() if not k.startswith('spatial_')},strict=True)
    value = torch.randn(2,19,257,8,2)*.02
    torch.testing.assert_close(net(value),source(value),rtol=0,atol=0)
    net(value).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    assert torch.count_nonzero(net.spatial_readout.weight.grad)>0


def test_fullband_spatial_no_future_dependency_owned_cache_and_spatial_gradients():
    net = model().eval()
    value = (torch.randn(1,75,257,8,2)*.02).requires_grad_()
    full = net(value)
    changed = value.detach().clone();changed[:,33:]*=17
    torch.testing.assert_close(full[:,:,:33],net(changed)[:,:,:33],rtol=0,atol=0)
    full[:,:,:33].square().mean().backward()
    assert torch.count_nonzero(value.grad[:,33:])==0
    assert torch.count_nonzero(net.spatial_input.weight.grad)>0
    assert torch.count_nonzero(net.spatial_condition.weight.grad)>0
    with torch.inference_mode():
        pieces,state = [],None
        for start,end in ((0,1),(1,16),(16,45),(45,75)):
            part,state = net.forward_stream(value[:,start:end],state)
            pieces.append(part)
            for layer,history in zip(net.spatial_layers,state['spatial_layers']):
                assert history.shape==(1,12,layer.history_frames,257)
                assert history.untyped_storage().nbytes()==12*layer.history_frames*257*4
        torch.testing.assert_close(torch.cat(pieces,2),full,rtol=1e-4,atol=2e-6)


def test_fullband_spatial_original_frontend_waveform_streaming():
    net = model().eval()
    wave = torch.randn(1,11637,8)*.02
    with torch.inference_mode():
        features,_ = build_stft_batch(wave,wave[:,:,0],512,160,320,.5,torch.device('cpu'))
        full = QualityObjective().reconstruct(net(features),wave.shape[1])
        stream = CausalWaveformStream(net);stream.warmup()
        pieces = [stream.push(wave[:,start:start+160]) for start in range(0,wave.shape[1],160)]
        actual = torch.cat([*pieces,stream.flush()],1)
        torch.testing.assert_close(actual,full,rtol=1e-4,atol=2e-6)

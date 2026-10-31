"""Nonzero-output causality and fixed caches for the experimental capacity screen."""

import pytest
import torch

from experiments.causal_global_capacity import CausalGlobalCapacity
from light_causal_recurrent65_model import CausalRecurrentUNet65
from light_causal_stream import CausalWaveformStream
from light_quality_loss import QualityObjective
from train_light import build_stft_batch


SMALL=dict(hidden=32,ffn=64,layers=2,global_hidden=16,context=5)


@pytest.fixture
def model():
    torch.set_num_threads(2);torch.manual_seed(82)
    model=CausalGlobalCapacity(expanded_hidden=32,**SMALL)
    model.beamformer.weight.data.normal_(std=.01)
    model.layers[-1].output.weight.data.normal_(std=.01)
    return model


def test_capacity_zero_residual_retains_all_weights_output_and_crop_rng():
    torch.set_num_threads(2);torch.manual_seed(1234)
    source=CausalRecurrentUNet65(**SMALL);source.beamformer.weight.data.normal_(std=.01)
    expected=torch.get_rng_state()
    torch.manual_seed(1234);changed=CausalGlobalCapacity(expanded_hidden=32,**SMALL)
    changed.beamformer.weight.data.normal_(std=.01)
    torch.testing.assert_close(torch.get_rng_state(),expected,rtol=0,atol=0)
    values=changed.state_dict();values.update(source.state_dict());changed.load_state_dict(values,strict=True)
    assert all(p.requires_grad for p in changed.parameters())
    source.eval();changed.eval()
    with torch.inference_mode():
        x=torch.randn(1,11,257,8,2)*.02
        torch.testing.assert_close(changed(x),source(x),rtol=0,atol=0)


@pytest.mark.parametrize('training',[False,True])
def test_capacity_future_frames_have_zero_gradient(model,training):
    model.train(training)
    x=(torch.randn(1,15,257,8,2)*.02).requires_grad_()
    altered=x.detach().clone();altered[:,9:]*=17
    output=model(x)
    torch.testing.assert_close(output[:,:,:9],model(altered)[:,:,:9],rtol=0,atol=0)
    output[:,:,:9].square().mean().backward()
    assert torch.count_nonzero(x.grad[:,9:])==0 and torch.count_nonzero(x.grad[:,:9])>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_capacity_stream_and_extra_hidden_size_are_fixed(model):
    model.eval();x=torch.randn(1,23,257,8,2)*.02
    with torch.inference_mode():
        expected=model(x);state=None;parts=[]
        for start,end in ((0,1),(1,7),(7,23)):
            y,state=model.forward_stream(x[:,start:end],state);parts.append(y)
            assert state['middle'][-1]['hidden'].shape==(1,1,32)
        torch.testing.assert_close(torch.cat(parts,2),expected,rtol=1e-4,atol=3e-6)


def test_capacity_waveform_stream_matches_full_frontend(model):
    model.eval();mixture=torch.randn(1,1481,8)*.02
    with torch.inference_mode():
        features,_=build_stft_batch(mixture,mixture[:,:,0],512,160,320,.5,torch.device('cpu'))
        expected=QualityObjective().reconstruct(model(features),1481)
        stream=CausalWaveformStream(model);stream.warmup()
        actual=torch.cat((stream.push(mixture[:,:311]),stream.push(mixture[:,311:]),stream.flush()),1)
    torch.testing.assert_close(actual,expected,rtol=1e-3,atol=1e-5)

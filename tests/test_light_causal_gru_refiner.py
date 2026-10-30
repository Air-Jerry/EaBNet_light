"""Nonzero-head causality, gradient and waveform consistency of the prototype."""

import pytest
import torch

from experiments.causal_gru_refiner import CausalGRURefiner
from light_causal_fast_model import CausalUNetSpatial
from light_causal_stream import CausalWaveformStream
from light_quality_loss import QualityObjective
from train_light import build_stft_batch


@pytest.fixture
def model():
    torch.set_num_threads(2);torch.manual_seed(45)
    backbone=CausalUNetSpatial(hidden=32,ffn=64,layers=2,context=5)
    model=CausalGRURefiner(backbone,hidden=16,channels=16)
    model.backbone.beamformer.weight.data.normal_(std=.01)
    model.output.weight.data.normal_(std=.01)
    return model


@pytest.mark.parametrize('training',[False,True])
def test_refiner_future_independence_and_gradients(model,training):
    model.train(training)
    x=(torch.randn(1,13,257,8,2)*.01).requires_grad_()
    changed=x.detach().clone();changed[:,8:]*=13
    out=model(x)
    torch.testing.assert_close(out[:,:,:8],model(changed)[:,:,:8],rtol=0,atol=0)
    out[:,:,:8].square().mean().backward()
    assert torch.count_nonzero(x.grad[:,8:])==0
    assert torch.count_nonzero(x.grad[:,:8])>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_stream_chunks_and_zero_head_identity(model):
    model.eval()
    x=torch.randn(1,31,257,8,2)*.01
    with torch.inference_mode():
        full=model(x)
        a,s=model.forward_stream(x[:,:9]);b,s=model.forward_stream(x[:,9:],s)
        torch.testing.assert_close(full,torch.cat((a,b),2),rtol=3e-5,atol=2e-6)
        assert s['rnn1'].shape==s['rnn2'].shape==(1,257,16)
        model.output.weight.zero_();model.output.bias.zero_()
        torch.testing.assert_close(model(x),model.backbone(x),rtol=0,atol=0)


def test_waveform_stream_matches_original_frontend(model):
    model.eval();mixture=torch.randn(1,1327,8)*.01
    with torch.inference_mode():
        features,_=build_stft_batch(mixture,mixture[:,:,0],512,160,320,.5,torch.device('cpu'))
        expected=QualityObjective().reconstruct(model(features),1327)
        stream=CausalWaveformStream(model);stream.warmup()
        actual=torch.cat((stream.push(mixture[:,:111]),stream.push(mixture[:,111:]),stream.flush()),1)
    torch.testing.assert_close(actual,expected,rtol=3e-4,atol=3e-6)

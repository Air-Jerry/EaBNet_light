"""Analytic microphone pairing, phase invariance, warm identity and causal streams."""
import pytest
import torch

from experiments.causal_spatial_cues import CausalSpatialCueUNet65,spatial_cues
from light_causal_recurrent65_model import CausalRecurrentUNet65
from light_causal_stream import CausalWaveformStream
from light_quality_loss import QualityObjective
from train_light import build_stft_batch

SMALL=dict(hidden=32,ffn=64,layers=2,global_hidden=16,context=5)


def test_spatial_cues_match_analytic_phases_ratios_and_common_rotation():
    angle=torch.arange(8)*.23
    amplitude=1+torch.arange(8)*.1
    value=torch.stack((amplitude*angle.cos(),amplitude*angle.sin()),dim=1).reshape(1,16,1,1)
    actual=spatial_cues(value).flatten()
    expected=torch.cat((angle[1:].cos(),angle[1:].sin(),amplitude[1:].log()))
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    rotated=torch.stack((3*amplitude*(angle+.7).cos(),3*amplitude*(angle+.7).sin()),dim=1).reshape(1,16,1,1)
    torch.testing.assert_close(spatial_cues(rotated),spatial_cues(value),rtol=1e-5,atol=1e-6)
    assert torch.isfinite(spatial_cues(torch.zeros_like(value))).all()
    with pytest.raises(ValueError):spatial_cues(torch.zeros(1,18,1,1))


def test_zero_spatial_projection_preserves_source_and_rng():
    torch.set_num_threads(2);torch.manual_seed(1234)
    original=CausalRecurrentUNet65(**SMALL);rng=torch.get_rng_state()
    torch.manual_seed(1234);changed=CausalSpatialCueUNet65(**SMALL)
    torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
    original.beamformer.weight.data.normal_(std=.01)
    values=changed.state_dict();values.update(original.state_dict());changed.load_state_dict(values,strict=True)
    assert all(p.requires_grad for p in changed.parameters())
    x=torch.randn(1,11,257,8,2)*.02
    with torch.inference_mode():
        original.eval();changed.eval()
        torch.testing.assert_close(changed(x),original(x),rtol=0,atol=0)


def test_nonzero_cues_are_causal_and_all_new_parameters_receive_gradients():
    torch.set_num_threads(2);torch.manual_seed(17)
    model=CausalSpatialCueUNet65(**SMALL)
    model.beamformer.weight.data.normal_(std=.01)
    model.stem.spatial_projection.weight.data.normal_(std=.01)
    x=(torch.randn(1,17,257,8,2)*.02).requires_grad_()
    changed=x.detach().clone();changed[:,9:]*=13
    output=model(x)
    torch.testing.assert_close(output[:,:,:9],model(changed)[:,:,:9],rtol=0,atol=0)
    output[:,:,:9].square().mean().backward()
    assert torch.count_nonzero(x.grad[:,9:])==0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert torch.count_nonzero(model.stem.spatial_projection.weight.grad)>0


def test_nonzero_spatial_cues_match_chunks_and_waveform_stream():
    torch.set_num_threads(2);torch.manual_seed(19)
    model=CausalSpatialCueUNet65(**SMALL).eval()
    model.beamformer.weight.data.normal_(std=.01)
    model.stem.spatial_projection.weight.data.normal_(std=.01)
    with torch.inference_mode():
        x=torch.randn(1,37,257,8,2)*.02;full=model(x);state=None;parts=[]
        for a,b in ((0,1),(1,5),(5,19),(19,37)):
            y,state=model.forward_stream(x[:,a:b],state);parts.append(y)
            assert state['stem'].shape==(1,16,2,257)
            assert state['stem'].untyped_storage().nbytes()==16*2*257*4
        torch.testing.assert_close(torch.cat(parts,2),full,rtol=1e-4,atol=1e-6)
        waveform=torch.randn(1,1617,8)*.02
        features,_=build_stft_batch(waveform,waveform[:,:,0],512,160,320,.5,torch.device('cpu'))
        expected=QualityObjective().reconstruct(model(features),1617)
        stream=CausalWaveformStream(model);stream.warmup()
        output=[stream.push(waveform[:,i:i+160]) for i in range(0,1617,160)]
        torch.testing.assert_close(torch.cat([*output,stream.flush()],dim=1),expected,rtol=1e-4,atol=1e-6)

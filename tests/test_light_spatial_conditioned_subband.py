"""Nonzero source preservation, new spatial gradients and actual causal caches."""

import copy

import pytest
import torch

from experiments.causal_frozen_conditioned_subband import CausalFrozenBaseConditionedSubband
from experiments.causal_spatial_conditioned_subband import CausalSpatialConditionedSubband
from light_causal_stream import CausalWaveformStream
from light_quality_loss import QualityObjective
from test_light_fullband_conditioned_subband import model as conditioned_source
from train_light import build_stft_batch


def models(nonzero=False):
    trained,_=conditioned_source(nonzero=True)
    source=CausalFrozenBaseConditionedSubband(copy.deepcopy(trained.base)).eval()
    source.load_state_dict(trained.state_dict(),strict=True)
    rng=torch.get_rng_state().clone()
    model=CausalSpatialConditionedSubband(copy.deepcopy(source.base))
    torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
    missing,unexpected=model.load_state_dict(source.state_dict(),strict=False)
    assert len(missing)==22 and all(key.startswith('spatial_') for key in missing) and not unexpected
    if nonzero:model.spatial_projection.weight.data.normal_(std=.001)
    return model,source


def test_global_spatial_branch_preserves_source_then_updates_without_changing_base():
    model,source=models()
    value=torch.randn(2,25,257,8,2)*.02
    with torch.no_grad():torch.testing.assert_close(model(value),source(value),rtol=0,atol=0)
    before={key:value.clone() for key,value in model.base.state_dict().items()}
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    for _ in range(3):
        model.train()
        assert not model.base.training
        optimizer.zero_grad(set_to_none=True)
        model(value).square().mean().backward()
        assert all(p.grad is None for p in model.base.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        optimizer.step()
    assert torch.count_nonzero(model.spatial_encoder[0].weight.grad)>0
    assert torch.count_nonzero(model.spatial_recurrent.weight_hh_l0.grad)>0
    assert torch.count_nonzero(model.spatial_projection.weight)>0
    assert all(torch.isfinite(value).all() for state in optimizer.state.values() for value in state.values())
    for key,value in model.base.state_dict().items():torch.testing.assert_close(value,before[key],rtol=0,atol=0)


def test_nonzero_spatial_branch_has_no_future_gradient_and_owns_stream_state():
    model,_=models(nonzero=True);model.eval()
    value=(torch.randn(2,53,257,8,2)*.02).requires_grad_()
    full=model(value)
    changed=value.detach().clone();changed[:,19:]*=11
    torch.testing.assert_close(full[:,:,:19],model(changed)[:,:,:19],rtol=0,atol=0)
    full[:,:,:19].square().mean().backward()
    assert torch.count_nonzero(value.grad[:,19:])==0
    assert torch.count_nonzero(model.spatial_recurrent.weight_hh_l0.grad)>0
    with torch.no_grad():
        torch.testing.assert_close(torch.cat([model(value[i:i+1]) for i in range(2)]),full,rtol=1e-4,atol=2e-6)
        parts=[];state=None
        for begin,end in ((0,1),(1,7),(7,29),(29,53)):
            part,state=model.forward_stream(value[:,begin:end],state);parts.append(part)
            assert state['spatial_hidden'].shape==(1,2,256)
            assert state['spatial_hidden'].untyped_storage().nbytes()==2*1024
        torch.testing.assert_close(torch.cat(parts,2),full,rtol=1e-4,atol=2e-6)
        snapshot=state['spatial_hidden'].clone()
        model.forward_stream(value[:,:1],state)
        torch.testing.assert_close(state['spatial_hidden'],snapshot,rtol=0,atol=0)
        with pytest.raises(ValueError,match='Spatial full-band state'):
            model.forward_stream(value[:1,:1],dict(spatial_hidden=state['spatial_hidden']))


def test_global_spatial_branch_original_frontend_600_waveform_hops():
    model,_=models(nonzero=True);model.eval()
    wave=torch.randn(1,96000,8)*.02
    with torch.inference_mode():
        features,_=build_stft_batch(wave,wave[:,:,0],512,160,320,.5,torch.device('cpu'))
        full=QualityObjective().reconstruct(model(features),96000)
        stream=CausalWaveformStream(model);stream.warmup()
        parts=[stream.push(wave[:,start:start+160]) for start in range(0,96000,160)]
        actual=torch.cat([*parts,stream.flush()],1)
        torch.testing.assert_close(actual,full,rtol=1e-4,atol=2e-6)
        assert stream.state['spatial_hidden'].untyped_storage().nbytes()==1024

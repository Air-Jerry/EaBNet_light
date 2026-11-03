"""Independent monaural supervision, source preservation and causal owned states."""
import copy

import pytest
import torch

from experiments.causal_auxiliary_monaural import CausalAuxiliaryMonauralFusion
from test_light_frequency_state_memory import tensors
from test_light_warm_conditioned_global_beamformer import CausalSource


def network():
    torch.set_num_threads(2);torch.manual_seed(1234)
    return CausalAuxiliaryMonauralFusion(CausalSource())


@pytest.mark.parametrize('scale',[0.,1e-6,.1])
def test_initial_fusion_preserves_source_and_mono_starts_at_reference(scale):
    model=network().train();value=torch.randn(1,9,257,8,2)*scale
    assert not model.base.training and all(not p.requires_grad for p in model.base.parameters())
    with torch.no_grad():
        source,_=model.base.process(value)
        assert torch.equal(model(value),source)
        assert torch.equal(model.monaural(value[...,0,:]),value[...,0,:].permute(0,3,1,2))


def learn_twice(model):
    initial={name:value.clone() for name,value in model.state_dict().items()}
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    captured={}
    handle=model.monaural.register_forward_hook(lambda module,arguments,result:captured.update(mono=result))
    try:
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            value=torch.randn(1,9,257,8,2)*.1;target=torch.randn(1,2,9,257)*.03
            enhanced=model(value)
            loss=(enhanced-target).square().mean()+.3*(captured.pop('mono')-target).square().mean()
            loss.backward()
            assert all(p.grad is None for p in model.base.parameters())
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
            optimizer.step()
    finally:handle.remove()
    assert {int(item['step']) for item in optimizer.state.values()}=={2}
    assert all(torch.equal(value,initial[name]) for name,value in model.state_dict().items() if name.startswith('base.'))
    for name in ('monaural.encoders.0.0.weight','monaural.blocks.0.frequency.weight_ih_l0_reverse',
            'monaural.blocks.1.temporal.weight_ih_l0','monaural.output.weight','gate.weight'):
        assert not torch.equal(model.state_dict()[name],initial[name]),name


def test_auxiliary_target_trains_independent_frequency_and_time_paths_without_changing_source():
    learn_twice(network().train())


def test_independent_mono_cannot_read_other_microphones_or_source_prediction():
    model=network().train();learn_twice(model);model.eval()
    value=torch.randn(1,11,257,8,2)*.1
    captured={}
    handle=model.monaural.register_forward_hook(lambda module,arguments,result:captured.update(mono=result.clone()))
    try:
        with torch.no_grad():
            model(value);before=captured.pop('mono')
            altered=value.clone();altered[...,1:,:]=torch.randn_like(altered[...,1:,:])*10
            model.base.gain.mul_(2);model(altered)
            assert torch.equal(before,captured.pop('mono'))
    finally:handle.remove()


def test_auxiliary_supervision_reaches_mono_while_initial_zero_gate_blocks_primary_gradient():
    model=network().train();captured={}
    handle=model.monaural.register_forward_hook(lambda module,arguments,result:captured.update(mono=result))
    try:
        value=torch.randn(1,9,257,8,2)*.1;target=torch.randn(1,2,9,257)*.03
        enhanced=model(value)
        (enhanced-target).square().mean().backward(retain_graph=True)
        assert model.monaural.output.weight.grad is not None and torch.count_nonzero(model.monaural.output.weight.grad)==0
        model.zero_grad(set_to_none=True)
        (captured.pop('mono')-target).square().mean().backward()
        assert torch.isfinite(model.monaural.output.weight.grad).all() and model.monaural.output.weight.grad.abs().sum()>0
        assert all(p.grad is None for p in model.base.parameters())
    finally:handle.remove()


def test_updated_future_independence_chunking_and_immutable_disjoint_states():
    model=network().train();learn_twice(model);model.eval()
    value=(torch.randn(1,19,257,8,2)*.1).requires_grad_()
    derivative,=torch.autograd.grad(model(value)[:,:,:8].square().sum(),value)
    assert torch.count_nonzero(derivative[:,8:])==0 and derivative[:,:8].abs().sum()>0
    with torch.no_grad():
        expected=model(value.detach());future=value.detach().clone();future[:,8:]=torch.randn_like(future[:,8:])*10
        assert torch.equal(model(future)[:,:,:8],expected[:,:,:8])
        state=None;pieces=[]
        for a,b in ((0,1),(1,7),(7,9),(9,19)):
            old=copy.deepcopy(state)
            piece,updated=model.forward_stream(value.detach()[:,a:b],state)
            assert all(torch.equal(x,y) for x,y in zip(tensors(old),tensors(state)))
            pieces.append(piece);state=updated
        torch.testing.assert_close(torch.cat(pieces,2),expected,rtol=1e-5,atol=1e-5)
        other,other_state=model.forward_stream(value.detach()[:,:3])
        repeat,repeat_state=model.forward_stream(value.detach()[:,:3])
        assert torch.equal(other,repeat)
        assert {t.untyped_storage().data_ptr() for t in tensors(other_state)}.isdisjoint(
            {t.untyped_storage().data_ptr() for t in tensors(repeat_state)})
        assert sum(t.numel()*t.element_size() for t in tensors(state))==19464
        with pytest.raises(ValueError,match='complete independently owned'):model.forward_stream(value,{'base':{}})
        with pytest.raises(ValueError,match='differs from'):model.forward_stream(value.expand(2,-1,-1,-1,-1),state)


def test_updated_six_seconds_original_frontend_matches600_waveform_hops():
    from light_causal_stream import CausalWaveformStream
    from light_quality_loss import QualityObjective
    from train_light import build_stft_batch
    model=network().train();learn_twice(model);model.eval();wave=torch.randn(1,96000,8)*.02
    with torch.no_grad():
        features,_=build_stft_batch(wave,wave[:,:,0],512,160,320,.5,torch.device('cpu'))
        expected=QualityObjective().reconstruct(model(features),96000)
        stream=CausalWaveformStream(model);stream.warmup();assert stream.state is None
        pieces=[stream.push(wave[:,i:i+160]) for i in range(0,96000,160)];pieces.append(stream.flush())
        actual=torch.cat(pieces,-1)
    assert actual.shape==expected.shape==(1,96000) and (actual-expected).abs().max()<1e-5


def test_real_source_fully_frozen_with_exact_initial_output_and_fixed_cache():
    from light_candidate_runtime import build_model_candidate
    torch.set_num_threads(2);torch.manual_seed(1234)
    source=build_model_candidate('causal_monaural_spectral_v1')
    model=CausalAuxiliaryMonauralFusion(source).train()
    assert all(not p.requires_grad for p in model.base.parameters()) and not model.base.training
    assert all(not layer.training for layer in model.base.modules())
    model.eval();value=torch.randn(1,9,257,8,2)*.1
    with torch.no_grad():
        expected,_=model.base.process(value);assert torch.equal(model(value),expected)
        first,state=model.forward_stream(value[:,:3]);last,updated=model.forward_stream(value[:,3:],state)
        torch.testing.assert_close(torch.cat((first,last),2),expected,rtol=1e-5,atol=1e-5)
        assert sum(t.numel()*t.element_size() for t in tensors(updated))==6236432

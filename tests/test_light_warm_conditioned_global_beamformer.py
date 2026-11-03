"""Warm source preservation, trainable conditioning and causal owned-state waveform behavior."""
import copy

import pytest
import torch
from torch import nn

from experiments.causal_warm_conditioned_global_beamformer import CausalWarmConditionedGlobalBeamformer
from test_light_frequency_state_memory import tensors


class CausalSource(nn.Module):
    def __init__(self):
        super().__init__();self.gain=nn.Parameter(torch.tensor(.4));self.M=8;self.is_causal=True;self.context=1

    def process(self,value,state=None):
        reference=value[...,0,:];cache=state is not None;state={} if state is None else state
        if not isinstance(state,dict) or (state and set(state)!={'previous'}):raise ValueError('Incomplete source state')
        previous=state.get('previous',torch.zeros_like(reference[:,:1]))
        if previous.shape!=reference[:,:1].shape:raise ValueError('Source state differs from this stream')
        history=torch.cat((previous,reference[:,:-1]),1)
        output=(self.gain*(reference+.25*history)).permute(0,3,1,2).contiguous()
        return output,dict(previous=reference[:,-1:].detach().clone()) if cache else None


def network():
    torch.set_num_threads(2);torch.manual_seed(1234)
    return CausalWarmConditionedGlobalBeamformer(CausalSource())


@pytest.mark.parametrize('frames,scale',[(1,0.),(9,1e-6),(33,.1)])
def test_zero_new_readout_preserves_nonzero_source_exactly(frames,scale):
    model=network().eval();value=torch.randn(1,frames,257,8,2)*scale
    with torch.no_grad():
        source,_=model.base.process(value)
        assert torch.equal(model(value),source)
        if scale:assert not torch.equal(source,value[...,0,:].permute(0,3,1,2))


def learn_twice(model):
    before={name:value.detach().clone() for name,value in model.named_parameters()}
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        value=torch.randn(1,9,257,8,2)*.1
        loss=(model(value)-torch.randn(1,2,9,257)*.05).square().mean()
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        optimizer.step()
    assert {int(v['step']) for v in optimizer.state.values()}=={2}
    assert all(torch.isfinite(v).all() for item in optimizer.state.values() for v in item.values())
    for name in ('base.gain','encoders.0.0.weight','recurrent.weight_ih_l0','recurrent.weight_ih_l2','output.weight'):
        assert not torch.equal(before[name],dict(model.named_parameters())[name])


def test_actual_two_updates_reach_source_and_new_global_memory():
    model=network().train();learn_twice(model)
    assert sum(p.numel() for p in model.parameters())==4946369


def test_new_branch_uses_source_estimate_as_a_condition():
    model=network().eval()
    with torch.no_grad():
        model.output.weight.normal_(std=.01)
        value=torch.randn(1,9,257,8,2)*.1
        source,_=model.base.process(value);delta=model(value)-source
        model.base.gain.mul_(1.5)
        changed_source,_=model.base.process(value);changed_delta=model(value)-changed_source
        assert (delta-changed_delta).abs().max()>1e-7


def test_future_independence_chunk_equivalence_and_owned_state():
    model=network().eval()
    with torch.no_grad():model.output.weight.normal_(std=.01)
    value=(torch.randn(1,19,257,8,2)*.1).requires_grad_()
    derivative,=torch.autograd.grad(model(value)[:,:,:8].square().sum(),value)
    assert derivative[:,:8].abs().sum()>0 and torch.count_nonzero(derivative[:,8:])==0
    with torch.no_grad():
        expected=model(value.detach());future=value.detach().clone();future[:,8:]=torch.randn_like(future[:,8:])*10
        assert torch.equal(model(future)[:,:,:8],expected[:,:,:8])
        pieces=[];state=None
        for a,b in ((0,1),(1,8),(8,10),(10,19)):
            saved=[x.clone() for x in tensors(state)]
            piece,updated=model.forward_stream(value.detach()[:,a:b],state)
            assert all(torch.equal(a,b) for a,b in zip(saved,tensors(state)))
            assert sum(x.numel()*x.element_size() for x in tensors(updated))==6664
            state=updated;pieces.append(piece)
        torch.testing.assert_close(torch.cat(pieces,2),expected,rtol=1e-5,atol=1e-5)
        old=copy.deepcopy(state);first,next_state=model.forward_stream(value.detach()[:,:3],state)
        model.forward_stream(torch.randn_like(value));second,other_state=model.forward_stream(value.detach()[:,:3],state)
        assert torch.equal(first,second)
        assert all(torch.equal(a,b) for a,b in zip(tensors(old),tensors(state)))
        assert all(a.data_ptr()!=b.data_ptr() for a,b in zip(tensors(next_state),tensors(other_state)))
        with pytest.raises(ValueError,match='complete independently owned'):model.forward_stream(value,{'base':{}})
        with pytest.raises(ValueError,match='differs from this stream'):model.forward_stream(value.expand(2,-1,-1,-1,-1),state)


def test_updated_complete6s_original_frontend_matches600_hops():
    from light_causal_stream import CausalWaveformStream
    from light_quality_loss import QualityObjective
    from train_light import build_stft_batch
    model=network().train();learn_twice(model);model.eval();waveform=torch.randn(1,96000,8)*.02
    with torch.no_grad():
        features,_=build_stft_batch(waveform,waveform[:,:,0],512,160,320,.5,torch.device('cpu'))
        expected=QualityObjective().reconstruct(model(features),96000)
        stream=CausalWaveformStream(model);stream.warmup()
        pieces=[stream.push(waveform[:,i:i+160]) for i in range(0,96000,160)];pieces.append(stream.flush())
        actual=torch.cat(pieces,-1)
    assert actual.shape==expected.shape==(1,96000) and (actual-expected).abs().max()<1e-5


def test_real_source_architecture_keeps_nested_causal_freeze_policy_and_output():
    from light_candidate_runtime import build_model_candidate
    torch.set_num_threads(2);torch.manual_seed(1234)
    source=build_model_candidate('causal_monaural_spectral_v1')
    policy={name:p.requires_grad for name,p in source.named_parameters()}
    model=CausalWarmConditionedGlobalBeamformer(source).train()
    assert all(p.requires_grad==policy[name] for name,p in model.base.named_parameters())
    assert sum(p.numel() for p in model.parameters())==10632498
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==8052800
    assert not model.base.base.base.training
    model.eval();value=torch.randn(1,9,257,8,2)*.1
    with torch.no_grad():
        expected,_=model.base.process(value);assert torch.equal(model(value),expected)
        left,state=model.forward_stream(value[:,:3]);right,updated=model.forward_stream(value[:,3:],state)
        torch.testing.assert_close(torch.cat((left,right),2),expected,rtol=1e-5,atol=1e-5)
        assert sum(x.numel()*x.element_size() for x in tensors(updated))==6223632

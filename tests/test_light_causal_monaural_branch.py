"""Nonzero warm output, actual learning, causality and independently owned state."""
import copy

import pytest
import torch
from torch import nn

from experiments.causal_monaural_spectral_branch import CausalMonauralSpectralBranch


class CausalStub(nn.Module):
    is_causal=True
    def __init__(self):
        super().__init__();self.scale=nn.Parameter(torch.tensor(.8))
    def process(self,value,state=None):
        reference=value[...,0,:]
        estimate=reference*(self.scale+.05*torch.tanh(reference[...,:1]))
        return estimate.permute(0,3,1,2),({} if state is not None else None)


def model():
    torch.set_num_threads(2);torch.manual_seed(1234)
    return CausalMonauralSpectralBranch(CausalStub())


def tensors(value):
    if torch.is_tensor(value):return [value]
    if isinstance(value,dict):return [v for item in value.values() for v in tensors(item)]
    if isinstance(value,(tuple,list)):return [v for item in value for v in tensors(item)]
    return []


def learn(network):
    network.train();optimizer=torch.optim.Adam(network.parameters(),lr=.001)
    before=network.encoders[0][0].weight.detach().clone()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        value=torch.randn(1,17,257,8,2)*.1
        loss=(network(value)-torch.randn(1,2,17,257)*.05).square().mean()
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in network.parameters() if p.requires_grad)
        if step==0:assert torch.count_nonzero(network.encoders[0][0].weight.grad)==0
        else:assert network.encoders[0][0].weight.grad.abs().sum()>0
        torch.nn.utils.clip_grad_norm_(network.parameters(),5.,error_if_nonfinite=True);optimizer.step()
    assert not torch.equal(network.encoders[0][0].weight,before)
    assert {int(item['step']) for item in optimizer.state.values()}=={2}
    assert all(torch.isfinite(v).all() for item in optimizer.state.values() for v in item.values())


@pytest.mark.parametrize('frames',[1,17,65,131])
def test_monaural_zero_readout_keeps_nonzero_warm_source(frames):
    network=model().eval();value=torch.randn(1,frames,257,8,2)*.1
    with torch.no_grad():
        expected,_=network.base.process(value)
        assert torch.count_nonzero(expected)>0
        torch.testing.assert_close(network(value),expected,rtol=0,atol=0)


def test_monaural_natural_zero_readout_learns_and_keeps_finite_adam():
    network=model();learn(network)


def test_trained_monaural_future_gradient_and_owned_fixed_stream_state():
    network=model();learn(network);network.eval();network.zero_grad(set_to_none=True)
    value=(torch.randn(1,131,257,8,2)*.1).requires_grad_()
    full=network(value);changed=value.detach().clone();changed[:,65:]*=11
    torch.testing.assert_close(full[:,:,:65],network(changed)[:,:,:65],rtol=0,atol=0)
    full[:,:,:65].square().mean().backward()
    assert torch.count_nonzero(value.grad[:,65:])==0 and value.grad[:,:65].abs().sum()>0
    with torch.no_grad():
        state=None;parts=[]
        for start,end in ((0,1),(1,17),(17,64),(64,65),(65,131)):
            part,state=network.forward_stream(value[:,start:end],state);parts.append(part)
        torch.testing.assert_close(torch.cat(parts,2),full,rtol=1e-4,atol=2e-6)
        assert sum(v.untyped_storage().nbytes() for v in tensors(state))==512
        before=[v.clone() for v in tensors(state)]
        _,later=network.forward_stream(value[:,:1],state)
        for old,current in zip(before,tensors(state)):torch.testing.assert_close(old,current,rtol=0,atol=0)
        assert not ({v.data_ptr() for v in tensors(state)} & {v.data_ptr() for v in tensors(later)})
        _,other=network.forward_stream(value[:,:1])
        assert not ({v.data_ptr() for v in tensors(state)} & {v.data_ptr() for v in tensors(other)})


def test_monaural_source_rejection_and_stream_validation():
    bad=CausalStub();bad.is_causal=False
    with pytest.raises(ValueError,match='causal'):CausalMonauralSpectralBranch(bad)
    network=model().eval();value=torch.zeros(1,17,257,8,2)
    with pytest.raises(RuntimeError,match='disabled gradients'):network.forward_stream(value)
    with torch.no_grad():
        with pytest.raises(ValueError,match='complete'):network.forward_stream(value,dict(base={}))
        with pytest.raises(ValueError,match='differs'):network.forward_stream(value,dict(base={},monaural_hidden=torch.zeros(1,1,128,dtype=torch.float64)))
        output,state=network.forward_stream(value)
        assert torch.isfinite(output).all() and torch.isfinite(state['monaural_hidden']).all()


def test_monaural_branch_couples_distant_current_frequencies_beyond_frame_scaling():
    network=model();learn(network);network.eval()
    value=torch.randn(1,17,257,8,2)*.1;changed=value.clone()
    changed[:,:,200,0,0]=-value[:,:,200,0,1]
    changed[:,:,200,0,1]=value[:,:,200,0,0]
    with torch.no_grad():
        torch.testing.assert_close(torch.linalg.vector_norm(value[...,0,:],dim=-1),
            torch.linalg.vector_norm(changed[...,0,:],dim=-1),rtol=0,atol=0)
        original,_=network.base.process(value);altered,_=network.base.process(changed)
        torch.testing.assert_close(original[:,:,:,0],altered[:,:,:,0],rtol=0,atol=0)
        assert torch.count_nonzero(network(value)[:,:,:,0]-network(changed)[:,:,:,0])>0


def test_actual_wide_source_warm_learning_policy_and_future_state():
    from light_candidate_runtime import WIDE_CONDITIONED_CANDIDATE,build_model_candidate
    torch.set_num_threads(2);torch.manual_seed(1234)
    source=build_model_candidate(WIDE_CONDITIONED_CANDIDATE).eval()
    with torch.no_grad():
        for module in (source.base.beamformer,source.base.mapping_output,source.base.retention.output,
                       source.output,source.fullband_projection,source.spatial_projection):module.weight.normal_(std=.003)
    network=CausalMonauralSpectralBranch(copy.deepcopy(source)).eval()
    fixed={key:value.clone() for key,value in network.base.base.state_dict().items()}
    value=torch.randn(1,17,257,8,2)*.1
    with torch.no_grad():torch.testing.assert_close(network(value),source(value),rtol=0,atol=0)
    learn(network)
    assert not network.base.base.training and network.base.training
    assert all(not p.requires_grad for p in network.base.base.parameters())
    assert all(p.requires_grad for name,p in network.base.named_parameters() if not name.startswith('base.'))
    for key,item in network.base.base.state_dict().items():torch.testing.assert_close(item,fixed[key],rtol=0,atol=0)
    network.eval();network.zero_grad(set_to_none=True)
    value=(torch.randn(1,131,257,8,2)*.1).requires_grad_();full=network(value)
    full[:,:,:65].square().mean().backward();assert torch.count_nonzero(value.grad[:,65:])==0
    with torch.no_grad():
        state=None;parts=[]
        for start,end in ((0,1),(1,17),(17,64),(64,65),(65,131)):
            part,state=network.forward_stream(value[:,start:end],state);parts.append(part)
        torch.testing.assert_close(torch.cat(parts,2),full,rtol=1e-4,atol=2e-6)
        assert sum(v.untyped_storage().nbytes() for v in tensors(state))==6219024


def test_actual_source_trained_branch_full6s_waveform_frontend_streaming():
    import evaluate_light as evaluation
    from light_causal_stream import CausalWaveformStream
    from light_candidate_runtime import WIDE_CONDITIONED_CANDIDATE,build_model_candidate
    torch.set_num_threads(2);torch.manual_seed(1234)
    source=build_model_candidate(WIDE_CONDITIONED_CANDIDATE).eval()
    with torch.no_grad():source.output.weight.normal_(std=.003)
    network=CausalMonauralSpectralBranch(source);learn(network);network.eval()
    time=torch.arange(96000,dtype=torch.float32)/16000
    speech=.1*(torch.sin(2*torch.pi*143*time)+.2*torch.sin(2*torch.pi*286*time))
    waveform=speech[None,:,None]+torch.randn(1,96000,8)*.02
    frontend=dict(n_fft=512,hop_length=160,win_length=320,power=.5,device=torch.device('cpu'))
    with torch.inference_mode():
        features=evaluation.build_stft_batch(waveform,**frontend)
        expected=evaluation.reconstruct_waveform(network(features),length=96000,**frontend)
        stream=CausalWaveformStream(network);stream.warmup()
        assert stream.state is None and stream.total_samples==stream.emitted_samples==0
        parts=[stream.push(waveform[:,start:start+160]) for start in range(0,96000,160)]
        parts.append(stream.flush());actual=torch.cat(parts,1)
        assert actual.shape==expected.shape==(1,96000) and torch.isfinite(actual).all()
        assert (actual-expected).abs().max()<1e-5
        assert sum(v.untyped_storage().nbytes() for v in tensors(stream.state))==6219024

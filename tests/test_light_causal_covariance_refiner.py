"""Independent algebra, finite updates and strict prefix/owned-state checks."""
import copy

import pytest
import torch
from torch import nn

from experiments.causal_covariance_refiner import CausalCovarianceRefiner,_covariance_block,_souden_weights


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
    return CausalCovarianceRefiner(CausalStub())


def learn(network):
    network.train();optimizer=torch.optim.Adam(network.parameters(),lr=.001)
    original=network.output.weight.detach().clone()
    for step in range(2):
        value=torch.randn(1,17,257,8,2)*.1
        optimizer.zero_grad(set_to_none=True)
        loss=(network(value)-torch.randn(1,2,17,257)*.05).square().mean()
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in network.parameters() if p.requires_grad)
        if step==0:assert torch.count_nonzero(network.output.weight.grad[0])==0
        else:assert network.output.weight.grad[0].abs().sum()>0
        assert network.output.weight.grad[1].abs().sum()>0
        torch.nn.utils.clip_grad_norm_(network.parameters(),5.,error_if_nonfinite=True);optimizer.step()
    assert not torch.equal(network.output.weight,original)
    assert {int(item['step']) for item in optimizer.state.values()}=={2}
    assert all(torch.isfinite(v).all() for item in optimizer.state.values() for v in item.values())


@pytest.mark.parametrize('frames',[1,17,64])
def test_vectorized_covariance_is_the_independent_per_frame_ema(frames):
    torch.manual_seed(8)
    signal=torch.randn(2,frames,7,8,dtype=torch.complex64)
    mask=torch.rand(2,frames,7)
    initial=torch.randn(2,7,8,8,dtype=torch.complex64)
    speech=initial@initial.mH;noise=2*speech
    powers=torch.tensor(.98).pow(torch.arange(1,frames+1,dtype=torch.float32))
    actual_s,actual_n=_covariance_block(signal,mask,speech,noise,powers)
    expected_s=[];expected_n=[]
    for index in range(frames):
        frame=signal[:,index];outer=frame[...,None]*frame.conj()[...,None,:]
        speech=.98*speech+.02*mask[:,index,...,None,None]*outer
        noise=.98*noise+.02*(1-mask[:,index,...,None,None])*outer
        expected_s.append(speech);expected_n.append(noise)
    torch.testing.assert_close(actual_s,torch.stack(expected_s,1),rtol=2e-6,atol=4e-6)
    torch.testing.assert_close(actual_n,torch.stack(expected_n,1),rtol=2e-6,atol=4e-6)


def test_souden_uses_reference0_conjugation_and_rejects_interferer_without_reference_gain():
    torch.manual_seed(8)
    steering=torch.randn(9,8,dtype=torch.complex64)
    noise_steering=torch.randn(9,8,dtype=torch.complex64)
    speech=steering[...,None]*steering.conj()[...,None,:]
    noise=noise_steering[...,None]*noise_steering.conj()[...,None,:]+.001*torch.eye(8)
    weights=_souden_weights(speech,noise,torch.eye(8,dtype=torch.complex64))
    response=(weights.conj()*steering).sum(-1)
    torch.testing.assert_close(response,steering[:,0],rtol=2e-5,atol=2e-5)
    interferer=(weights.conj()*noise_steering).sum(-1)
    assert interferer.abs().mean()<.01*noise_steering[:,0].abs().mean()
    s,n=speech.cdouble(),noise.cdouble()
    normalized_s=s/(s.diagonal(dim1=-2,dim2=-1).real.sum(-1)/8)[...,None,None]
    normalized_n=n/(n.diagonal(dim1=-2,dim2=-1).real.sum(-1)/8)[...,None,None]
    independent=torch.linalg.inv(normalized_n+.01*torch.eye(8,dtype=torch.complex128))
    product=independent@normalized_s
    expected=product[..., :,0]/product.diagonal(dim1=-2,dim2=-1).sum(-1)[...,None]
    torch.testing.assert_close(weights.cdouble(),expected,rtol=2e-4,atol=2e-5)


@pytest.mark.parametrize('kind',['silence','duplicate_microphones'])
def test_singular_or_silent_inputs_remain_finite(kind):
    network=model().eval()
    value=torch.zeros(1,65,257,8,2)
    if kind=='duplicate_microphones':value=torch.randn(1,65,257,1,2).expand(-1,-1,-1,8,-1).clone()*.1
    with torch.no_grad():
        network.output.bias[1]=.3
        output,state=network.forward_stream(value)
    assert torch.isfinite(output).all()
    assert torch.isfinite(state['speech']).all() and torch.isfinite(state['noise']).all()


@pytest.mark.parametrize('frames',[17,65,131])
def test_zero_readout_preserves_nonzero_causal_source(frames):
    network=model().eval();value=torch.randn(1,frames,257,8,2)*.1
    with torch.no_grad():
        expected,_=network.base.process(value)
        torch.testing.assert_close(network(value),expected,rtol=0,atol=0)
    assert sum(p.numel() for p in network.parameters() if p.requires_grad)==8850
    assert all(not p.requires_grad for p in network.base.parameters())


def test_trained_covariance_branch_has_zero_future_gradient_and_owned_constant_state():
    network=model();original=copy.deepcopy(network.base.state_dict());learn(network);network.eval();network.zero_grad(set_to_none=True)
    value=(torch.randn(1,131,257,8,2)*.1).requires_grad_()
    complete=network(value);changed=value.detach().clone();changed[:,65:]*=11
    torch.testing.assert_close(complete[:,:,:65],network(changed)[:,:,:65],rtol=0,atol=0)
    complete[:,:,:65].square().mean().backward()
    assert torch.count_nonzero(value.grad[:,65:])==0 and value.grad[:,:65].abs().sum()>0
    with torch.no_grad():
        parts=[];state=None
        for start,stop in ((0,1),(1,17),(17,64),(64,65),(65,131)):
            part,state=network.forward_stream(value[:,start:stop],state);parts.append(part)
        torch.testing.assert_close(torch.cat(parts,2),complete,rtol=2e-4,atol=1e-5)
        tensors=[state[key] for key in ('speech','noise','mask_hidden')]
        assert sum(v.untyped_storage().nbytes() for v in tensors)==296064
        before=[v.clone() for v in tensors]
        _,later=network.forward_stream(value[:,:1],state)
        for old,current in zip(before,tensors):torch.testing.assert_close(old,current,rtol=0,atol=0)
        assert sum(later[key].untyped_storage().nbytes() for key in ('speech','noise','mask_hidden'))==296064
        _,other=network.forward_stream(value[:,:1])
        assert not ({v.data_ptr() for v in tensors}&{other[key].data_ptr() for key in ('speech','noise','mask_hidden')})
    for key,tensor in network.base.state_dict().items():torch.testing.assert_close(tensor,original[key],rtol=0,atol=0)


def test_covariance_rejects_noncausal_source_bad_state_and_training_stream():
    with pytest.raises(ValueError,match='causal'):CausalCovarianceRefiner(nn.Linear(2,2))
    network=model();value=torch.randn(1,1,257,8,2)
    with torch.no_grad(),pytest.raises(RuntimeError,match='eval'):network.forward_stream(value)
    network.eval()
    with torch.no_grad(),pytest.raises(ValueError,match='complete'):network.forward_stream(value,{'speech':torch.zeros(1)})
    with torch.no_grad():_,state=network.forward_stream(value)
    bad=dict(state,speech=state['speech'].cdouble())
    with torch.no_grad(),pytest.raises(ValueError,match='Covariance'):network.forward_stream(value,bad)


def test_actual_joint_source_warm_output_nonzero_branch_and_stream_state():
    from light_candidate_runtime import build_model_candidate,JOINT_CONDITIONED_CANDIDATE
    torch.set_num_threads(2);torch.manual_seed(1234)
    source=build_model_candidate(JOINT_CONDITIONED_CANDIDATE).eval()
    with torch.no_grad():
        for module in (source.base.beamformer,source.base.mapping_output,source.output,source.spatial_projection):
            module.weight.normal_(std=.003)
    network=CausalCovarianceRefiner(source).eval()
    value=torch.randn(1,131,257,8,2)*.1
    with torch.no_grad():
        expected=source(value)
        assert not torch.equal(expected,value[...,0,:].permute(0,3,1,2))
        torch.testing.assert_close(network(value),expected,rtol=0,atol=0)
    original={key:tensor.clone() for key,tensor in source.state_dict().items()}
    learn(network);network.eval()
    differentiable=value.detach().clone().requires_grad_()
    network(differentiable)[:,:,:65].square().mean().backward()
    assert torch.count_nonzero(differentiable.grad[:,65:])==0
    assert differentiable.grad[:,:65].abs().sum()>0
    for key,tensor in source.state_dict().items():torch.testing.assert_close(tensor,original[key],rtol=0,atol=0)
    with torch.no_grad():
        complete=network(value);changed=value.clone();changed[:,65:]*=11
        torch.testing.assert_close(complete[:,:,:65],network(changed)[:,:,:65],rtol=0,atol=0)
        parts=[];state=None
        for start,stop in ((0,1),(1,17),(17,64),(64,65),(65,131)):
            part,state=network.forward_stream(value[:,start:stop],state);parts.append(part)
        torch.testing.assert_close(torch.cat(parts,2),complete,rtol=2e-4,atol=1e-5)
        def storage_bytes(value):
            if torch.is_tensor(value):return value.untyped_storage().nbytes()
            if isinstance(value,dict):return sum(storage_bytes(item) for item in value.values())
            if isinstance(value,(tuple,list)):return sum(storage_bytes(item) for item in value)
            return 0
        assert storage_bytes(state)==6415888
        _,later=network.forward_stream(value[:,:1],state)
        assert storage_bytes(later)==6415888
    assert sum(p.numel() for p in network.parameters())==5067530
    assert all(not p.requires_grad for p in network.base.parameters())

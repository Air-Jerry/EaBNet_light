"""Conjugation/distortionless formula, exact prefix recurrence and future gradients."""

import torch

from experiments.causal_souden_beamformer import CausalSoudenBeamformer


def inputs(dtype=torch.complex128):
    torch.set_num_threads(2)
    torch.manual_seed(7360)
    value=torch.complex(torch.randn(1,35,5,3),torch.randn(1,35,5,3)).to(dtype)
    estimate=.6*value[...,0]+.1*value[...,1]
    return value,estimate


def test_souden_conjugation_preserves_rank1_reference_response():
    torch.set_num_threads(2)
    torch.manual_seed(7359)
    steering=torch.polar(torch.ones(1,5,3,dtype=torch.float64),torch.randn(1,5,3,dtype=torch.float64))
    factor=torch.complex(torch.randn(1,5,3,3,dtype=torch.float64),torch.randn(1,5,3,3,dtype=torch.float64))
    noise=factor@factor.mH+.01*torch.eye(3,dtype=torch.complex128)
    speech=steering.unsqueeze(-1)*steering.conj().unsqueeze(-2)
    for ref in (0,2):
        module=CausalSoudenBeamformer(reference_mic=ref)
        weights=module._weights(speech,noise)
        torch.testing.assert_close((weights.conj()*steering).sum(-1),steering[...,ref],rtol=1e-11,atol=1e-11)


def test_souden_prefix_matches_frame_recurrence_and_arbitrary_stream_chunks():
    value,estimate=inputs()
    module=CausalSoudenBeamformer(chunk_size=8)
    expected=[]
    speech=value.new_zeros(1,5,3,3);noise=speech.clone()
    for t in range(35):
        current=value[:,t]
        a=estimate[:,t].abs().square();b=(current[...,0]-estimate[:,t]).abs().square()
        mask=(a/(a+b+1e-12)).clamp(.001,.999)[...,None,None]
        outer=current.unsqueeze(-1)*current.conj().unsqueeze(-2)
        speech=.98*speech+.02*mask*outer;noise=.98*noise+.02*(1-mask)*outer
        regularized=noise+.001*noise.diagonal(dim1=-2,dim2=-1).real.mean(-1)[...,None,None]*torch.eye(3,dtype=noise.dtype)
        transfer=torch.linalg.solve(regularized,speech)
        weights=transfer[..., :,0]/transfer.diagonal(dim1=-2,dim2=-1).sum(-1).real[...,None]
        expected.append((weights.conj()*current).sum(-1))
    full=module(value,estimate)
    torch.testing.assert_close(full,torch.stack(expected,1),rtol=1e-9,atol=1e-9)
    pieces,state=[],None
    for start,end in ((0,1),(1,7),(7,24),(24,35)):
        part,state=module.process(value[:,start:end],estimate[:,start:end],state,cache=True)
        pieces.append(part)
        for tensor in state.values():
            assert tensor.shape==(1,5,3,3) and tensor.untyped_storage().nbytes()==1*5*3*3*16
    torch.testing.assert_close(torch.cat(pieces,1),full,rtol=1e-9,atol=1e-9)


def test_souden_future_perturbations_and_gradients_cannot_change_past():
    value,estimate=inputs(torch.complex64)
    value.requires_grad_();estimate=estimate.detach().requires_grad_()
    module=CausalSoudenBeamformer(chunk_size=8)
    expected=module(value,estimate)
    changed=value.detach().clone();changed[:,17:]*=13
    alternative=estimate.detach().clone();alternative[:,17:]*=19
    torch.testing.assert_close(expected[:,:17],module(changed,alternative)[:,:17],rtol=0,atol=0)
    expected[:,:17].abs().square().mean().backward()
    for tensor in (value,estimate):
        assert torch.isfinite(tensor.grad).all() and torch.count_nonzero(tensor.grad[:,17:])==0
    with torch.no_grad():
        assert torch.count_nonzero(module(torch.zeros_like(value),torch.zeros_like(estimate)))==0

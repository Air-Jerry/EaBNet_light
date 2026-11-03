"""Analytic physical filtering, actual learning and causal streaming for the alternate head."""
import copy

import pytest
import torch

from experiments.causal_global_spectral_beamformer import CausalGlobalSpectralBeamformer


def model():
    torch.set_num_threads(2);torch.manual_seed(1234)
    return CausalGlobalSpectralBeamformer()


@pytest.mark.parametrize('frames',[1,9,33])
def test_zero_physical_weights_preserve_reference(frames):
    network=model().eval();value=torch.randn(1,frames,257,8,2)*.1
    with torch.no_grad():
        torch.testing.assert_close(network(value),value[...,0,:].permute(0,3,1,2),rtol=1e-6,atol=1e-7)
    assert sum(p.numel() for p in network.parameters())==4945728


@pytest.mark.parametrize('case',['quarter_physical_gain','complex_phase','select_microphone1'])
def test_physical_complex_coefficients_have_the_analytic_output(case):
    network=model().eval();value=torch.randn(1,9,257,8,2)*.1
    with torch.no_grad():
        if case=='quarter_physical_gain':
            network.output.bias[0]=-.75;expected=value[...,0,:]*.5
        elif case=='complex_phase':
            network.output.bias[1]=1
            real,imaginary=value[...,0,0],value[...,0,1]
            expected=torch.stack((real+imaginary,imaginary-real),-1)/(2**.25)
        else:
            network.output.bias[0]=-1;network.output.bias[2]=1;expected=value[...,1,:]
        torch.testing.assert_close(network(value),expected.permute(0,3,1,2),rtol=2e-5,atol=1e-6)


@pytest.fixture(scope='module')
def learned_beam():
    network=model().train()
    initial={name:p.detach().clone() for name,p in network.named_parameters()}
    optimizer=torch.optim.Adam(network.parameters(),lr=.001)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss=(network(torch.randn(1,9,257,8,2)*.1)-torch.randn(1,2,9,257)*.05).square().mean()
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in network.parameters())
        optimizer.step()
    for name in ('encoders.0.0.weight','recurrent.weight_ih_l0','recurrent.weight_ih_l2','output.weight'):
        assert not torch.equal(initial[name],dict(network.named_parameters())[name])
    assert {int(item['step']) for item in optimizer.state.values()}=={2}
    assert all(torch.isfinite(value).all() for item in optimizer.state.values() for value in item.values())
    return network.eval()


def test_learned_physical_head_has_no_future_and_owned_fixed_state(learned_beam):
    network=learned_beam;value=(torch.randn(1,19,257,8,2)*.1).requires_grad_()
    derivative,=torch.autograd.grad(network(value)[:,:,:8].square().sum(),value)
    assert derivative[:,:8].abs().sum()>0 and torch.count_nonzero(derivative[:,8:])==0
    with torch.no_grad():
        expected=network(value);future=value.detach().clone();future[:,8:]=torch.randn_like(future[:,8:])*10
        assert torch.equal(network(future)[:,:,:8],expected[:,:,:8])
        pieces=[];state=None
        for start,stop in ((0,1),(1,8),(8,10),(10,19)):
            old=copy.deepcopy(state);piece,new=network.forward_stream(value[:,start:stop],state)
            if state:
                assert torch.equal(state['hidden'],old['hidden'])
                assert state['hidden'].data_ptr()!=new['hidden'].data_ptr()
            assert new['hidden'].numel()*new['hidden'].element_size()==4608
            state=new;pieces.append(piece)
        torch.testing.assert_close(torch.cat(pieces,2),expected,rtol=1e-5,atol=1e-5)


def test_learned_physical_head_full6s_waveform_matches600_hops(learned_beam):
    from light_causal_stream import CausalWaveformStream
    from light_quality_loss import QualityObjective
    from train_light import build_stft_batch
    network=learned_beam;waveform=torch.randn(1,96000,8)*.02
    with torch.no_grad():
        features,_=build_stft_batch(waveform,waveform[:,:,0],512,160,320,.5,torch.device('cpu'))
        expected=QualityObjective().reconstruct(network(features),96000)
        stream=CausalWaveformStream(network);stream.warmup()
        pieces=[stream.push(waveform[:,start:start+160]) for start in range(0,96000,160)]
        pieces.append(stream.flush());actual=torch.cat(pieces,1)
    assert actual.shape==expected.shape==(1,96000) and (actual-expected).abs().max()<1e-5

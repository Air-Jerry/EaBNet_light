"""Physical output, actual learning and complete bounded causal waveform state."""
import copy

import pytest
import torch

from experiments.causal_frequency_state_beamformer import CausalFrequencyStateBeamformer
from experiments.fp16_lstm_training import fp16_lstm_training
from test_light_frequency_state_memory import tensors


def network():
    torch.set_num_threads(2);torch.manual_seed(1234)
    return CausalFrequencyStateBeamformer()


@pytest.mark.parametrize('frames,scale',[(1,0.),(9,1e-6),(33,.1)])
def test_zero_physical_readout_preserves_even_tiny_or_zero_reference(frames,scale):
    model=network().eval();value=torch.randn(1,frames,257,8,2)*scale
    with torch.no_grad():assert torch.equal(model(value),value[...,0,:].permute(0,3,1,2))


def test_known_physical_gain_and_phase_switch_mic():
    model=network().eval();value=torch.randn(1,9,257,8,2)*.1
    with torch.no_grad():
        model.decoder.conv.bias[0]=-.75
        torch.testing.assert_close(model(value),.5*value[...,0,:].permute(0,3,1,2),rtol=1e-5,atol=1e-7)
        model.decoder.conv.bias.zero_();model.decoder.conv.bias[0]=-1.;model.decoder.conv.bias[3]=-1.
        expected=torch.stack((-value[...,1,1],value[...,1,0]),1)
        torch.testing.assert_close(model(value),expected,rtol=1e-5,atol=1e-7)


def train_twice(model):
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    before={name:value.detach().clone() for name,value in model.named_parameters()}
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        value=torch.randn(1,9,257,8,2)*.1
        loss=(model(value)-torch.randn(1,2,9,257)*.05).square().mean()
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    assert {int(item['step']) for item in optimizer.state.values()}=={2}
    assert all(torch.isfinite(v).all() for item in optimizer.state.values() for v in item.values())
    for name in ('encoder.conv.weight','blocks.0.frequency_rnn.weight_ih_l0','blocks.0.hidden_time.weight_ih_l0','blocks.0.cell_time.weight_ih_l0'):
        assert not torch.equal(before[name],dict(model.named_parameters())[name])


def test_actual_two_updates_reach_local_frequency_and_temporal_memories():
    model=network().train();train_twice(model)
    assert sum(p.numel() for p in model.parameters())==504080


def test_future_independence_chunk_equivalence_and_owned_fixed_state():
    model=network().eval()
    with torch.no_grad():model.decoder.conv.weight.normal_(std=.01)
    value=(torch.randn(1,19,257,8,2)*.1).requires_grad_()
    gradient,=torch.autograd.grad(model(value)[:,:,:8].square().sum(),value)
    assert gradient[:,:8].abs().sum()>0 and torch.count_nonzero(gradient[:,8:])==0
    with torch.no_grad():
        expected=model(value.detach());changed=value.detach().clone();changed[:,8:]=torch.randn_like(changed[:,8:])*10
        assert torch.equal(model(changed)[:,:,:8],expected[:,:,:8])
        pieces=[];state=None
        for a,b in ((0,1),(1,8),(8,10),(10,19)):
            old=copy.deepcopy(state);saved=[x.clone() for x in tensors(state)]
            piece,state=model.forward_stream(value.detach()[:,a:b],state);pieces.append(piece)
            assert all(torch.equal(a,b) for a,b in zip(saved,tensors(old)))
            assert sum(x.numel()*x.element_size() for x in tensors(state))==1086576
        torch.testing.assert_close(torch.cat(pieces,2),expected,rtol=1e-5,atol=1e-5)
        saved=[x.clone() for x in tensors(state)]
        first,next_state=model.forward_stream(value.detach()[:,:3],state)
        assert all(torch.equal(a,b) for a,b in zip(saved,tensors(state)))
        model.forward_stream(torch.randn_like(value))
        second,other_state=model.forward_stream(value.detach()[:,:3],state)
        assert torch.equal(first,second)
        assert all(a.data_ptr()!=b.data_ptr() for a,b in zip(tensors(next_state),tensors(other_state)))
        with pytest.raises(ValueError,match='complete independently owned'):model.forward_stream(value,{'encoder':None})
        with pytest.raises(ValueError,match='differs from this stream'):model.forward_stream(value.expand(2,-1,-1,-1,-1),state)


def test_updated_original_frontend_complete6s_matches600_hops():
    from light_causal_stream import CausalWaveformStream
    from light_quality_loss import QualityObjective
    from train_light import build_stft_batch
    model=network().train();train_twice(model);model.eval()
    waveform=torch.randn(1,96000,8)*.02
    with torch.no_grad():
        features,_=build_stft_batch(waveform,waveform[:,:,0],512,160,320,.5,torch.device('cpu'))
        expected=QualityObjective().reconstruct(model(features),96000)
        stream=CausalWaveformStream(model);stream.warmup()
        pieces=[stream.push(waveform[:,i:i+160]) for i in range(0,96000,160)];pieces.append(stream.flush())
        actual=torch.cat(pieces,-1)
    assert actual.shape==expected.shape==(1,96000)
    assert (actual-expected).abs().max()<1e-5
    assert sum(x.numel()*x.element_size() for x in tensors(stream.state))==1086576


def test_lstm_training_precision_context_preserves_cpu_and_restores_on_error():
    original=torch.nn.LSTM.forward;model=torch.nn.LSTM(3,5,batch_first=True)
    value=torch.randn(2,7,3)
    expected=model(value)
    with pytest.raises(RuntimeError,match='test restoration'):
        with fp16_lstm_training():
            actual=model(value)
            assert torch.equal(expected[0],actual[0]) and all(torch.equal(a,b) for a,b in zip(expected[1],actual[1]))
            actual[0].square().mean().backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            raise RuntimeError('test restoration')
    assert torch.nn.LSTM.forward is original

"""Independent causal backbone, trained nonzero paths and actual waveform state."""
import pytest
import torch

from experiments.causal_gated_multichannel_crn import CausalGatedMultichannelCRN
from light_causal_stream import CausalWaveformStream
from light_quality_loss import QualityObjective
from train_light import build_stft_batch


@pytest.fixture
def trained_model():
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    model=CausalGatedMultichannelCRN()
    value=torch.randn(2,31,257,8,2)*.02
    torch.testing.assert_close(model(value),value[...,0,:].permute(0,3,1,2),rtol=0,atol=0)
    fixed={key:tensor.clone() for key,tensor in model.named_buffers()}
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    target=(value[...,0,:]*.6+value[...,7,:]*.2).permute(0,3,1,2)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        (model(value)-target).square().mean().backward()
        assert all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
        optimizer.step()
    for parameter in (model.recurrent[0].frequency[0].weight_hh_l0,
                      model.recurrent[0].temporal[0].weight_hh_l0,
                      model.encoder[0].gate_recurrent.weight_hh_l0):
        assert torch.count_nonzero(parameter.grad)>0
    assert all(torch.equal(value,fixed[key]) for key,value in model.named_buffers())
    assert all(torch.isfinite(value).all() for state in optimizer.state.values() for value in state.values())
    assert sum(p.numel() for p in model.parameters())==405266
    return model


def test_all_mics_and_frequency_time_paths_are_causal_in_train_and_eval(trained_model):
    model=trained_model
    value=(torch.randn(2,79,257,8,2)*.02).requires_grad_()
    changed=value.detach().clone();changed[:,23:]*=13
    for mode in (True,False):
        model.train(mode)
        expected=model(value)
        torch.testing.assert_close(expected[:,:,:23],model(changed)[:,:,:23],rtol=0,atol=0)
    expected[:,:,:23].square().mean().backward()
    assert torch.count_nonzero(value.grad[:,23:])==0
    assert torch.count_nonzero(value.grad[:,:23,:,7])>0


def test_partition_batch_and_owned_stream_state(trained_model):
    model=trained_model.eval()
    value=torch.randn(2,79,257,8,2)*.02
    with torch.inference_mode():
        expected=model(value)
        separate=torch.cat([model(value[i:i+1]) for i in range(2)])
        torch.testing.assert_close(expected,separate,rtol=1e-4,atol=2e-6)
        state=None;pieces=[]
        for begin,end in ((0,1),(1,17),(17,32),(32,79)):
            result,state=model.forward_stream(value[:,begin:end],state);pieces.append(result)
        torch.testing.assert_close(torch.cat(pieces,2),expected,rtol=1e-4,atol=2e-6)
        saved=state['encoder'][0]['history'].clone()
        saved_recurrent=state['recurrent'][0].clone()
        model.forward_stream(value[:,:1],state)
        torch.testing.assert_close(saved,state['encoder'][0]['history'],rtol=0,atol=0)
        torch.testing.assert_close(saved_recurrent,state['recurrent'][0],rtol=0,atol=0)
        with pytest.raises(ValueError,match='state differs'):
            model.forward_stream(value[:1,:1],state)


def test_original_frontend_and_600_actual_waveform_hops(trained_model):
    model=trained_model.eval()
    waveform=torch.randn(1,96000,8)*.02
    with torch.inference_mode():
        features,_=build_stft_batch(waveform,waveform[:,:,0],512,160,320,.5,torch.device('cpu'))
        expected=QualityObjective().reconstruct(model(features),96000)
        stream=CausalWaveformStream(model);stream.warmup()
        parts=[stream.push(waveform[:,begin:begin+160]) for begin in range(0,96000,160)]
        actual=torch.cat([*parts,stream.flush()],1)
        torch.testing.assert_close(actual,expected,rtol=1e-4,atol=2e-6)
        def bytes_used(value):
            if torch.is_tensor(value):
                assert value.untyped_storage().nbytes()==value.numel()*value.element_size()
                return value.untyped_storage().nbytes()
            if isinstance(value,dict):return sum(bytes_used(item) for item in value.values())
            if isinstance(value,(tuple,list)):return sum(bytes_used(item) for item in value)
            return 0
        assert bytes_used(stream.state)==600576


@pytest.mark.parametrize('settings',[dict(M=9),dict(channels=7),dict(channels=10),dict(recurrent_layers=0),dict(is_causal=False)])
def test_incompatible_architecture_is_rejected(settings):
    with pytest.raises(ValueError):CausalGatedMultichannelCRN(**settings)

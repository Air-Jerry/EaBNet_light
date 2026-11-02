"""Lossless frequency organization, learned causality and original frontend."""
import copy

import pytest
import torch

from experiments.causal_folded_retention import CausalFoldedRetention
from evaluate_light import build_stft_batch,reconstruct_waveform
from light_causal_stream import CausalWaveformStream


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(2);torch.manual_seed(8421)


def tensors(value):
    if torch.is_tensor(value):return [value]
    if isinstance(value,dict):return [item for child in value.values() for item in tensors(child)]
    if isinstance(value,(list,tuple)):return [item for child in value for item in tensors(child)]
    return []


def trained():
    model=CausalFoldedRetention(hidden=32,ffn=64,layers=2)
    value=torch.randn(1,11,257,8,2)*.1
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True);model(value).square().mean().backward();optimizer.step()
    return model.eval()


@pytest.mark.parametrize('fold',(1,2,4))
def test_every_bin_and_microphone_survives_frequency_pack_unpack(fold):
    model=CausalFoldedRetention(fold=fold,hidden=32,ffn=64,layers=1)
    value=torch.randn(2,7,257,16,requires_grad=True)
    packed=model.pack(value)
    assert packed.shape==(2,(257+fold-1)//fold,7,16*fold)
    restored=model.unpack(packed)
    torch.testing.assert_close(restored,value,rtol=0,atol=0)
    restored.sum().backward()
    torch.testing.assert_close(value.grad,torch.ones_like(value),rtol=0,atol=0)


def test_default_full_eight_layers_and_all_paths_receive_finite_adam_updates():
    model=CausalFoldedRetention();assert model.groups==65 and len(model.layers)==8
    assert model.encoder.in_channels==64 and model.decoder.out_features==8
    assert model.layers[0].full.weight.shape[-2:]==(65,65)
    value=torch.randn(1,9,257,8,2)*.1
    torch.testing.assert_close(model(value),value[...,0,:].permute(0,3,1,2),rtol=0,atol=0)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True);model(value).square().mean().backward();optimizer.step()
    assert all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    for layer in model.layers:
        assert layer.mhsa.qkv.weight.grad.abs().sum()>0
        assert layer.full.weight.grad.abs().sum()>0
    assert all(torch.isfinite(value).all() for state in optimizer.state.values() for value in state.values())


def test_trained_model_never_uses_future_frames_and_receives_last_microphone():
    model=trained();value=(torch.randn(1,133,257,8,2)*.1).requires_grad_()
    output=model(value);changed=value.detach().clone();changed[:,65:]*=13
    torch.testing.assert_close(output[:,:,:65],model(changed)[:,:,:65],rtol=0,atol=1e-7)
    derivative=torch.autograd.grad(output[:,:,64].square().sum(),value)[0]
    assert torch.count_nonzero(derivative[:,65:])==0
    assert derivative[...,7,:].abs().sum()>0


def test_unequal_chunks_match_and_state_is_owned_and_bounded():
    model=trained();value=torch.randn(2,133,257,8,2)*.1
    with torch.inference_mode():
        expected=model(value);parts=[];state=None
        for begin,end in ((0,1),(1,64),(64,65),(65,133)):
            saved=copy.deepcopy(state);part,updated=model.forward_stream(value[:,begin:end],state)
            for first,second in zip(tensors(saved),tensors(state)):
                torch.testing.assert_close(first,second,rtol=0,atol=0)
            assert not {value.data_ptr() for value in tensors(state)} & {value.data_ptr() for value in tensors(updated)}
            parts.append(part);state=updated
        torch.testing.assert_close(torch.cat(parts,2),expected,rtol=2e-5,atol=2e-6)
        size=sum(value.numel()*value.element_size() for value in tensors(state))
        _,updated=model.forward_stream(value[:,:1],state)
        assert sum(value.numel()*value.element_size() for value in tensors(updated))==size


def test_original_257_bin_waveform_frontend_matches_streaming():
    model=trained();waveform=torch.randn(1,1920,8)*.01
    with torch.inference_mode():
        features=build_stft_batch(waveform,512,160,320,.5,torch.device('cpu'))
        expected=reconstruct_waveform(model(features),1920,512,160,320,.5,torch.device('cpu'))
        stream=CausalWaveformStream(model);parts=[]
        for begin in range(0,1920,160):parts.append(stream.push(waveform[:,begin:begin+160]))
        parts.append(stream.flush())
        torch.testing.assert_close(torch.cat(parts,1),expected,rtol=1e-4,atol=1e-6)


def test_invalid_frequency_packing_and_noncausal_mode_are_rejected():
    for fold in (0,3,True):
        with pytest.raises(ValueError):CausalFoldedRetention(fold=fold)
    with pytest.raises(ValueError):CausalFoldedRetention(M=9)
    with pytest.raises(ValueError):CausalFoldedRetention(is_causal=False)
    model=CausalFoldedRetention(hidden=32,ffn=64,layers=1)
    with pytest.raises(ValueError):model.pack(torch.zeros(1,2,256,16))
    with pytest.raises(RuntimeError):model.forward_stream(torch.zeros(1,1,257,8,2))

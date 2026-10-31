"""Global frequency support without time lookahead, changed warm weights or growing state."""
import pytest
import torch

from experiments.causal_frequency_fourier import CausalFrequencyFourierTCN,FrequencyFourierInput
from experiments.causal_tcn_mapping import CausalTCNMapping
from light_causal_stream import CausalWaveformStream
from light_quality_loss import QualityObjective
from train_light import build_stft_batch

SMALL=dict(hidden=32,ffn=64,layers=2,global_hidden=16,context=5,mapping_channels=8)


def test_fourier_frequency_support_is_global_and_current_frame_only():
    torch.set_num_threads(2);torch.manual_seed(83)
    layer=FrequencyFourierInput(3,4)
    layer.fourier_out.weight.data.normal_(std=.01)
    x=torch.randn(1,3,3,257,requires_grad=True)
    layer(x)[:,:,0,0].sum().backward()
    assert x.grad[:,:,0,100:].abs().max()>1e-9
    assert torch.count_nonzero(x.grad[:,:,1:])==0
    with pytest.raises(ValueError):layer(torch.zeros(1,3,3,129))


def test_zero_fourier_output_preserves_source_and_rng():
    torch.set_num_threads(2);torch.manual_seed(1234)
    original=CausalTCNMapping(**SMALL);rng=torch.get_rng_state()
    torch.manual_seed(1234);changed=CausalFrequencyFourierTCN(**SMALL)
    torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
    original.beamformer.weight.data.normal_(std=.01)
    original.mapping_output.weight.data.normal_(std=.01)
    state=changed.state_dict();state.update(original.state_dict());changed.load_state_dict(state,strict=True)
    assert all(p.requires_grad for p in changed.parameters())
    with torch.inference_mode():
        original.eval();changed.eval();x=torch.randn(1,11,257,8,2)*.02
        torch.testing.assert_close(changed(x),original(x),rtol=0,atol=0)


def nonzero_model():
    model=CausalFrequencyFourierTCN(**SMALL)
    model.beamformer.weight.data.normal_(std=.01)
    model.mapping_output.weight.data.normal_(std=.01)
    model.mapping_input.fourier_out.weight.data.normal_(std=.01)
    return model


def test_nonzero_fourier_branch_has_no_future_gradient():
    torch.set_num_threads(2);torch.manual_seed(87);model=nonzero_model()
    x=(torch.randn(1,17,257,8,2)*.02).requires_grad_()
    other=x.detach().clone();other[:,9:]*=13
    y=model(x)
    torch.testing.assert_close(y[:,:,:9],model(other)[:,:,:9],rtol=0,atol=0)
    y[:,:,:9].square().mean().backward()
    assert torch.count_nonzero(x.grad[:,9:])==0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert torch.count_nonzero(model.mapping_input.fourier_out.weight.grad)>0


def test_nonzero_fourier_branch_matches_chunks_and_waveform_stream():
    torch.set_num_threads(2);torch.manual_seed(91);model=nonzero_model().eval()
    with torch.inference_mode():
        x=torch.randn(1,97,257,8,2)*.02;full=model(x);parts=[];state=None
        for a,b in ((0,1),(1,3),(3,35),(35,63),(63,97)):
            y,state=model.forward_stream(x[:,a:b],state);parts.append(y)
            for layer,history in zip(model.mapping_blocks,state['mapping']):
                assert history.shape==(1,8,layer.history_frames,257)
                assert history.untyped_storage().nbytes()==8*layer.history_frames*257*4
        torch.testing.assert_close(torch.cat(parts,2),full,rtol=1e-4,atol=1e-6)
        waveform=torch.randn(1,1617,8)*.02
        features,_=build_stft_batch(waveform,waveform[:,:,0],512,160,320,.5,torch.device('cpu'))
        expected=QualityObjective().reconstruct(model(features),1617)
        stream=CausalWaveformStream(model);stream.warmup()
        output=[stream.push(waveform[:,i:i+160]) for i in range(0,1617,160)]
        torch.testing.assert_close(torch.cat([*output,stream.flush()],1),expected,rtol=1e-4,atol=1e-6)

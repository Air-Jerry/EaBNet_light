"""CPU contracts of the independent full-resolution retention prototype."""
import copy

import pytest
import torch

from experiments.causal_fullband_retention import CausalFullbandRetention
from evaluate_light import build_stft_batch, reconstruct_waveform
from light_causal_stream import CausalWaveformStream


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(2)
    torch.manual_seed(7341)


def tensors(value):
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        return [item for child in value.values() for item in tensors(child)]
    if isinstance(value, (list, tuple)):
        return [item for child in value for item in tensors(child)]
    return []


def trained_small():
    model = CausalFullbandRetention(hidden=32, ffn=64, layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    value = torch.randn(1, 11, 257, 8, 2) * .1
    target = value[..., 0, :].permute(0, 3, 1, 2) * .6
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        (model(value)-target).square().mean().backward()
        optimizer.step()
    return model.eval()


def test_default_full_model_real_adam_reaches_every_retention_and_frequency_path():
    model = CausalFullbandRetention()
    value = torch.randn(1, 9, 257, 8, 2) * .1
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    torch.testing.assert_close(model(value), value[..., 0, :].permute(0, 3, 1, 2), rtol=0, atol=0)
    assert len(model.layers) == 8
    assert all(layer.full is model.layers[0].full for layer in model.layers)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        model(value).square().mean().backward()
        optimizer.step()
    for layer in model.layers:
        assert layer.mhsa.qkv.weight.grad.abs().sum() > 0
        assert layer.full.weight.grad.abs().sum() > 0
    assert all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(torch.isfinite(value).all() for state in optimizer.state.values() for value in state.values())


def test_nonzero_model_is_causal_across_chunk_boundaries_and_in_input_gradients():
    model = trained_small()
    value = (torch.randn(1, 131, 257, 8, 2)*.1).requires_grad_()
    output = model(value)
    changed = value.detach().clone(); changed[:, 65:] *= 13
    torch.testing.assert_close(output[:, :, :65], model(changed)[:, :, :65], rtol=0, atol=1e-7)
    derivative = torch.autograd.grad(output[:, :, 64].square().sum(), value)[0]
    assert torch.count_nonzero(derivative[:, 65:]) == 0
    assert derivative[:, :65].abs().sum() > 0
    assert derivative[..., 7, :].abs().sum() > 0


def test_activation_checkpointing_preserves_outputs_and_parameter_gradients():
    direct=trained_small().train(); direct.gradient_checkpointing=False
    checkpointed=copy.deepcopy(direct); checkpointed.gradient_checkpointing=True
    direct.zero_grad(set_to_none=True); checkpointed.zero_grad(set_to_none=True)
    value=torch.randn(1, 13, 257, 8, 2)*.1
    left=direct(value); right=checkpointed(value)
    torch.testing.assert_close(left, right, rtol=0, atol=0)
    left.square().mean().backward(); right.square().mean().backward()
    for first, second in zip(direct.parameters(), checkpointed.parameters()):
        torch.testing.assert_close(first.grad, second.grad, rtol=0, atol=0)


def test_nonzero_stream_partitions_match_batch_and_states_are_owned_and_fixed_size():
    model = trained_small()
    value = torch.randn(1, 133, 257, 8, 2)*.1
    with torch.inference_mode():
        expected = model(value)
        outputs=[]; state=None
        for begin, end in ((0, 1), (1, 64), (64, 65), (65, 133)):
            previous=copy.deepcopy(state)
            output, updated=model.forward_stream(value[:, begin:end], state)
            for before, after in zip(tensors(previous), tensors(state)):
                torch.testing.assert_close(before, after, rtol=0, atol=0)
            assert not {item.data_ptr() for item in tensors(state)} & {item.data_ptr() for item in tensors(updated)}
            outputs.append(output); state=updated
        torch.testing.assert_close(torch.cat(outputs, 2), expected, rtol=2e-5, atol=2e-6)
        count=sum(item.numel()*item.element_size() for item in tensors(state))
        _, updated=model.forward_stream(value[:, :1], state)
        assert sum(item.numel()*item.element_size() for item in tensors(updated)) == count


def test_trained_model_matches_the_unchanged_waveform_frontend():
    model=trained_small()
    waveform=torch.randn(1, 1920, 8)*.01
    with torch.inference_mode():
        features=build_stft_batch(waveform, 512, 160, 320, .5, torch.device('cpu'))
        expected=reconstruct_waveform(model(features), 1920, 512, 160, 320, .5, torch.device('cpu'))
        stream=CausalWaveformStream(model); outputs=[]
        for begin in range(0, 1920, 160):outputs.append(stream.push(waveform[:, begin:begin+160]))
        outputs.append(stream.flush())
        torch.testing.assert_close(torch.cat(outputs, 1), expected, rtol=1e-4, atol=1e-6)


def test_invalid_configuration_and_training_stream_are_rejected():
    with pytest.raises(ValueError):CausalFullbandRetention(M=9)
    with pytest.raises(ValueError):CausalFullbandRetention(hidden=33)
    with pytest.raises(ValueError):CausalFullbandRetention(is_causal=False)
    with pytest.raises(RuntimeError):CausalFullbandRetention().forward_stream(torch.zeros(1, 1, 257, 8, 2))

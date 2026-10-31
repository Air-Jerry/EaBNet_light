"""CPU/evaluation paths and global normalization state remain unchanged."""

import pytest
import torch

from experiments.compiled_training_norm import compiled_training_norms, fp32_layer_norm
from light_causal_fast_model import ActivationLayerNorm


@pytest.mark.parametrize('seq_last', [False, True])
def test_norm_preserves_cpu_output_gradients_and_restores_method(seq_last):
    torch.set_num_threads(2)
    layer = ActivationLayerNorm(8, seq_last=seq_last)
    shape = (2, 8, 7) if seq_last else (2, 7, 8)
    value = torch.randn(*shape, requires_grad=True)
    expected = layer(value)
    gradients = torch.autograd.grad(expected.square().sum(), (value, layer.weight, layer.bias))
    original = ActivationLayerNorm.forward
    with compiled_training_norms():
        torch.testing.assert_close(layer(value), expected, rtol=0, atol=0)
        actual = torch.autograd.grad(layer(value).square().sum(), (value, layer.weight, layer.bias))
        for a, b in zip(actual, gradients):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert ActivationLayerNorm.forward is original
    assert all(parameter.dtype == torch.float32 for parameter in layer.parameters())


def test_compile_target_keeps_fp32_statistics_and_input_output_dtype():
    value = torch.randn(2, 7, 8).bfloat16().requires_grad_()
    layer = ActivationLayerNorm(8)
    expected = layer(value)
    actual = fp32_layer_norm(value, layer.weight, layer.bias, layer.eps)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.float().square().mean().backward()
    assert actual.dtype == torch.bfloat16 and torch.isfinite(value.grad).all()
    assert layer.weight.grad.dtype == torch.float32 and torch.isfinite(layer.weight.grad).all()

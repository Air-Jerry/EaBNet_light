"""The training optimization leaves CPU, inference and cache equations unchanged."""

import torch

from experiments.causal_retention_mapping import CausalRetention
from experiments.compiled_training_retention import compiled_training_retention


def test_retention_compilation_preserves_cpu_outputs_gradients_and_method():
    torch.set_num_threads(2)
    torch.manual_seed(7331)
    layer = CausalRetention(16)
    layer.output.weight.data.normal_(std=.02)
    value = torch.randn(2, 16, 145, 9, requires_grad=True)
    expected = layer(value)
    gradients = torch.autograd.grad(expected.square().mean(), (value, *layer.parameters()))
    original = CausalRetention.process
    with compiled_training_retention():
        actual = layer(value)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        observed = torch.autograd.grad(actual.square().mean(), (value, *layer.parameters()))
        for a, b in zip(observed, gradients):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert CausalRetention.process is original


def test_retention_compilation_preserves_eval_cache_and_restores_after_error():
    torch.set_num_threads(2)
    layer = CausalRetention(16).eval()
    layer.output.weight.data.normal_(std=.02)
    value = torch.randn(1, 16, 81, 9)
    original = CausalRetention.process
    with torch.inference_mode():
        expected, memory = layer.process(value, cache=True)
        try:
            with compiled_training_retention():
                actual, observed = layer.process(value, cache=True)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for key in memory:
                    torch.testing.assert_close(memory[key], observed[key], rtol=0, atol=0)
                raise RuntimeError('synthetic context failure')
        except RuntimeError as error:
            assert str(error) == 'synthetic context failure'
    assert CausalRetention.process is original

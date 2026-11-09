"""Nonzero readout, gradients and restoration for the original retention equation."""
import copy

import pytest
import torch

from experiments.causal_retention_mapping import CausalRetention
from experiments.retention_training_chunks import retention_training_chunks


@pytest.mark.parametrize('chunk_size', [16, 32])
def test_training_partition_preserves_output_gradient_and_saved_state(chunk_size):
    torch.set_num_threads(2)
    torch.manual_seed(152)
    original = CausalRetention(16)
    with torch.no_grad():
        original.output.weight.normal_(std=.02)
    changed = copy.deepcopy(original)
    value = torch.randn(1, 16, 601, 5, requires_grad=True)
    other = value.detach().clone().requires_grad_()
    expected = original(value)
    expected.square().mean().backward()
    with retention_training_chunks(changed, chunk_size):
        actual = changed(other)
        actual.square().mean().backward()
        assert changed.chunk_size == chunk_size
    assert changed.chunk_size == 64
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(other.grad, value.grad, rtol=1e-4, atol=1e-7)
    for (name, parameter), (other_name, candidate) in zip(original.named_parameters(), changed.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(candidate.grad, parameter.grad, rtol=1e-4, atol=1e-7)
    assert all(torch.equal(value, changed.state_dict()[name]) for name, value in original.state_dict().items())
    original.eval()
    changed.eval()
    with torch.no_grad():
        assert torch.equal(original(value), changed(value))


def test_chunk_context_restores_on_failure_and_rejects_eval_or_invalid_sizes():
    model = CausalRetention(16, chunk_size=23)
    with pytest.raises(RuntimeError, match='interrupted'):
        with retention_training_chunks(model, 16):
            assert model.chunk_size == 16
            raise RuntimeError('interrupted')
    assert model.chunk_size == 23
    for invalid in (True, 0, -1, 1.5):
        with pytest.raises(ValueError):
            with retention_training_chunks(model, invalid):
                pass
    model.eval()
    with pytest.raises(ValueError, match='training epoch'):
        with retention_training_chunks(model, 16):
            pass
    assert model.chunk_size == 23

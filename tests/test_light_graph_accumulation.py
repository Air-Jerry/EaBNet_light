"""Replaying an overwritten gradient buffer must not replace earlier microbatch contributions."""
from contextlib import nullcontext

import torch

from experiments.graphed_model_training import _independent_accumulated_gradients


class ReplayedBackward(torch.autograd.Function):
    buffer = torch.zeros(3)

    @staticmethod
    def forward(ctx, parameter, value):
        ctx.save_for_backward(value)
        return parameter * value

    @staticmethod
    def backward(ctx, derivative):
        value, = ctx.saved_tensors
        ReplayedBackward.buffer.copy_(derivative * value)
        return ReplayedBackward.buffer.detach(), None


def test_replay_storage_reproduces_failure_and_independent_gradients_preserve_all_microbatches():
    first, second = torch.tensor([1., 2., 3.]), torch.tensor([4., 5., 6.])
    for protected in (False, True):
        model = torch.nn.Linear(3, 1, bias=False)
        model.weight = torch.nn.Parameter(torch.ones(3))
        previous_hooks = len(model.weight._backward_hooks or {})
        context = _independent_accumulated_gradients(model) if protected else nullcontext()
        with context:
            for _ in range(2):
                model.zero_grad(set_to_none=True)
                ReplayedBackward.apply(model.weight, first).sum().backward()
                aliases = model.weight.grad.data_ptr() == ReplayedBackward.buffer.data_ptr()
                assert aliases is not protected
                ReplayedBackward.apply(model.weight, second).sum().backward()
                assert torch.equal(model.weight.grad, first + second if protected else 2 * second)
        assert len(model.weight._backward_hooks or {}) == previous_hooks

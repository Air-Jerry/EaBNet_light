"""CPU and evaluation remain unchanged; scope restoration also survives errors."""
import pytest
import torch

from experiments.fp16_gru_training import fp16_gru_training


@pytest.mark.parametrize('training',[False,True])
def test_cpu_gru_keeps_outputs_gradients_parameter_objects_and_precision(training):
    torch.manual_seed(1234);module=torch.nn.GRU(5,7,batch_first=True).train(training)
    value=torch.randn(2,11,5,requires_grad=True)
    expected=module(value);parameters=list(module.parameters())
    with fp16_gru_training():
        actual=module(value)
    for old,new in zip(expected,actual):torch.testing.assert_close(old,new,rtol=0,atol=0)
    actual[0].square().mean().backward()
    assert all(p.dtype==torch.float32 and p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
    assert all(old is new for old,new in zip(parameters,module.parameters()))


def test_gru_scope_restores_original_after_failure():
    original=torch.nn.GRU.forward
    with pytest.raises(RuntimeError,match='input.size'):
        with fp16_gru_training():torch.nn.GRU(5,7)(torch.randn(3,2,6))
    assert torch.nn.GRU.forward is original

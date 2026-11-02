"""Frame validation and restoration of a per-epoch compiled-forward wrapper."""
import pytest
import torch

from experiments.causal_folded_retention import CausalFoldedRetention
from experiments.compiled_folded_training import compiled_folded_forward


def test_forward_lengths_weights_and_exception_restoration(monkeypatch):
    torch.set_num_threads(2);torch.manual_seed(41)
    model=CausalFoldedRetention(hidden=32,ffn=64,layers=1)
    torch.nn.init.normal_(model.decoder.weight,std=.01)
    value=torch.randn(1,7,257,8,2)*.1
    original=model.forward;expected=model(value,frame_lengths=[7])
    settings=torch._dynamo.config.optimize_ddp;calls=[];cache={}
    def compile_cpu_contract(function,**options):
        calls.append(options)
        return function
    monkeypatch.setattr(torch,'compile',compile_cpu_contract)
    with compiled_folded_forward(model,cache):
        torch.testing.assert_close(model(value,frame_lengths=[7]),expected,rtol=0,atol=0)
        with pytest.raises(ValueError):model(value,frame_lengths=[0])
        with pytest.raises(ValueError):model(value,frame_lengths=[8])
        with pytest.raises(ValueError):model(value,frame_lengths=[7,7])
    assert model.forward==original and torch._dynamo.config.optimize_ddp==settings
    with pytest.raises(RuntimeError,match='synthetic exit'):
        with compiled_folded_forward(model,cache):raise RuntimeError('synthetic exit')
    assert len(calls)==1 and model.forward==original
    assert torch._dynamo.config.optimize_ddp==settings
    assert calls[0]['fullgraph'] and calls[0]['options']['force_same_precision']

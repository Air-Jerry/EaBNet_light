import copy
import pytest
import torch

from experiments.frozen_source_training_precision import frozen_source_training_precision
from test_light_residual_retention_head import models


def test_source_mode_and_all_flags_restore_while_real_head_updates_keep_source_fixed():
    _,model=models();reference=copy.deepcopy(model).train()
    values=torch.randn(1,17,257,8,2)*.1
    before={k:v.clone() for k,v in model.base.state_dict().items()}
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    with frozen_source_training_precision(model):
        assert model.train() is model and model.base.training
        assert all(not p.requires_grad for p in model.base.parameters())
        assert torch.equal(reference(values),model(values))
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True);(model(values)-.1).square().mean().backward();optimizer.step()
        assert all(p.grad is None for p in model.base.parameters())
        assert all(torch.equal(v,model.base.state_dict()[k]) for k,v in before.items())
        model.eval();assert not model.base.training
    assert 'train' not in model.__dict__ and not model.base.training
    model.train();assert not model.base.training


@pytest.mark.parametrize('fault',['unfrozen','dropout','dropout2d','recurrent_dropout','running_stats'])
def test_stateful_or_unfrozen_source_rejected_without_overriding_train(fault):
    _,model=models()
    if fault=='unfrozen':next(model.base.parameters()).requires_grad_(True)
    if fault=='dropout':model.base.invalid=torch.nn.Dropout(.1)
    if fault=='dropout2d':model.base.invalid=torch.nn.Dropout2d(.1)
    if fault=='recurrent_dropout':model.base.invalid=torch.nn.GRU(4,4,num_layers=2,dropout=.1)
    if fault=='running_stats':model.base.invalid=torch.nn.BatchNorm1d(4)
    if fault!='unfrozen':model.base.invalid.requires_grad_(False)
    with pytest.raises(ValueError,match='Freeze every' if fault=='unfrozen' else 'running statistics'):
        with frozen_source_training_precision(model):pass
    assert 'train' not in model.__dict__


def test_training_mode_restores_after_an_exception_inside_the_scope():
    _,model=models()
    with pytest.raises(RuntimeError,match='stop test'):
        with frozen_source_training_precision(model):
            model.train();assert model.base.training
            raise RuntimeError('stop test')
    assert 'train' not in model.__dict__ and not model.base.training
    assert all(not p.requires_grad for p in model.base.parameters())

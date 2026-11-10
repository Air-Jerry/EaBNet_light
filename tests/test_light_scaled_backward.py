"""Exercise the real weighted accumulation, clipping and finite-update guard."""
import copy

import pytest
import torch

import train_light
from experiments.scaled_backward import backward_loss_scaling


class ScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.25))

    def forward(self, values, frame_lengths=None):
        return self.weight * values


def settings():
    return train_light.parse_args(['--batch-size','1','--grad-accum-steps','2','--log-perf','no',
        '--mem-log-every-batches','0','--malloc-trim-every-batches','0','--gc-every-batches','0'])


def batches():
    return [dict(mixture=torch.tensor([[value]]),target=torch.zeros(1,1),
                 lengths=torch.tensor([length]),sample_ids=torch.tensor([index]))
            for index,(value,length) in enumerate([(2.,160),(3.,480),(4.,800)])]


def run(model, optimizer, scaler, records, objective, args=None):
    return train_light.run_epoch(model,records,optimizer,scaler,torch.device('cpu'),[],args or settings(),True,
        loss_fn=objective,feature_builder=lambda mixture,target,**kw:(mixture,target),
        correct_accumulation=True,loss_weighting='frames',fail_nonfinite_grad=True,forward_with_lengths=True)


def test_scale_preserves_actual_weighted_tail_Adam_updates_and_checkpoint_resume():
    models=[ScalarModel(),ScalarModel()]
    optimizers=[torch.optim.Adam(m.parameters(),lr=.01) for m in models]
    original_factory=train_light.GradScaler
    scalers=[torch.amp.GradScaler('cpu',enabled=False)]
    with backward_loss_scaling(train_light,1/1024):
        scalers.append(train_light.GradScaler(device='cpu',enabled=False))
    assert train_light.GradScaler is original_factory
    objective=lambda estimate,target,frames:(estimate-target).square().mean()
    for epoch in range(3):
        for model,optimizer,scaler in zip(models,optimizers,scalers):
            run(model,optimizer,scaler,batches(),objective)
        assert torch.equal(models[0].weight,models[1].weight)
        for key,value in optimizers[0].state[models[0].weight].items():
            assert torch.equal(value,optimizers[1].state[models[1].weight][key])
        assert scalers[1].get_scale()==1/1024
        if epoch==0:
            saved=(copy.deepcopy(models[1].state_dict()),copy.deepcopy(optimizers[1].state_dict()),scalers[1].state_dict())
            models[1]=ScalarModel();optimizers[1]=torch.optim.Adam(models[1].parameters(),lr=.01)
            models[1].load_state_dict(saved[0]);optimizers[1].load_state_dict(saved[1])
            with backward_loss_scaling(train_light,1/1024):
                scalers[1]=train_light.GradScaler('cpu',enabled=False)
            scalers[1].load_state_dict(saved[2])


def test_half_backward_overflow_is_prevented_and_finite_guard_stays_enabled():
    class HalfBoundary(torch.autograd.Function):
        @staticmethod
        def forward(ctx,value): return value.clone()
        @staticmethod
        def backward(ctx,gradient): return gradient.half().float()
    record=dict(mixture=torch.ones(1,1),target=torch.zeros(1,1),lengths=torch.tensor([96000]),sample_ids=torch.tensor([0]))
    objective=lambda estimate,target,frames:HalfBoundary.apply(estimate).sum()*200
    args=settings();args.grad_clip=5.
    model=ScalarModel();optimizer=torch.optim.SGD(model.parameters(),lr=.01)
    with pytest.raises(FloatingPointError,match='Non-finite gradient'):
        run(model,optimizer,torch.amp.GradScaler('cpu',enabled=False),[record],objective,args)
    assert model.weight.item()==.25
    with backward_loss_scaling(train_light,1/1024):
        scaler=train_light.GradScaler(device='cpu',enabled=False)
    run(model,optimizer,scaler,[record],objective,args)
    assert model.weight.item()==pytest.approx(.2,abs=1e-7)
    assert scaler.get_scale()==1/1024


@pytest.mark.parametrize('factor',[0,-1,.3,2,float('nan'),float('inf')])
def test_bad_factors_rejected_without_patching_trainer(factor):
    original=train_light.GradScaler
    with pytest.raises(ValueError,match='power-of-two'):
        with backward_loss_scaling(train_light,factor):pass
    assert train_light.GradScaler is original


def test_scaler_factory_restores_on_failure_and_rejects_outer_amp():
    original=train_light.GradScaler
    with pytest.raises(ValueError,match='outer AMP'):
        with backward_loss_scaling(train_light,1/1024):
            train_light.GradScaler(device='cpu',enabled=True)
    assert train_light.GradScaler is original


def test_unrecoverable_gradient_still_aborts_scaled_update():
    class BrokenGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx,value):return value.clone()
        @staticmethod
        def backward(ctx,gradient):return torch.full_like(gradient,float('nan'))
    current=ScalarModel();optimizer=torch.optim.Adam(current.parameters(),lr=.01)
    with backward_loss_scaling(train_light,1/1024):
        scaler=train_light.GradScaler('cpu',enabled=False)
    with pytest.raises(FloatingPointError,match='Non-finite gradient'):
        run(current,optimizer,scaler,batches()[:1],
            lambda estimate,target,frames:BrokenGradient.apply(estimate).sum())
    assert current.weight.item()==.25 and not optimizer.state

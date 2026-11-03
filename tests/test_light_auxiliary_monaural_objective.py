"""Original waveform losses, real branch gradients, padding and hook ownership."""
import pytest
import torch

from experiments.auxiliary_monaural_objective import AuxiliaryMonauralObjective
from experiments.causal_auxiliary_monaural import CausalAuxiliaryMonauralFusion
from train_light_causal_pmsqe import FrontendPMSQE
from test_light_warm_conditioned_global_beamformer import CausalSource


def batch():
    generator=torch.Generator().manual_seed(341)
    target=torch.randn(2,1921,generator=generator)*.03
    mixture=target[...,None].repeat(1,1,8)+torch.randn(2,1921,8,generator=generator)*.02
    return mixture,target,[1601,1921],[11,13]


def network():
    torch.set_num_threads(2);torch.manual_seed(1234)
    return CausalAuxiliaryMonauralFusion(CausalSource()).train()


def features(objective,mixture,target,lengths):
    return objective.build_features(mixture,target,512,160,320,.5,'cpu',lengths=lengths)


def test_actual_auxiliary_loss_opens_zero_gate_mono_gradient_and_preserves_source():
    model=network();mixture,target,lengths,frames=batch()
    frozen={name:t.clone() for name,t in model.base.state_dict().items()}
    primary=FrontendPMSQE(512,160,320,.5)
    value,reference=features(primary,mixture,target,lengths)
    primary(model(value),reference,frames).backward()
    assert torch.count_nonzero(model.monaural.output.weight.grad)==0
    model.zero_grad(set_to_none=True)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    with AuxiliaryMonauralObjective(model.monaural) as objective:
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            value,reference=features(objective,mixture,target,lengths)
            loss=objective(model(value),reference,frames)
            assert torch.isfinite(loss)
            loss.backward()
            assert model.monaural.output.weight.grad.abs().sum()>0
            assert all(p.grad is None for p in model.base.parameters())
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
            optimizer.step()
        assert objective.stats()['samples']==4 and objective.stats()['batches']==2
        assert objective.stats()['auxiliary_monaural_total']>0
        assert objective._captured is None and not objective._armed
    assert not model.monaural._forward_hooks
    assert all(torch.equal(value,model.base.state_dict()[name]) for name,value in frozen.items())


def test_zero_weight_is_exact_original_loss_and_gradient_without_changing_frontend():
    model=network();mixture,target,lengths,frames=batch()
    original=FrontendPMSQE(512,160,320,.5)
    value,reference=features(original,mixture,target,lengths)
    expected=original(model(value),reference,frames)
    expected_gradient,=torch.autograd.grad(expected,model.gate.weight)
    with AuxiliaryMonauralObjective(model.monaural,weight=0) as objective:
        other,other_reference=features(objective,mixture,target,lengths)
        assert torch.equal(value,other) and torch.equal(reference,other_reference)
        actual=objective(model(other),other_reference,frames)
        actual_gradient,=torch.autograd.grad(actual,model.gate.weight)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        torch.testing.assert_close(actual_gradient,expected_gradient,rtol=0,atol=0)
        assert objective._auxiliary.stats()['samples']==0


def test_auxiliary_padding_is_masked_and_batch_capture_is_consumed_once():
    model=network();mixture,target,lengths,frames=batch()
    with AuxiliaryMonauralObjective(model.monaural) as objective:
        value,reference=features(objective,mixture,target,lengths);estimate=model(value)
        mono=objective._captured.detach().clone().requires_grad_();objective._captured=mono
        mono.data[0,:,frames[0]:]=1e20
        loss=objective(estimate,reference,frames);loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(mono.grad).all()
        assert torch.count_nonzero(mono.grad[0,:,frames[0]:])==0
        with pytest.raises(RuntimeError,match='Build features'):objective(estimate,reference,frames)
        features(objective,mixture,target,lengths)
        with pytest.raises(RuntimeError,match='previous feature batch'):features(objective,mixture,target,lengths)
        model(value)
        with pytest.raises(RuntimeError,match='one monaural forward'):model(value)
    assert objective._captured is None and not model.monaural._forward_hooks


def test_exception_removes_hook_and_invalid_weights_reject():
    model=network()
    for weight in (-1,float('nan'),float('inf')):
        with pytest.raises(ValueError,match='weight'):AuxiliaryMonauralObjective(model.monaural,weight)
    with pytest.raises(RuntimeError,match='deliberate'):
        with AuxiliaryMonauralObjective(model.monaural):raise RuntimeError('deliberate')
    assert not model.monaural._forward_hooks

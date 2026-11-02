"""Compare actual loss values, gradients, stats, masking and PMSQE inheritance."""
from contextlib import nullcontext

import pytest
import torch

from experiments.batched_quality_training import batched_quality_training
from experiments.causal_pmsqe_objective import PMSQEObjective
from light_quality_loss import QualityObjective


@pytest.mark.parametrize('lengths',[[1536],[1536,1536],[960,960,960],[1536,960]])
@pytest.mark.parametrize('perceptual',[False,True])
def test_batched_loss_keeps_per_utterance_values_gradients_and_stats(lengths,perceptual):
    torch.set_num_threads(2);torch.manual_seed(1234)
    mixture=torch.randn(len(lengths),max(lengths),8)*.03
    target=torch.randn(len(lengths),max(lengths))*.02
    results=[]
    for batched in (False,True):
        loss=PMSQEObjective() if perceptual else QualityObjective()
        _,reference=loss.build_features(mixture,target,512,160,320,.5,'cpu',lengths)
        estimate=(reference*.83+.001).detach().requires_grad_()
        with batched_quality_training() if batched else nullcontext():
            value=loss(estimate,reference,[n//160+1 for n in lengths])
            value.backward()
        results.append((value.detach(),estimate.grad,loss.stats()))
    old,new=results
    torch.testing.assert_close(old[0],new[0],rtol=2e-6,atol=2e-7)
    torch.testing.assert_close(old[1],new[1],rtol=1e-4,atol=3e-6)
    assert old[2].keys()==new[2].keys()
    for key in old[2]:assert new[2][key]==pytest.approx(old[2][key],rel=2e-6,abs=2e-7)


def test_context_restores_original_loss_and_preserves_invalid_input_error():
    original=QualityObjective.__call__
    with pytest.raises(RuntimeError,match='Call build_features'):
        with batched_quality_training():
            QualityObjective()(torch.zeros(1,2,10,257),torch.zeros(1,2,10,257),[10])
    assert QualityObjective.__call__ is original

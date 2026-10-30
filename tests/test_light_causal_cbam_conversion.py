"""Causality and preserved weights of the trained-CBAM conversion prototype."""

import pytest
import torch

from experiments.probe_causal_cbam_conversion import CausalCBAMConversion
from light_reference_variants import build_reference_candidate


@pytest.mark.parametrize('training',[False,True])
def test_all_converted_paths_are_causal_with_fixed_bn_statistics(training):
    torch.set_num_threads(2);torch.manual_seed(17)
    source=build_reference_candidate('cbam_flat_projection64')
    weights={name:value.clone() for name,value in source.state_dict().items()}
    model=CausalCBAMConversion(source).train(training)
    for name,value in model.net.state_dict().items():torch.testing.assert_close(value,weights[name],rtol=0,atol=0)
    assert all(not child.training for child in model.modules() if isinstance(child,torch.nn.BatchNorm2d))
    value=(torch.randn(1,19,257,8,2)*.1).requires_grad_()
    changed=value.detach().clone();changed[:,11:]*=12
    first=model(value);second=model(changed)
    torch.testing.assert_close(first[:,:,:11],second[:,:,:11],rtol=0,atol=0)
    first[:,:,:11].square().mean().backward()
    assert torch.count_nonzero(value.grad[:,11:])==0
    assert torch.count_nonzero(value.grad[:,:11])>0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)

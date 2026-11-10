"""Checkpoint recomputation must preserve real joint-model gradients, Adam and inference."""
import copy

import pytest
import torch

from experiments.causal_unfrozen_root_monaural import CausalUnfrozenRootMonaural
from experiments.checkpoint_spatial_training import checkpoint_spatial_training
from light_candidate_runtime import build_model_candidate


def model():
    torch.set_num_threads(2);torch.manual_seed(1234)
    result=CausalUnfrozenRootMonaural(build_model_candidate('causal_wide_conditioned_subband_v1'))
    with torch.no_grad():
        for module in (result.base.base.beamformer,result.base.base.mapping_output,
                       result.base.base.retention.output,result.base.output,result.output):
            module.weight.normal_(std=.01)
    return result


def test_recomputed_joint_blocks_preserve_real_input_parameter_gradients_and_Adam():
    reference=model().train();recomputed=copy.deepcopy(reference)
    assert recomputed.base.base.layers[0].full is recomputed.base.base.layers[3].full
    identities=[id(p) for p in recomputed.parameters()]
    optimizers=[torch.optim.Adam(m.parameters(),lr=1e-4) for m in (reference,recomputed)]
    features=torch.randn(1,17,257,8,2)*.1
    with checkpoint_spatial_training(recomputed.base.base):
        for _ in range(2):
            inputs=[features.clone().requires_grad_(),features.clone().requires_grad_()]
            predictions=[]
            for current,optimizer,value in zip((reference,recomputed),optimizers,inputs):
                optimizer.zero_grad(set_to_none=True)
                prediction=current(value);predictions.append(prediction.detach())
                (prediction-.1).square().mean().backward()
            assert torch.equal(predictions[0],predictions[1])
            torch.testing.assert_close(inputs[0].grad,inputs[1].grad,rtol=1e-6,atol=1e-8)
            assert reference.base.base.stem.conv.weight.grad.abs().sum()>0
            for a,b in zip(reference.parameters(),recomputed.parameters()):
                assert (a.grad is None)==(b.grad is None)
                if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,rtol=1e-6,atol=1e-8)
            for optimizer in optimizers:optimizer.step()
            for a,b in zip(reference.parameters(),recomputed.parameters()):
                torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-8)
    assert [id(p) for p in recomputed.parameters()]==identities
    assert all('forward' not in layer.__dict__ for layer in recomputed.base.base.layers)
    assert reference.state_dict().keys()==recomputed.state_dict().keys()


def test_scope_keeps_eval_exact_and_restores_layer_methods_after_failure():
    current=model().eval();features=torch.randn(1,9,257,8,2)*.1
    with torch.inference_mode():expected=current(features)
    with pytest.raises(RuntimeError,match='scope test'):
        with checkpoint_spatial_training(current.base.base):
            with torch.inference_mode():assert torch.equal(current(features),expected)
            raise RuntimeError('scope test')
    assert all('forward' not in layer.__dict__ for layer in current.base.base.layers)
    with torch.inference_mode():assert torch.equal(current(features),expected)

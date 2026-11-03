"""Exact inference inheritance, actual root updates and causal independent state."""
import copy

import torch

from experiments.causal_joint_monaural import CausalJointMonaural
from experiments.causal_frame_graph import tensor_leaves
from light_candidate_runtime import build_model_candidate,candidate_metadata


def models():
    torch.set_num_threads(2);torch.manual_seed(231)
    original=build_model_candidate('causal_monaural_spectral_v1')
    with torch.no_grad():
        original.base.base.beamformer.weight.normal_(std=.01)
        original.base.output.weight.normal_(std=.01)
        original.output.weight.normal_(std=.01)
    joint=CausalJointMonaural(build_model_candidate('causal_wide_conditioned_subband_v1'))
    joint.load_state_dict(original.state_dict(),strict=True)
    return original,joint


def test_exact_original_inference_parameters_metadata_and_cache():
    before=candidate_metadata('causal_monaural_spectral_v1')
    original,joint=models();original.eval();joint.eval()
    value=torch.randn(1,13,257,8,2)*.1
    assert sum(p.numel() for p in joint.parameters())==5686130
    assert all(p.requires_grad for p in joint.parameters())
    with torch.no_grad():
        assert torch.equal(original(value),joint(value))
        a,state=joint.forward_stream(value[:,:3]);old=copy.deepcopy(state)
        b,updated=joint.forward_stream(value[:,3:],state)
        torch.testing.assert_close(torch.cat((a,b),2),joint(value),rtol=1e-5,atol=1e-5)
        assert all(torch.equal(x,y) for x,y in zip(tensor_leaves(old),tensor_leaves(state)))
        _,other=joint.forward_stream(value[:,:3])
        assert {t.untyped_storage().data_ptr() for t in tensor_leaves(state)}.isdisjoint(
            {t.untyped_storage().data_ptr() for t in tensor_leaves(other)})
        assert sum(t.numel()*t.element_size() for t in tensor_leaves(updated))==6219024
    assert candidate_metadata('causal_monaural_spectral_v1')==before


def test_actual_adam_updates_root_with_all_modules_in_training_mode():
    _,joint=models();joint.train()
    assert all(module.training for module in joint.modules())
    original={name:t.clone() for name,t in joint.state_dict().items()}
    optimizer=torch.optim.Adam(joint.parameters(),lr=1e-4)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        joint(torch.randn(1,7,257,8,2)*.1).square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in joint.parameters())
        optimizer.step()
    assert not torch.equal(original['base.base.beamformer.weight'],joint.base.base.beamformer.weight)
    assert not torch.equal(original['base.base.stem.conv.weight'],joint.base.base.stem.conv.weight)
    assert {int(v['step']) for v in optimizer.state.values()}=={2}
    assert all(torch.isfinite(t).all() for item in optimizer.state.values() for t in item.values())
    joint.eval();assert all(not module.training for module in joint.modules())


def test_nonzero_model_cannot_read_future_frames():
    _,joint=models();joint.eval();value=(torch.randn(1,13,257,8,2)*.1).requires_grad_()
    output=joint(value)
    derivative,=torch.autograd.grad(output[:,:,:5].square().sum(),value)
    assert torch.count_nonzero(derivative[:,5:])==0
    with torch.no_grad():
        altered=value.detach().clone();altered[:,5:]=torch.randn_like(altered[:,5:])*10
        assert torch.equal(joint(altered)[:,:,:5],output[:,:,:5])

"""CPU bootstrap/data-interface checks; these do not stand in for actual CUDA replay."""
import pytest
import torch

from experiments.causal_frame_graph import CausalFrameGraph,checked_pairs,map_state,tensor_leaves
from experiments.causal_auxiliary_monaural import CausalAuxiliaryMonauralFusion
from test_light_warm_conditioned_global_beamformer import CausalSource


def test_checked_tree_rejects_invalid_state_before_any_copy_and_clones_are_independent():
    destination={'a':torch.ones(1,2),'b':(torch.zeros(1,3),)}
    source=map_state(destination,lambda value:value.clone()+2)
    pairs=checked_pairs(destination,source)
    assert len(pairs)==2 and all(a.data_ptr()!=b.data_ptr() for a,b in pairs)
    invalid={'a':source['a'],'b':(torch.ones(1,4),)}
    with pytest.raises(ValueError,match='differs from'):checked_pairs(destination,invalid)
    assert torch.equal(destination['a'],torch.ones(1,2))
    with pytest.raises(ValueError,match='differs from'):checked_pairs(destination,{'a':source['a']})
    with pytest.raises(ValueError,match='tensor-only'):tensor_leaves({'counter':1})


@pytest.mark.parametrize('real_source',[False,True])
def test_zero_bootstrap_matches_true_initial_state_for_auxiliary_core(real_source):
    torch.set_num_threads(2);torch.manual_seed(1234)
    if real_source:
        from light_candidate_runtime import build_model_candidate
        source=build_model_candidate('causal_monaural_spectral_v1')
    else:source=CausalSource()
    core=CausalAuxiliaryMonauralFusion(source).eval()
    with torch.no_grad():
        core.gate.weight.normal_(std=.01);core.monaural.output.weight.normal_(std=.01)
        _,template=core.forward_stream(torch.zeros(1,1,257,8,2))
        initial=map_state(template,torch.zeros_like)
        value=torch.randn(1,5,257,8,2)*.1
        first,first_state=core.forward_stream(value)
        other,other_state=core.forward_stream(value,initial)
        torch.testing.assert_close(first,other,rtol=1e-5,atol=1e-5)
        for a,b in checked_pairs(first_state,other_state):torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-5)
        assert all(torch.count_nonzero(value)==0 for value in tensor_leaves(initial))


def test_cpu_and_training_interfaces_preserve_core_values_gradients_and_owned_state():
    torch.set_num_threads(2);torch.manual_seed(1234)
    core=CausalAuxiliaryMonauralFusion(CausalSource())
    model=CausalFrameGraph(core).train();value=torch.randn(1,7,257,8,2)*.1
    actual=model(value);torch.testing.assert_close(actual,core(value),rtol=0,atol=0)
    actual.square().mean().backward();assert core.gate.weight.grad is not None
    model.eval()
    with torch.no_grad():
        expected=core(value)
        first,state=model.forward_stream(value[:,:3]);last,updated=model.forward_stream(value[:,3:],state)
        torch.testing.assert_close(torch.cat((first,last),2),expected,rtol=1e-5,atol=1e-5)
        assert model.graph_static_state_storage_bytes==0
        assert sum(v.numel()*v.element_size() for v in tensor_leaves(updated))==19464

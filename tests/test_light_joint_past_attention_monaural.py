import copy

import pytest
import torch

from experiments.causal_frame_graph import tensor_leaves
from experiments.causal_joint_past_attention_monaural import CausalJointPastAttentionMonaural
from experiments.joint_attention_adam import restore_joint_attention_adam
from light_candidate_runtime import build_model_candidate, RECURRENT_ADAPT_CANDIDATE, WIDE_CONDITIONED_CANDIDATE


def models():
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    source = build_model_candidate(RECURRENT_ADAPT_CANDIDATE).eval()
    with torch.no_grad():
        for module in (source.base.base.beamformer, source.base.base.mapping_output,
                       source.base.base.retention.output, source.base.output, source.output):
            module.weight.normal_(std=.01)
    current = CausalJointPastAttentionMonaural(build_model_candidate(WIDE_CONDITIONED_CANDIDATE)).eval()
    current.load_warm(source.state_dict())
    return source, current


def test_warm_output_and_original_training_scope_are_exact():
    source, current = models()
    original = list(source.named_parameters())
    new = list(current.named_parameters())
    assert [n for n, _ in new[:len(original)]] == [n for n, _ in original]
    assert all(a.requires_grad == b.requires_grad for (_, a), (_, b) in zip(original, new))
    assert sum(p.numel() for p in current.parameters()) == 6105122
    assert sum(p.numel() for p in current.parameters() if p.requires_grad) == 5520378
    assert len(new) - len(original) == 4
    with torch.inference_mode():
        value = torch.randn(1, 131, 257, 8, 2) * .1
        assert torch.equal(source(value), current(value))


def test_learning_uses_the_existing_scope_and_new_attention_without_future_frames():
    _, current = models()
    current.train()
    fixed = {n: p.detach().clone() for n, p in current.named_parameters() if not p.requires_grad}
    before = current.temporal_memory.qkv.weight.detach().clone()
    optimizer = torch.optim.Adam(current.parameters(), lr=1e-4)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        value = torch.randn(1, 19, 257, 8, 2) * .1
        (current(value) - .1).square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in current.parameters() if p.requires_grad)
        optimizer.step()
    assert not torch.equal(before, current.temporal_memory.qkv.weight)
    assert all(torch.equal(value, dict(current.named_parameters())[name]) for name, value in fixed.items())
    current.eval()
    value = (torch.randn(1, 131, 257, 8, 2) * .1).requires_grad_()
    current(value)[:, :, :67].square().sum().backward()
    assert torch.count_nonzero(value.grad[:, 67:]) == 0
    with torch.inference_mode():
        expected = current(value.detach())
        changed = value.detach().clone()
        changed[:, 67:] *= 100
        assert (current(changed)[:, :, :67] - expected[:, :, :67]).abs().max() < 1e-6


def test_active_attention_uses_independent_fixed_stream_state():
    _, current = models()
    with torch.no_grad():
        current.temporal_memory.output.weight.normal_(std=.01)
    with torch.inference_mode():
        value = torch.randn(1, 259, 257, 8, 2) * .1
        expected = current(value)
        parts, state = [], None
        for start, end in ((0, 1), (1, 17), (17, 128), (128, 129), (129, 259)):
            before = [t.clone() for t in tensor_leaves(state)] if state else []
            part, updated = current.forward_stream(value[:, start:end], state)
            if state:
                assert all(torch.equal(a, b) for a, b in zip(before, tensor_leaves(state)))
                assert {t.untyped_storage().data_ptr() for t in tensor_leaves(state)}.isdisjoint(
                    {t.untyped_storage().data_ptr() for t in tensor_leaves(updated)})
            parts.append(part)
            state = updated
            assert sum(t.numel() * t.element_size() for t in tensor_leaves(state)) == 6349199
        assert (expected - torch.cat(parts, 2)).abs().max() < 1e-5


def test_old_adam_is_preserved_and_only_new_attention_states_are_empty():
    source, current = models()
    optimizer = torch.optim.Adam(source.parameters(), lr=1e-4)
    for p in source.parameters():
        if p.requires_grad:
            optimizer.state[p] = dict(step=torch.tensor(107200.), exp_avg=torch.randn_like(p) * .001,
                                      exp_avg_sq=torch.rand_like(p) * .001)
    state = dict(epoch=16, args={}, model_state_dict=source.state_dict(), optimizer_state_dict=optimizer.state_dict())
    target = torch.optim.Adam(current.parameters(), lr=1e-4)
    result = restore_joint_attention_adam(target, current, source, state)
    assert result['existing_state_parameters'] == 235 and result['new_attention_parameter_tensors'] == 4
    assert result['source_adam_steps'] == 107200
    original = copy.deepcopy(target.state_dict())
    with pytest.raises(ValueError, match='formal'):
        restore_joint_attention_adam(target, current, source, dict(state, diagnostic_only=True))
    assert original['param_groups'] == target.state_dict()['param_groups']
    assert all(torch.equal(value, target.state_dict()['state'][index][key])
               for index, item in original['state'].items() for key, value in item.items())

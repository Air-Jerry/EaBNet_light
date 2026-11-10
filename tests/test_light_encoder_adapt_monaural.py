"""Check learning through frozen recurrent kernels and unchanged causal inference."""
import copy

import torch

from experiments.causal_encoder_adapt_monaural import CausalEncoderAdaptMonaural
from experiments.causal_frame_graph import tensor_leaves
from experiments.checkpoint_spatial_training import checkpoint_spatial_training
from light_candidate_runtime import build_model_candidate, RECURRENT_ADAPT_CANDIDATE, WIDE_CONDITIONED_CANDIDATE


def models():
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    source = build_model_candidate(RECURRENT_ADAPT_CANDIDATE).eval()
    with torch.no_grad():
        for module in (source.base.base.beamformer, source.base.base.mapping_output,
                       source.base.base.retention.output, source.base.output, source.output):
            module.weight.normal_(std=.01)
    current = CausalEncoderAdaptMonaural(build_model_candidate(WIDE_CONDITIONED_CANDIDATE)).eval()
    current.load_state_dict(source.state_dict(), strict=True)
    return source, current


def test_warm_identity_and_encoder_learning_through_fixed_recurrent_kernels():
    source, current = models()
    value = torch.randn(1, 19, 257, 8, 2) * .1
    assert list(source.state_dict()) == list(current.state_dict())
    assert sum(p.numel() for p in current.parameters()) == 5686130
    assert sum(p.numel() for p in current.parameters() if p.requires_grad) == 5201930
    old_parameters = dict(source.named_parameters())
    new_names = {name for name, p in current.named_parameters()
                 if p.requires_grad and not old_parameters[name].requires_grad}
    assert len(new_names) == 15
    assert sum(dict(current.named_parameters())[name].numel() for name in new_names) == 100544
    assert all(name.startswith(('base.base.stem.', 'base.base.encoders.')) for name in new_names)
    with torch.inference_mode():
        assert torch.equal(source(value), current(value))
    fixed = {name: p.detach().clone() for name, p in current.named_parameters() if not p.requires_grad}
    encoders = [current.base.base.stem, *current.base.base.encoders]
    before = [layer.conv.weight.detach().clone() for layer in encoders]
    current.train()
    optimizer = torch.optim.Adam(current.parameters(), lr=1e-4)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        (current(value) - .1).square().mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in current.parameters() if p.grad is not None)
        assert all(p.grad is None for p in current.parameters() if not p.requires_grad)
        optimizer.step()
    assert all(not torch.equal(old, layer.conv.weight) for old, layer in zip(before, encoders))
    assert all(torch.equal(old, dict(current.named_parameters())[name]) for name, old in fixed.items())
    assert all(torch.isfinite(v).all() for state in optimizer.state.values() for v in state.values())


def test_optional_recomputation_preserves_encoder_and_decoder_gradients():
    _, ordinary = models()
    ordinary.train()
    recomputed = copy.deepcopy(ordinary)
    value = torch.randn(1, 13, 257, 8, 2) * .1
    expected = ordinary(value)
    expected.square().mean().backward()
    with checkpoint_spatial_training(recomputed.base.base):
        actual = recomputed(value)
        actual.square().mean().backward()
    assert torch.equal(actual, expected)
    for first, second in zip(ordinary.parameters(), recomputed.parameters()):
        if first.grad is None:
            assert second.grad is None
        else:
            torch.testing.assert_close(second.grad, first.grad, rtol=1e-5, atol=1e-7)
    assert all('forward' not in layer.__dict__ for layer in recomputed.base.base.layers)


def test_causal_prefix_and_owned_stream_state_with_no_extra_cache():
    _, current = models()
    value = (torch.randn(1, 131, 257, 8, 2) * .1).requires_grad_()
    current(value)[:, :, :67].square().sum().backward()
    assert torch.count_nonzero(value.grad[:, 67:]) == 0
    with torch.inference_mode():
        expected = current(value.detach())
        changed = value.detach().clone()
        changed[:, 67:] *= 100
        assert (current(changed)[:, :, :67] - expected[:, :, :67]).abs().max() < 1e-6
        parts, sizes, state = [], [], None
        for start, end in ((0, 1), (1, 17), (17, 67), (67, 131)):
            before = [x.clone() for x in tensor_leaves(state)] if state else []
            part, updated = current.forward_stream(value.detach()[:, start:end], state)
            if state:
                assert all(torch.equal(a, b) for a, b in zip(before, tensor_leaves(state)))
                assert {x.untyped_storage().data_ptr() for x in tensor_leaves(state)}.isdisjoint(
                    {x.untyped_storage().data_ptr() for x in tensor_leaves(updated)})
            parts.append(part)
            state = updated
            sizes.append(sum(x.numel() * x.element_size() for x in tensor_leaves(state)))
        assert set(sizes) == {6219024}
        assert (expected - torch.cat(parts, 2)).abs().max() < 1e-5

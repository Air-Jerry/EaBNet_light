import torch

from experiments.causal_frame_graph import tensor_leaves
from experiments.causal_post_recurrent_adapt_monaural import CausalPostRecurrentAdaptMonaural
from light_candidate_runtime import build_model_candidate, RECURRENT_ADAPT_CANDIDATE, WIDE_CONDITIONED_CANDIDATE


def models():
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    source = build_model_candidate(RECURRENT_ADAPT_CANDIDATE).eval()
    with torch.no_grad():
        for module in (source.base.base.beamformer, source.base.base.mapping_output,
                       source.base.base.retention.output, source.base.output, source.output):
            module.weight.normal_(std=.01)
    current = CausalPostRecurrentAdaptMonaural(build_model_candidate(WIDE_CONDITIONED_CANDIDATE)).eval()
    current.load_state_dict(source.state_dict(), strict=True)
    return source, current


def test_warm_function_and_parameter_names_are_exact():
    source, current = models()
    original = dict(source.named_parameters())
    extra = {name for name, p in current.named_parameters()
             if p.requires_grad and not original[name].requires_grad}
    assert list(current.state_dict()) == list(source.state_dict())
    assert sum(p.numel() for p in current.parameters()) == 5686130
    assert sum(p.numel() for p in current.parameters() if p.requires_grad) == 5181162
    assert len(extra) == 14
    assert all(name.startswith('base.base.layers.2.tconvffn.') for name in extra)
    assert sum(dict(current.named_parameters())[name].numel() for name in extra) == 79776
    with torch.inference_mode():
        value = torch.randn(1, 19, 257, 8, 2) * .1
        assert torch.equal(current(value), source(value))


def test_new_temporal_weights_learn_without_frozen_gru_backward_graphs():
    _, current = models()
    current.train()
    fixed = {name: p.detach().clone() for name, p in current.named_parameters() if not p.requires_grad}
    new = current.base.base.layers[2].tconvffn
    before = {name: p.detach().clone() for name, p in new.named_parameters()}
    graphs = []
    handles = [layer.mhsa.register_forward_hook(
        lambda module, args, output: graphs.append((args[0].requires_grad, output[0].requires_grad)))
        for layer in current.base.base.layers[:3]]
    value = torch.randn(1, 17, 257, 8, 2) * .1
    optimizer = torch.optim.Adam(current.parameters(), lr=1e-4)
    try:
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            (current(value) - .1).square().mean().backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in new.parameters())
            assert all(p.grad is None for p in current.parameters() if not p.requires_grad)
            optimizer.step()
    finally:
        for handle in handles:
            handle.remove()
    assert graphs == [(False, False)] * 9
    assert all(not torch.equal(before[name], p) for name, p in new.named_parameters())
    assert all(torch.equal(value, dict(current.named_parameters())[name]) for name, value in fixed.items())
    assert all(torch.isfinite(value).all() for state in optimizer.state.values() for value in state.values())


def test_adapted_temporal_weights_keep_causality_and_owned_fixed_cache():
    _, current = models()
    with torch.no_grad():
        for p in current.base.base.layers[2].tconvffn.parameters():
            p.add_(torch.randn_like(p) * .001)
    value = (torch.randn(1, 131, 257, 8, 2) * .1).requires_grad_()
    current(value)[:, :, :67].square().sum().backward()
    assert torch.count_nonzero(value.grad[:, 67:]) == 0
    with torch.inference_mode():
        expected = current(value.detach())
        changed = value.detach().clone()
        changed[:, 67:] *= 100
        assert (current(changed)[:, :, :67] - expected[:, :, :67]).abs().max() < 1e-6
        parts, state = [], None
        for start, end in ((0, 1), (1, 17), (17, 67), (67, 131)):
            incoming = [x.clone() for x in tensor_leaves(state)] if state else []
            part, updated = current.forward_stream(value.detach()[:, start:end], state)
            if state:
                assert all(torch.equal(a, b) for a, b in zip(incoming, tensor_leaves(state)))
                assert {x.untyped_storage().data_ptr() for x in tensor_leaves(state)}.isdisjoint(
                    {x.untyped_storage().data_ptr() for x in tensor_leaves(updated)})
            parts.append(part)
            state = updated
            assert sum(x.numel() * x.element_size() for x in tensor_leaves(state)) == 6219024
        assert (expected - torch.cat(parts, 2)).abs().max() < 1e-5

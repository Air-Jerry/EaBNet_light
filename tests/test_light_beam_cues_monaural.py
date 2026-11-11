import torch

from experiments.causal_beam_cues_monaural import CausalBeamCuesMonaural
from experiments.causal_frame_graph import tensor_leaves
from light_candidate_runtime import RECURRENT_ADAPT_CANDIDATE, WIDE_CONDITIONED_CANDIDATE, build_model_candidate


def test_beam_cues_inherit_original_state_and_stream_after_learning():
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    source = build_model_candidate(RECURRENT_ADAPT_CANDIDATE).eval()
    with torch.no_grad():
        source.output.weight.normal_(std=.01)
        source.base.base.beamformer.weight.normal_(std=.01)
    model = CausalBeamCuesMonaural(build_model_candidate(WIDE_CONDITIONED_CANDIDATE))
    model.load_warm(source.state_dict())
    old_parameters = [(name, p.requires_grad) for name, p in source.named_parameters()]
    assert [(name, p.requires_grad) for name, p in model.named_parameters()] == [
        *old_parameters, ('beam_cues.weight', True)]
    assert model.beam_cues.weight.numel() == 1280
    example = torch.randn(1, 25, 257, 8, 2) * .1
    model.eval()
    with torch.inference_mode():
        assert torch.equal(model(example), source(example))

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    (model(example) - .1).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    optimizer.step()
    assert torch.count_nonzero(model.beam_cues.weight)
    model.eval()
    with torch.inference_mode():
        full = model(example)
        changed = example.clone()
        changed[:, 13:] *= 10
        assert (model(changed)[:, :, :13] - full[:, :, :13]).abs().max() < 1e-6
        left, cache = model.forward_stream(example[:, :13])
        snapshots = [t.clone() for t in tensor_leaves(cache)]
        right, updated = model.forward_stream(example[:, 13:], cache)
        assert all(torch.equal(a, b) for a, b in zip(snapshots, tensor_leaves(cache)))
        assert {t.data_ptr() for t in tensor_leaves(cache)}.isdisjoint(t.data_ptr() for t in tensor_leaves(updated))
        assert sum(t.numel() * t.element_size() for t in tensor_leaves(updated)) == 6219024
        assert (torch.cat((left, right), 2) - full).abs().max() < 1e-5

import torch

from experiments.causal_amplitude_refinement import CausalAmplitudeRefinement
from light_candidate_runtime import PARALLEL_PHASE_FILTER_CANDIDATE, WIDE_CONDITIONED_CANDIDATE, build_model_candidate


def test_amplitude_refinement_preserves_source_and_streams_after_learning():
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    source = build_model_candidate(PARALLEL_PHASE_FILTER_CANDIDATE).eval()
    with torch.no_grad():
        source.output.weight.normal_(std=.01)
        source.base.base.beamformer.weight.normal_(std=.01)
    model = CausalAmplitudeRefinement(build_model_candidate(WIDE_CONDITIONED_CANDIDATE)).eval()
    missing, unexpected = model.load_state_dict(source.state_dict(), strict=False)
    assert set(missing) == {name for name, _ in model.named_parameters() if name.startswith('amplitude_head.')}
    assert not unexpected
    mixture = torch.randn(1, 16, 257, 8, 2) * .1
    with torch.inference_mode():
        initial = source(mixture)
        assert torch.equal(model(mixture), initial)
    optimizer = torch.optim.Adam(model.amplitude_head.parameters(), lr=.01)
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        (model(mixture) - 1.2*initial.clone()).square().mean().backward()
        optimizer.step()
    assert all(torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad)
               for p in model.amplitude_head.parameters())
    with torch.inference_mode():
        estimate = model(mixture)
        assert not torch.equal(estimate, initial)
        changed = mixture.clone()
        changed[:, 7:] *= 10
        torch.testing.assert_close(model(changed)[:, :, :7], estimate[:, :, :7], rtol=0, atol=1e-6)
        first, state = model.forward_stream(mixture[:, :7])
        last, _ = model.forward_stream(mixture[:, 7:], state)
        torch.testing.assert_close(torch.cat((first, last), dim=2), estimate, rtol=0, atol=1e-5)

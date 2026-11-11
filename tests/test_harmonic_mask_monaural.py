import torch

from experiments.causal_harmonic_mask_monaural import CausalHarmonicMaskMonaural
from light_candidate_runtime import RECURRENT_ADAPT_CANDIDATE, WIDE_CONDITIONED_CANDIDATE, build_model_candidate


def test_harmonic_mask_warm_output_and_causal_stream():
    torch.set_num_threads(2)
    source = build_model_candidate(RECURRENT_ADAPT_CANDIDATE).eval()
    with torch.no_grad():
        source.base.base.beamformer.weight.normal_(std=.01)
        source.output.weight.normal_(std=.01)
    model = CausalHarmonicMaskMonaural(build_model_candidate(WIDE_CONDITIONED_CANDIDATE)).eval()
    model.load_warm(source.state_dict())
    example = torch.randn(1, 24, 257, 8, 2) * .1
    with torch.inference_mode():
        assert torch.equal(model(example), source(example))
    model.train()
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        model(example).square().mean().backward()
        with torch.no_grad():
            for parameter in model.harmonic_mask.parameters():
                parameter.add_(parameter.grad, alpha=-1e-3)
    assert all(torch.isfinite(p.grad).all() for p in model.harmonic_mask.parameters())
    assert torch.count_nonzero(model.harmonic_mask[-1].weight)
    model.eval()
    with torch.inference_mode():
        expected = model(example)
        changed = example.clone()
        changed[:, 13:] *= 10
        torch.testing.assert_close(model(changed)[:, :, :13], expected[:, :, :13], rtol=0, atol=1e-6)
        left, state = model.forward_stream(example[:, :13])
        right, _ = model.forward_stream(example[:, 13:], state)
        torch.testing.assert_close(torch.cat((left, right), dim=2), expected, rtol=0, atol=1e-5)

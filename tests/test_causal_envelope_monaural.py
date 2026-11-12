import torch

from experiments.causal_envelope_monaural import CausalEnvelopeMonaural
from light_candidate_runtime import RECURRENT_ADAPT_CANDIDATE, WIDE_CONDITIONED_CANDIDATE, build_model_candidate


def test_envelope_warm_start_and_causal_stream():
    torch.set_num_threads(2)
    source = build_model_candidate(RECURRENT_ADAPT_CANDIDATE).eval()
    with torch.no_grad():
        source.base.base.beamformer.weight.normal_(std=.01)
        source.output.weight.normal_(std=.01)
    model = CausalEnvelopeMonaural(build_model_candidate(WIDE_CONDITIONED_CANDIDATE)).eval()
    model.load_warm(source.state_dict())
    value = torch.randn(1, 23, 257, 8, 2) * .1
    with torch.inference_mode():
        assert torch.equal(model(value), source(value))
    assert sum(p.numel() for p in model.envelope.parameters()) == 131121
    model.train()
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        model(value).square().mean().backward()
        with torch.no_grad():
            for p in model.envelope.parameters():
                p.add_(p.grad, alpha=-1e-3)
    assert all(torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad) for p in model.envelope.parameters())
    model.eval()
    with torch.inference_mode():
        expected = model(value)
        changed = value.clone()
        changed[:, 12:] *= 10
        torch.testing.assert_close(model(changed)[:, :, :12], expected[:, :, :12], rtol=0, atol=1e-6)
        left, state = model.forward_stream(value[:, :12])
        old = state['envelope_hidden'].clone()
        right, next_state = model.forward_stream(value[:, 12:], state)
        assert torch.equal(state['envelope_hidden'], old)
        assert next_state['envelope_hidden'].data_ptr() != state['envelope_hidden'].data_ptr()
        assert old.numel() * old.element_size() == 384
        torch.testing.assert_close(torch.cat((left, right), dim=2), expected, rtol=0, atol=1e-5)

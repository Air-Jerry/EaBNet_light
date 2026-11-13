import torch

from experiments.causal_phase_filter_monaural import CausalPhaseFilterMonaural
from light_candidate_runtime import (
    BAND_GAIN_PHASE_FILTER_CANDIDATE, DOUBLE_PHASE_FILTER_CANDIDATE, PARALLEL_PHASE_FILTER_CANDIDATE, RECURRENT_ADAPT_CANDIDATE,
    WIDE_CONDITIONED_CANDIDATE, build_model_candidate,
)


def test_phase_filter_preserves_warm_start_and_streams_after_training():
    torch.set_num_threads(2)
    source = build_model_candidate(RECURRENT_ADAPT_CANDIDATE).eval()
    with torch.no_grad():
        source.base.base.beamformer.weight.normal_(std=.01)
        source.output.weight.normal_(std=.01)
    model = CausalPhaseFilterMonaural(build_model_candidate(WIDE_CONDITIONED_CANDIDATE)).eval()
    model.load_warm(source.state_dict())
    mixture = torch.randn(1, 19, 257, 8, 2) * .1
    with torch.inference_mode():
        initial = model(mixture)
        assert torch.equal(initial, source(mixture))
    optimizer = torch.optim.Adam(model.phase_filter.parameters(), lr=.001)
    target = mixture[..., 0, :].permute(0, 3, 1, 2) * .8
    model.train()
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        (model(mixture) - target).square().mean().backward()
        optimizer.step()
    assert all(torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad)
               for p in model.phase_filter.parameters())
    model.eval()
    with torch.inference_mode():
        expected = model(mixture)
        assert not torch.equal(expected, initial)
        doubled = build_model_candidate(DOUBLE_PHASE_FILTER_CANDIDATE).eval()
        doubled.load_state_dict(model.state_dict())
        base = source(mixture)
        parallel = build_model_candidate(PARALLEL_PHASE_FILTER_CANDIDATE).eval()
        parallel.load_state_dict(model.state_dict())
        residual = expected-base
        correction = (base*residual).sum(1,keepdim=True)*base / base.square().sum(1,keepdim=True).clamp_min(1e-8)
        torch.testing.assert_close(parallel(mixture), expected+correction, rtol=0, atol=1e-6)
        first_parallel, state_parallel = parallel.forward_stream(mixture[:, :9])
        last_parallel, _ = parallel.forward_stream(mixture[:, 9:], state_parallel)
        torch.testing.assert_close(torch.cat((first_parallel, last_parallel), dim=2),
                                   parallel(mixture), rtol=0, atol=1e-5)
        band_gain = build_model_candidate(BAND_GAIN_PHASE_FILTER_CANDIDATE).eval()
        band_gain.load_state_dict(model.state_dict())
        torch.testing.assert_close(band_gain(mixture), parallel(mixture)*band_gain.band_gain, rtol=0, atol=1e-6)
        first_band, state_band = band_gain.forward_stream(mixture[:, :9])
        last_band, _ = band_gain.forward_stream(mixture[:, 9:], state_band)
        torch.testing.assert_close(torch.cat((first_band, last_band), dim=2),
                                   band_gain(mixture), rtol=0, atol=1e-5)
        torch.testing.assert_close(doubled(mixture), base + 2*(expected-base), rtol=0, atol=1e-6)
        first_double, double_state = doubled.forward_stream(mixture[:, :9])
        last_double, _ = doubled.forward_stream(mixture[:, 9:], double_state)
        torch.testing.assert_close(torch.cat((first_double, last_double), dim=2),
                                   doubled(mixture), rtol=0, atol=1e-5)
        changed = mixture.clone()
        changed[:, 9:] *= 10
        torch.testing.assert_close(model(changed)[:, :, :9], expected[:, :, :9], rtol=0, atol=1e-6)
        first, state = model.forward_stream(mixture[:, :1])
        history = state['phase_history'].clone()
        middle, next_state = model.forward_stream(mixture[:, 1:9], state)
        last, _ = model.forward_stream(mixture[:, 9:], next_state)
        assert torch.equal(history, state['phase_history'])
        assert next_state['phase_history'].data_ptr() != state['phase_history'].data_ptr()
        assert history.numel() * history.element_size() == 8224
        torch.testing.assert_close(torch.cat((first, middle, last), dim=2), expected, rtol=0, atol=1e-5)

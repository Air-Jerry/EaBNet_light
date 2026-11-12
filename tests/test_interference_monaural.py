import torch
import pytest

from experiments.causal_interference_monaural import (
    CausalInterferenceMonaural,
    compress_spectrum,
    interference_loss,
)
from light_candidate_runtime import RECURRENT_ADAPT_CANDIDATE, WIDE_CONDITIONED_CANDIDATE, build_model_candidate
from experiments.causal_multiscale_interference import CausalMultiscaleInterference


def test_interference_label_uses_physical_mixture_and_excludes_padding():
    clean = torch.randn(1, 2, 8, 5)
    noise = torch.randn_like(clean)
    mixture = compress_spectrum(clean + noise).permute(0, 2, 3, 1).unsqueeze(-2)
    prediction = compress_spectrum(noise)
    assert interference_loss(prediction, mixture, compress_spectrum(clean), [8]) < 1e-6
    prediction[:, :, 6:] += 20
    assert interference_loss(prediction, mixture, compress_spectrum(clean), [6]) < 1e-6


@pytest.mark.parametrize(('model_type', 'params', 'history_bytes'), [
    (CausalInterferenceMonaural, 2355, 197376),
    (CausalMultiscaleInterference, 4299, 789504),
])
def test_interference_branch_warm_identity_learns_and_streams(model_type, params, history_bytes):
    torch.set_num_threads(2)
    source = build_model_candidate(RECURRENT_ADAPT_CANDIDATE).eval()
    with torch.no_grad():
        source.base.base.beamformer.weight.normal_(std=.01)
        source.output.weight.normal_(std=.01)
    model = model_type(build_model_candidate(WIDE_CONDITIONED_CANDIDATE)).eval()
    model.load_warm(source.state_dict())
    example = torch.randn(1, 27, 257, 8, 2) * .1
    with torch.inference_mode():
        assert torch.equal(model(example), source(example))
    assert sum(p.numel() for p in model.interference.parameters()) == params
    model.train()
    estimate, noise = model(example, [27], return_noise=True)
    target = example[..., 0, :].permute(0, 3, 1, 2) * .8
    loss = (estimate - target).square().mean() + .1 * interference_loss(noise, example, target, [27])
    loss.backward()
    assert all(torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad) for p in model.interference.parameters())
    with torch.no_grad():
        for p in model.interference.parameters():
            p.add_(p.grad, alpha=-1e-3)
    model.eval()
    with torch.inference_mode():
        expected = model(example)
        changed = example.clone()
        changed[:, 15:] *= 10
        torch.testing.assert_close(model(changed)[:, :, :15], expected[:, :, :15], rtol=0, atol=1e-6)
        left, state = model.forward_stream(example[:, :15])
        history = state['noise_history'].clone()
        right, next_state = model.forward_stream(example[:, 15:], state)
        assert torch.equal(state['noise_history'], history)
        assert next_state['noise_history'].data_ptr() != state['noise_history'].data_ptr()
        assert history.numel() * history.element_size() == history_bytes
        torch.testing.assert_close(torch.cat((left, right), dim=2), expected, rtol=0, atol=1e-5)

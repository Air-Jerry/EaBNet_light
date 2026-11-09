"""Decoder-only root adaptation with unchanged inference and frozen encoder."""
import torch

from experiments.causal_decoder_monaural import CausalDecoderMonaural
from light_candidate_runtime import build_model_candidate, candidate_metadata


def models():
    torch.set_num_threads(2)
    torch.manual_seed(411)
    source = build_model_candidate('causal_monaural_spectral_v1')
    with torch.no_grad():
        source.base.base.beamformer.weight.normal_(std=.01)
        source.base.output.weight.normal_(std=.01)
        source.output.weight.normal_(std=.01)
    changed = CausalDecoderMonaural(build_model_candidate('causal_wide_conditioned_subband_v1'))
    changed.load_state_dict(source.state_dict(), strict=True)
    return source, changed


def test_inference_and_public_metadata_preserved_with_zero_future_gradient():
    metadata = candidate_metadata('causal_monaural_spectral_v1')
    source, changed = models()
    source.eval()
    changed.eval()
    value = (torch.randn(1, 13, 257, 8, 2)*.1).requires_grad_()
    actual = changed(value)
    assert torch.equal(actual, source(value))
    derivative, = torch.autograd.grad(actual[:, :, :5].square().sum(), value)
    assert torch.count_nonzero(derivative[:, 5:]) == 0
    with torch.no_grad():
        first, state = changed.forward_stream(value[:, :3])
        second, _ = changed.forward_stream(value[:, 3:], state)
        torch.testing.assert_close(torch.cat((first, second), 2), actual, rtol=1e-5, atol=1e-5)
    assert metadata == candidate_metadata('causal_monaural_spectral_v1')


def test_adam_adapts_decoder_without_updating_encoder_or_recurrence():
    _, model = models()
    model.train()
    previous = {name: value.clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        model(torch.randn(1, 7, 257, 8, 2)*.1).square().mean().backward()
        assert all(parameter.grad is None for parameter in model.base.base.stem.parameters())
        assert all(parameter.grad is None for parameter in model.base.base.layers.parameters())
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters() if parameter.requires_grad)
        optimizer.step()
    for name, parameter in model.named_parameters():
        if name.startswith('base.base.'):
            adapted = name.startswith(('base.base.upsamplers.', 'base.base.decoders.', 'base.base.beamformer.'))
            assert parameter.requires_grad == adapted
            if not adapted:
                assert torch.equal(parameter, previous[name])
    assert not torch.equal(model.base.base.beamformer.weight, previous['base.base.beamformer.weight'])
    assert {int(item['step']) for item in optimizer.state.values()} == {2}

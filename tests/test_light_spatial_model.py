"""Check the author block adapter, exact initial source, and checkpoint gradients."""

import copy

import pytest
import torch

from light_spatial_model import EaBNetSpatialQuality, SpatialResidual


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)
    torch.manual_seed(163)


def test_adapter_matches_author_network_and_shares_full_band_weights():
    refiner = SpatialResidual().eval()
    torch.nn.init.normal_(refiner.network.decoder.weight, std=0.01)
    value = torch.randn(1, 257, 9, 18)
    with torch.no_grad():
        actual = refiner(value)
        expected = refiner.network(value)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(refiner.network.layers) == 8
    assert all(layer.mhsa.num_heads == 4 for layer in refiner.network.layers)
    assert all(layer.full is refiner.network.layers[0].full for layer in refiner.network.layers)


def test_checkpointing_preserves_forward_and_parameter_gradients():
    plain = SpatialResidual().train()
    torch.nn.init.normal_(plain.network.decoder.weight, std=0.01)
    recompute = copy.deepcopy(plain)
    plain.gradient_checkpointing = False
    recompute.gradient_checkpointing = True
    value = torch.randn(1, 257, 9, 18)
    outputs = []
    for model in (plain, recompute):
        output = model(value)
        output.square().mean().backward()
        outputs.append(output.detach())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    for first, second in zip(plain.parameters(), recompute.parameters()):
        torch.testing.assert_close(first.grad, second.grad, rtol=1e-6, atol=1e-8)


def test_zero_head_is_exact_source_and_only_refiner_updates():
    model = EaBNetSpatialQuality().train()
    mixture = torch.randn(1, 9, 257, 8, 2) * 0.1
    original = {name: value.clone() for name, value in model.base.state_dict().items()}
    with torch.no_grad():
        baseline = model.base(mixture)
    estimate = model(mixture)
    torch.testing.assert_close(estimate, baseline, rtol=0, atol=0)
    assert not model.base.training and not model.base.base.training
    optimizer = torch.optim.Adam(model.refiner.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(mixture) - baseline * 0.9).square().mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.refiner.parameters())
        optimizer.step()
    assert all(p.grad is None for p in model.base.parameters())
    for name, value in model.base.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    assert model.refiner.network.decoder.weight.abs().sum() > 0


def test_padding_does_not_affect_valid_frames():
    model = EaBNetSpatialQuality().eval()
    torch.nn.init.normal_(model.refiner.network.decoder.weight, std=0.01)
    mixture = torch.randn(2, 12, 257, 8, 2) * 0.1
    changed = mixture.clone()
    changed[0, 8:] = 100
    with torch.no_grad():
        first = model(mixture, frame_lengths=[8, 12])
        second = model(changed, frame_lengths=[8, 12])
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.count_nonzero(first[0, :, 8:]) == 0


def test_compression_inverse_including_silence_and_epsilon_region():
    raw = torch.tensor([0.0, 1e-12, 1e-9, 1e-8, 1e-6, 0.1, 10.0])
    raw = torch.stack((raw, raw * -0.7), dim=-1)
    recovered = EaBNetSpatialQuality.decompress(EaBNetSpatialQuality.compress(raw))
    torch.testing.assert_close(recovered, raw, rtol=1e-6, atol=1e-15)

"""Independent equations and counts for reference-derived hypotheses.

These tests verify the candidate code, not equivalence to an unavailable
target-author implementation. Expected counts are reconstructed from layer
dimensions rather than copied from analysis outputs.
"""

import pytest
import torch
import torch.nn.functional as functional
from torch import nn

from EaBNet_light import EaBNet
from light_reference_variants import (
    CBAMChannelAttention,
    FRCRNMemory,
    all_reference_candidates,
    build_reference_candidate,
)


CANDIDATES = (
    "literal_per_frequency",
    "literal_flat_projection64",
    "cbam_per_frequency",
    "cbam_flat_projection64",
    "cbam_flat_hidden64",
)


@pytest.fixture(autouse=True)
def deterministic_reference_checks():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(901)
        yield
    torch.set_num_threads(previous_threads)


def parameter_oracle(candidate_id, *, without_skip_attention=False):
    channels, hidden, microphones = 64, 64, 8
    ca = (2 * channels * channels + channels if candidate_id.startswith("literal_")
          else channels * 4 + 4 + 4 * channels + channels)
    # Two spatial gates, real/imaginary mixing + BN, CA, output 1x1.
    fourier = 2 * (2 * 7 * 7) + 128 * 128 + 2 * 128 + ca + 64 * 64
    kernels = [10, 6, 6, 6, 6]
    encoder = sum((16 if index == 0 else 64) * 64 * area
                  + 64 + 128 + 64 + fourier
                  for index, area in enumerate(kernels))
    decoder = sum(128 * 64 * area + 64 + 128 + 64 + fourier
                  for area in reversed(kernels))
    dimension = 64 if "per_frequency" in candidate_id else 64 * 7
    if candidate_id.endswith("hidden64"):
        memory = dimension * hidden + hidden + hidden * dimension + dimension * 20
    else:
        memory = (dimension * hidden + hidden + hidden * 21
                  + hidden * dimension + dimension)
    skip = 0 if without_skip_attention else 5 * (ca + 2 * 7 * 7)
    lstm = 2 * (4 * hidden * channels + 4 * hidden * hidden + 8 * hidden)
    head = 2 * channels + lstm + hidden * (2 * microphones) + 2 * microphones
    return encoder + decoder + memory + skip + head


def test_reference_candidate_set_is_explicit_and_unique():
    assert tuple(all_reference_candidates()) == CANDIDATES
    assert len(set(all_reference_candidates())) == 5


@pytest.mark.parametrize("candidate_id", CANDIDATES)
@pytest.mark.parametrize("without_skip_attention", [False, True])
def test_independent_counts_shapes_and_shared_cell(candidate_id, without_skip_attention):
    model = build_reference_candidate(
        candidate_id, without_skip_attention=without_skip_attention
    ).eval()
    assert sum(parameter.numel() for parameter in model.parameters()) == parameter_oracle(
        candidate_id, without_skip_attention=without_skip_attention
    )
    calls = []
    handle = model.cred.dfsmn.register_forward_hook(lambda module, inputs, output: calls.append(id(module)))
    try:
        with torch.no_grad():
            output = model(torch.randn(1, 3, 257, 8, 2))
    finally:
        handle.remove()
    assert output.shape == (1, 2, 3, 257)
    assert torch.isfinite(output).all()
    assert calls == [id(model.cred.dfsmn)] * 3
    if without_skip_attention:
        assert all(isinstance(module, nn.Identity) for module in model.cred.skip_attention)
    else:
        assert len({id(module) for module in model.cred.skip_attention}) == 5


def test_literal_default_preserves_exact_initialization_and_forward():
    torch.manual_seed(412)
    expected = EaBNet().eval()
    torch.manual_seed(412)
    actual = build_reference_candidate("literal_per_frequency").eval()
    assert actual.state_dict().keys() == expected.state_dict().keys()
    for name, tensor in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], tensor, rtol=0, atol=0)
    sample = torch.randn(1, 3, 257, 8, 2)
    with torch.no_grad():
        torch.testing.assert_close(actual(sample), expected(sample), rtol=0, atol=0)


def test_cbam_matches_independent_two_pool_shared_mlp_equation():
    module = CBAMChannelAttention(channels=64, reduction_ratio=16).double()
    sample = torch.randn(2, 64, 3, 5, dtype=torch.float64)
    sample[:, :, 0, 0] += 4  # Ensure max and average contain different evidence.
    first, last = module.mlp[0], module.mlp[2]
    assert first.in_features == 64 and first.out_features == 4
    assert last.in_features == 4 and last.out_features == 64
    assert first.bias is not None and last.bias is not None

    def mlp(pooled):
        hidden = functional.linear(pooled, first.weight, first.bias).clamp_min(0)
        return functional.linear(hidden, last.weight, last.bias)

    average = sample.mean(dim=(2, 3))
    maximum = sample.amax(dim=(2, 3))
    expected = sample * (mlp(average) + mlp(maximum)).sigmoid()[:, :, None, None]
    torch.testing.assert_close(module(sample), expected, rtol=1e-12, atol=1e-12)
    assert sum(parameter.numel() for parameter in module.parameters()) == 580


def manual_frcrn(module, features):
    batch, channels, frames, frequencies = features.shape
    # Explicit channel-major c*F+f indexing, independently of implementation reshape.
    sequence = torch.stack([
        torch.stack([features[b, c, :, f] for c in range(channels) for f in range(frequencies)], dim=-1)
        for b in range(batch)
    ])
    hidden = functional.linear(sequence, module.linear.weight, module.linear.bias).clamp_min(0)
    projected = functional.linear(hidden, module.project.weight)
    result = sequence + projected
    taps = module.conv.weight[:, 0, :]
    for destination in range(frames):
        for kernel_index in range(taps.shape[-1]):
            source = destination - (taps.shape[-1] - 1 - kernel_index)
            if source >= 0:
                result[:, destination, :] += projected[:, source, :] * taps[:, kernel_index]
    return torch.stack([
        torch.stack([torch.stack([result[b, :, c * frequencies + f] for f in range(frequencies)], dim=-1)
                     for c in range(channels)])
        for b in range(batch)
    ])


@pytest.mark.parametrize("batch,frames", [(1, 1), (1, 7), (2, 7), (2, 1)])
def test_frcrn_matches_manual_taps_and_preserves_singleton_dimensions(batch, frames):
    module = FRCRNMemory(channels=3, frequencies=2, hidden_units=4, memory_size=3).double()
    sample = torch.randn(batch, 3, frames, 2, dtype=torch.float64)
    assert module.linear.in_features == 6 and module.linear.out_features == 4
    assert module.project.in_features == 4 and module.project.out_features == 6
    assert module.project.bias is None and module.conv.bias is None
    assert module.conv.groups == 6
    expected = manual_frcrn(module, sample)
    actual, memory = module(sample, return_memory=True)
    assert memory is None
    assert actual.shape == sample.shape
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_frcrn_current_tap_and_input_residual_are_separate():
    module = FRCRNMemory(channels=1, frequencies=1, hidden_units=1, memory_size=3).double()
    with torch.no_grad():
        module.linear.weight.fill_(1)
        module.linear.bias.zero_()
        module.project.weight.fill_(1)
        module.conv.weight.copy_(torch.tensor([[[7.0, 5.0, 3.0]]], dtype=torch.float64))
    signal = torch.tensor([[[[1.0], [2.0], [4.0], [8.0]]]], dtype=torch.float64)
    # x + projected + 3*current + 5*previous + 7*two-frames-ago.
    expected = torch.tensor([[[[5.0], [15.0], [37.0], [74.0]]]], dtype=torch.float64)
    torch.testing.assert_close(module(signal), expected, rtol=0, atol=0)


def test_frcrn_memory_uses_only_current_and_past_frames():
    module = FRCRNMemory(channels=3, frequencies=2, hidden_units=4, memory_size=4).double()
    original = torch.randn(2, 3, 9, 2, dtype=torch.float64)
    changed = original.clone()
    changed[:, :, 5:, :] += 100 * torch.randn_like(changed[:, :, 5:, :])
    torch.testing.assert_close(module(original)[:, :, :5, :], module(changed)[:, :, :5, :], rtol=0, atol=0)


@pytest.mark.parametrize("candidate_id", CANDIDATES[1:])
def test_reference_variants_have_finite_gradients_for_every_parameter(candidate_id):
    model = build_reference_candidate(candidate_id).train()
    mixture = torch.randn(2, 4, 257, 8, 2)
    target = torch.randn(2, 2, 4, 257)
    loss = (model(mixture) - target).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("candidate_id", CANDIDATES)
@pytest.mark.parametrize("kwargs", [{"c": 8, "cd1": 8}, {"dfsmn_layers": 4}])
def test_named_candidate_cannot_silently_change_its_recorded_dimensions(candidate_id, kwargs):
    with pytest.raises(ValueError, match="reference candidates require"):
        build_reference_candidate(candidate_id, **kwargs)

"""Independent checks for explicitly hypothetical structure candidates.

These checks certify implementations and accounting, not equivalence to the
authors' unpublished configuration. The parameter oracle is constructed from
layer dimensions, including LSTM biases, rather than candidate audit output.
"""

import dataclasses

import pytest
import torch
from torch import nn

from EaBNet_light import EaBNet
from light_structure_variants import StructureConfig, all_configs, build_candidate


@pytest.fixture(autouse=True)
def deterministic_cpu_checks():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(711)
        yield
    torch.set_num_threads(previous_threads)


def independent_parameter_count(config, *, without_skip_attention=False):
    """Count the default eight-microphone, 64-channel architecture algebraically."""
    channels, hidden, microphones = 64, 64, 8
    kernel_areas = [10, 6, 6, 6, 6]

    def fourier_block():
        spatial_attention = 2 * (2 * 7 * 7)
        fft_projection = (2 * channels) ** 2
        fft_batch_norm = 2 * (2 * channels)
        channel_attention = 2 * channels * channels // config.ca_groups + channels
        output_projection = channels * channels
        return (spatial_attention + fft_projection + fft_batch_norm
                + channel_attention + output_projection)

    # Every initial conv has a bias, a BN scale/offset, and a per-channel PReLU.
    encoder = sum(
        (2 * microphones if index == 0 else channels) * channels * area
        + channels + 2 * channels + channels + fourier_block()
        for index, area in enumerate(kernel_areas)
    )
    decoder_input = channels * (2 if config.skip_fusion == "cat" else 1)
    decoder = sum(
        decoder_input * channels * area
        + channels + 2 * channels + channels + fourier_block()
        for area in reversed(kernel_areas)
    )
    memory_input = channels * (7 if config.dfsmn_layout == "flattened" else 1)
    memory = (memory_input * hidden + hidden
              + hidden * 21  # a_0 plus 20 past-frame taps; shared over three uses.
              + hidden * memory_input + memory_input)
    skip_count = 0 if without_skip_attention else (1 if config.share_skip_attention else 5)
    skip = skip_count * (2 * channels * channels // config.ca_groups + channels + 2 * 7 * 7)
    # Two 64-unit LSTMs: four gates, both input/recurrent weights and two biases.
    lstm = 2 * (4 * hidden * channels + 4 * hidden * hidden + 2 * 4 * hidden)
    beamforming_head = lstm + hidden * (2 * microphones) + 2 * microphones + 2 * channels
    return encoder + decoder + memory + skip + beamforming_head


def test_configuration_grid_is_complete_unique_and_frozen():
    configs = list(all_configs())
    assert len(configs) == 24
    assert len({config.candidate_id for config in configs}) == 24
    assert {
        (config.ca_groups, config.skip_fusion, config.share_skip_attention, config.dfsmn_layout)
        for config in configs
    } == {
        (groups, fusion, shared, layout)
        for groups in (1, 2, 4)
        for fusion in ("cat", "add")
        for shared in (False, True)
        for layout in ("per_frequency", "flattened")
    }
    with pytest.raises(dataclasses.FrozenInstanceError):
        configs[0].ca_groups = 2


def test_default_candidate_preserves_baseline_state_and_output_exactly():
    torch.manual_seed(882)
    baseline = EaBNet().eval()
    torch.manual_seed(882)
    candidate = build_candidate(StructureConfig()).eval()
    original_state, candidate_state = baseline.state_dict(), candidate.state_dict()
    assert original_state.keys() == candidate_state.keys()
    for name, value in original_state.items():
        torch.testing.assert_close(candidate_state[name], value, rtol=0, atol=0)
    mixture = torch.randn(2, 4, 257, 8, 2)
    with torch.no_grad():
        torch.testing.assert_close(candidate(mixture), baseline(mixture), rtol=0, atol=0)
    assert independent_parameter_count(StructureConfig()) == 800_674
    assert independent_parameter_count(StructureConfig(), without_skip_attention=True) == 758_904


@pytest.mark.parametrize("config", list(all_configs()), ids=lambda config: config.candidate_id)
@pytest.mark.parametrize("without_skip_attention", [False, True], ids=["full", "ablated"])
def test_all_candidates_match_independent_parameter_formula_and_preserve_interface(
    config, without_skip_attention
):
    model = build_candidate(config, without_skip_attention=without_skip_attention).eval()
    assert isinstance(model, EaBNet)
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    assert actual_parameters == independent_parameter_count(
        config, without_skip_attention=without_skip_attention
    )
    # Inspect every CA occurrence, including the repeated aliases when shared.
    attention_convs = [
        block.fourier_attention.ca.conv
        for block in [*model.cred.encoder, *model.cred.decoder]
    ]
    if without_skip_attention:
        assert all(isinstance(module, nn.Identity) for module in model.cred.skip_attention)
    else:
        attention_convs += [module.ca.conv for module in model.cred.skip_attention]
        skip_parameter_sets = [
            {id(parameter) for parameter in module.parameters()}
            for module in model.cred.skip_attention
        ]
        if config.share_skip_attention:
            assert len({id(module) for module in model.cred.skip_attention}) == 1
            assert all(parameters == skip_parameter_sets[0] for parameters in skip_parameter_sets)
        else:
            assert len({id(module) for module in model.cred.skip_attention}) == 5
            assert sum(map(len, skip_parameter_sets)) == len(set.union(*skip_parameter_sets))
    assert all(conv.groups == config.ca_groups for conv in attention_convs)
    assert all(conv.in_channels == 128 and conv.out_channels == 64 for conv in attention_convs)
    expected_decoder_input = 128 if config.skip_fusion == "cat" else 64
    for decoder in model.cred.decoder:
        transposed = [module for module in decoder.in_deconv.modules() if isinstance(module, nn.ConvTranspose2d)]
        assert len(transposed) == 1
        assert transposed[0].in_channels == expected_decoder_input
    with torch.no_grad():
        output = model(torch.randn(1, 3, 257, 8, 2))
    assert output.shape == (1, 2, 3, 257)
    assert torch.isfinite(output).all()


def independent_flattened_dfsmn(memory_module, features, previous_memory=None):
    """Explicit indexing oracle: flattened dimension is channel * F + frequency."""
    batch, channels, frames, frequencies = features.shape
    sequence = torch.stack([
        torch.stack([features[b, c, :, f] for c in range(channels) for f in range(frequencies)])
        for b in range(batch)
    ])
    projected = torch.einsum("hc,bct->bht", memory_module.in_conv.weight[..., 0], sequence)
    projected = projected + memory_module.in_conv.bias[None, :, None]
    memory = projected.clone()
    for destination in range(frames):
        for tap in range(memory_module.left_memory.shape[1]):
            source = destination - tap
            if source >= 0:
                memory[:, :, destination] += projected[:, :, source] * memory_module.left_memory[:, tap]
        if memory_module.right_memory is not None:
            for tap in range(memory_module.right_memory.shape[1]):
                source = destination + tap + 1
                if source < frames:
                    memory[:, :, destination] += projected[:, :, source] * memory_module.right_memory[:, tap]
    if previous_memory is not None:
        memory = memory + previous_memory
    output = torch.einsum("ch,bht->bct", memory_module.out_conv.weight[..., 0], memory)
    output = (output + memory_module.out_conv.bias[None, :, None]).clamp_min(0)
    restored = torch.empty_like(features)
    for b in range(batch):
        for c in range(channels):
            for f in range(frequencies):
                restored[b, c, :, f] = output[b, c * frequencies + f, :]
    return restored, memory


def test_flattened_memory_preserves_batch_time_channel_and_frequency_indices():
    model = build_candidate(StructureConfig(dfsmn_layout="flattened")).double()
    memory_module = model.cred.dfsmn
    assert (memory_module.in_conv.in_channels, memory_module.in_conv.out_channels) == (448, 64)
    assert (memory_module.out_conv.in_channels, memory_module.out_conv.out_channels) == (64, 448)
    # Deliberately distinguish batch, time and frequency, including distinct batches.
    features = torch.randn(2, 64, 5, 7, dtype=torch.float64)
    features[1] += 0.75
    features[:, :, :, 3] += 0.25
    actual_memory = expected_memory = None
    expected = features.clone()
    with torch.no_grad():
        for _ in range(3):
            features, actual_memory = memory_module(features, actual_memory, return_memory=True)
            expected, expected_memory = independent_flattened_dfsmn(memory_module, expected, expected_memory)
            assert actual_memory.shape == (2, 64, 5)
            torch.testing.assert_close(actual_memory, expected_memory, rtol=1e-11, atol=1e-11)
            torch.testing.assert_close(features, expected, rtol=1e-11, atol=1e-11)
    default = StructureConfig()
    flattened = StructureConfig(dfsmn_layout="flattened")
    assert independent_parameter_count(flattened) - independent_parameter_count(default) == 49_536


def test_shared_skip_is_applied_at_all_five_resolutions_and_memory_reused_three_times():
    model = build_candidate(StructureConfig(
        ca_groups=2, skip_fusion="add", share_skip_attention=True, dfsmn_layout="flattened"
    )).eval()
    skip_shapes, memory_parameters = [], []
    handles = [
        model.cred.skip_attention[0].register_forward_hook(
            lambda _module, args, _output: skip_shapes.append(tuple(args[0].shape))
        ),
        model.cred.dfsmn.register_forward_hook(
            lambda module, _args, _output: memory_parameters.append(
                tuple(id(parameter) for parameter in module.parameters())
            )
        ),
    ]
    try:
        with torch.no_grad():
            model(torch.randn(1, 3, 257, 8, 2))
    finally:
        for handle in handles:
            handle.remove()
    assert skip_shapes == [(1, 64, 3, bins) for bins in (7, 15, 31, 63, 127)]
    assert len(memory_parameters) == 3
    assert memory_parameters[0] == memory_parameters[1] == memory_parameters[2]


@pytest.mark.parametrize("config", [
    StructureConfig(),
    StructureConfig(ca_groups=2, skip_fusion="add", share_skip_attention=True, dfsmn_layout="flattened"),
    StructureConfig(ca_groups=4, skip_fusion="cat", share_skip_attention=False, dfsmn_layout="flattened"),
    StructureConfig(ca_groups=4, skip_fusion="add", share_skip_attention=True, dfsmn_layout="per_frequency"),
], ids=lambda config: config.candidate_id)
def test_representative_combinations_backpropagate_finite_gradients_to_all_parameters(config):
    model = build_candidate(config).train()
    mixture = torch.randn(2, 4, 257, 8, 2, requires_grad=True)
    target = torch.randn(2, 2, 4, 257)
    loss = (model(mixture) - target).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert mixture.grad is not None and torch.isfinite(mixture.grad).all()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name

"""Equation-level checks for the disclosed FCAE architecture.

The numerical oracles below come from the paper equations and explicit
complex arithmetic, not from an older model implementation.  DFSMN's memory
order and skip fusion are not specified by the paper; their tests establish
the documented implementation contract, not author-verified equivalence.
"""

import numpy as np
import pytest
import torch
from torch import nn

from EaBNet_light import (
    ChannelAttention,
    EaBNet,
    FourierAttentionBlock,
    LSTM_BF,
    SharedDFSMN,
    SpatialAttention,
    com_mag_mse_loss,
)


@pytest.fixture(autouse=True)
def reproducible_cpu_checks():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(417)
        yield
    torch.set_num_threads(previous_threads)


class FixedWeights(nn.Module):
    def __init__(self, weights, cnn=False):
        super().__init__()
        self.register_buffer("weights", weights)
        self.cnn = cnn

    def forward(self, embedding):
        batch, _, frames, frequencies = embedding.shape
        weights = self.weights.expand(batch, frames, frequencies, -1, -1)
        if self.cnn:
            return weights.permute(0, 3, 4, 1, 2).reshape(
                batch, -1, frames, frequencies
            )
        return weights


@pytest.mark.parametrize("bf_type", ["lstm", "cnn"])
def test_equation_2_conjugates_weights_and_sums_all_microphones(bf_type):
    model = EaBNet(M=3, bf_type=bf_type).double()
    # Bypass estimation, leaving the public filter-and-sum path under test.
    model.cred = nn.Identity()
    weights = torch.tensor([[2.0, -3.0], [-1.0, 4.0], [0.5, 2.0]], dtype=torch.float64)
    model.bf_map = FixedWeights(weights, cnn=bf_type == "cnn")
    mixture = torch.randn(2, 4, 257, 3, 2, dtype=torch.float64)
    expected_complex = (
        torch.view_as_complex(weights).conj()
        * torch.view_as_complex(mixture)
    ).sum(dim=-1)
    expected = torch.view_as_real(expected_complex).permute(0, 3, 1, 2)
    actual = model(mixture)
    assert actual.shape == (2, 2, 4, 257)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    # These inputs must distinguish conjugated and unconjugated weights.
    wrong = (torch.view_as_complex(weights) * torch.view_as_complex(mixture)).sum(-1)
    assert not torch.allclose(expected_complex, wrong)


def test_encoder_input_order_is_all_real_microphones_then_all_imaginary():
    model = EaBNet(M=3)
    model.cred = nn.Identity()
    model.bf_map = FixedWeights(torch.ones(3, 2))
    mixture = torch.arange(2 * 3 * 257 * 3 * 2, dtype=torch.float32).reshape(
        2, 3, 257, 3, 2
    )
    seen = []
    handle = model.cred.register_forward_pre_hook(lambda _module, args: seen.append(args[0]))
    try:
        model(mixture)
    finally:
        handle.remove()
    expected = torch.stack(
        [mixture[..., mic, component] for component in range(2) for mic in range(3)],
        dim=1,
    )
    torch.testing.assert_close(seen[0], expected, rtol=0, atol=0)


def reference_single_channel_conv(image, kernel):
    """Small zero-padded cross-correlation, without calling a convolution."""
    height, width = image.shape
    radius = kernel.shape[0] // 2
    output = torch.zeros_like(image)
    for row in range(height):
        for col in range(width):
            for kr in range(kernel.shape[0]):
                for kc in range(kernel.shape[1]):
                    source_row, source_col = row + kr - radius, col + kc - radius
                    if 0 <= source_row < height and 0 <= source_col < width:
                        output[row, col] += image[source_row, source_col] * kernel[kr, kc]
    return output


def test_equation_5_spatial_pool_relu_separate_convolutions_and_sigmoid():
    attention = SpatialAttention(kernel_size=3).double()
    avg_kernel = torch.tensor([[0.1, -0.2, 0.3], [0.4, 0.2, -0.1], [-0.3, 0.5, 0.1]], dtype=torch.float64)
    max_kernel = torch.tensor([[-0.2, 0.1, 0.2], [0.3, -0.4, 0.1], [0.2, -0.1, 0.4]], dtype=torch.float64)
    with torch.no_grad():
        attention.avg_conv.weight.copy_(avg_kernel[None, None])
        attention.max_conv.weight.copy_(max_kernel[None, None])
    features = torch.tensor(
        [[[[1.0, -2.0, 3.0], [-4.0, -1.0, 0.5]],
          [[-3.0, -4.0, 1.0], [2.0, -5.0, -0.5]],
          [[0.5, -1.0, -2.0], [1.0, -2.0, 2.0]]]], dtype=torch.float64,
    )
    avg = features[0].sum(0) / 3
    maximum = torch.stack([features[0, c] for c in range(3)]).max(0).values
    logits = reference_single_channel_conv(avg.clamp_min(0), avg_kernel)
    logits += reference_single_channel_conv(maximum.clamp_min(0), max_kernel)
    expected = features / (1 + torch.exp(-logits[None, None]))
    torch.testing.assert_close(attention(features), expected, rtol=1e-12, atol=1e-12)


def test_equation_11_pools_time_frequency_and_concatenates_avg_then_max():
    attention = ChannelAttention(channels=2).double()
    weight = torch.tensor([[0.2, -0.3, 0.4, 0.1], [-0.1, 0.5, 0.2, -0.4]], dtype=torch.float64)
    bias = torch.tensor([0.15, -0.25], dtype=torch.float64)
    with torch.no_grad():
        attention.conv.weight.copy_(weight[:, :, None, None])
        attention.conv.bias.copy_(bias)
    features = torch.tensor(
        [[[[1., -3., 2.], [4., 0., -2.]], [[-1., 2., 3.], [5., -4., 1.]]],
         [[[3., -2., 1.], [0., -1., 2.]], [[-4., 1., 0.], [2., 3., -2.]]]],
        dtype=torch.float64,
    )
    expected = torch.empty_like(features)
    for batch in range(2):
        pooled = torch.tensor(
            [features[batch, c].sum().item() / 6 for c in range(2)]
            + [features[batch, c].max().item() for c in range(2)],
            dtype=torch.float64,
        )
        for channel in range(2):
            logit = (weight[channel] * pooled).sum() + bias[channel]
            expected[batch, channel] = features[batch, channel] / (1 + torch.exp(-logit))
    torch.testing.assert_close(attention(features), expected, rtol=1e-12, atol=1e-12)


class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


@pytest.mark.parametrize("frequency_length", [7, 15, 31, 63, 127, 257])
def test_equations_4_to_10_fft_axis_ri_layout_and_odd_inverse_length(frequency_length):
    block = FourierAttentionBlock(2, "BN").double().eval()
    # Isolate the FFT path with observable real/imaginary channel mixing.
    block.sa_q = Scale(0.75)
    block.sa_p = Scale(0.4)
    block.ca = Scale(0.3)
    block.out_conv = nn.Identity()
    mixing = nn.Conv2d(4, 4, 1, bias=False).double()
    weight = torch.tensor(
        [[1., 0.2, -0.3, 0.4], [0.1, 0.8, 0.5, -0.2],
         [0.4, -0.6, 0.7, 0.2], [-0.3, 0.1, 0.2, 0.9]], dtype=torch.float64,
    )
    with torch.no_grad():
        mixing.weight.copy_(weight[:, :, None, None])
    block.fft_conv = mixing
    features = torch.randn(2, 2, 3, frequency_length, dtype=torch.float64)
    seen = []
    handle = mixing.register_forward_pre_hook(lambda _module, args: seen.append(args[0]))
    try:
        actual = block(features)
    finally:
        handle.remove()
    # NumPy supplies an independent FFT implementation and double precision oracle.
    spectrum = np.fft.rfft(0.75 * features.numpy(), axis=-1)
    ri = np.concatenate([spectrum.real, spectrum.imag], axis=1)
    projected = np.einsum("oc,bctf->botf", weight.numpy(), ri) * 0.4
    expected_complex = projected[:, :2] + 1j * projected[:, 2:]
    expected = np.fft.irfft(expected_complex, n=frequency_length, axis=-1)
    expected += 0.3 * features.numpy()
    assert actual.shape == features.shape
    assert seen[0].shape == (2, 4, 3, frequency_length // 2 + 1)
    torch.testing.assert_close(seen[0], torch.from_numpy(ri), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual, torch.from_numpy(expected), rtol=1e-11, atol=1e-11)


def test_section_4_2_encoder_decoder_geometry_and_three_shared_dfsmn_calls():
    model = EaBNet().eval()
    encoder_shapes, decoder_shapes, memory_calls, memory_states = [], [], [], []
    handles = []
    for layer in model.cred.encoder:
        handles.append(layer.register_forward_hook(
            lambda _module, _args, output: encoder_shapes.append(tuple(output.shape))))
    for layer in model.cred.decoder:
        handles.append(layer.register_forward_hook(
            lambda _module, _args, output: decoder_shapes.append(tuple(output.shape))))
    handles.append(model.cred.dfsmn.register_forward_hook(
        lambda module, _args, _output: memory_calls.append(
            tuple(id(parameter) for parameter in module.parameters()))))
    handles.append(model.cred.dfsmn.register_forward_hook(
        lambda _module, args, kwargs, output: memory_states.append(
            (args[1] if len(args) > 1 else kwargs.get("previous_memory"), output[1])),
        with_kwargs=True,
    ))
    try:
        with torch.no_grad():
            output = model(torch.randn(1, 5, 257, 8, 2))
    finally:
        for handle in handles:
            handle.remove()
    assert output.shape == (1, 2, 5, 257)
    assert encoder_shapes == [(1, 64, 5, f) for f in [127, 63, 31, 15, 7]]
    assert decoder_shapes == [(1, 64, 5, f) for f in [15, 31, 63, 127, 257]]
    assert len(memory_calls) == 3
    assert memory_calls[0] == memory_calls[1] == memory_calls[2]
    assert memory_states[0][0] is None
    assert memory_states[1][0] is memory_states[0][1]
    assert memory_states[2][0] is memory_states[1][1]
    assert len([module for module in model.cred.modules() if isinstance(module, SharedDFSMN)]) == 1
    assert isinstance(model.cred.out_conv, nn.Identity)


def test_figure_1_two_64_unit_lstms_then_one_linear():
    mapper = LSTM_BF(embed_dim=64, M=8)
    recurrent = [module for module in mapper.modules() if isinstance(module, nn.LSTM)]
    linear = [module for module in mapper.modules() if isinstance(module, nn.Linear)]
    assert len(recurrent) == 2
    assert all(module.input_size == module.hidden_size == 64 for module in recurrent)
    assert all(not module.bidirectional and module.num_layers == 1 for module in recurrent)
    assert len(linear) == 1
    assert (linear[0].in_features, linear[0].out_features) == (64, 16)
    assert mapper(torch.randn(2, 64, 3, 7)).shape == (2, 3, 7, 8, 2)


@pytest.mark.parametrize("right_order", [0, 2])
def test_documented_dfsmn_filter_delay_sign_and_zero_boundaries(right_order):
    memory = SharedDFSMN(
        channels=2, hidden_units=2, memory_size=3,
        is_causal=right_order == 0, right_memory_size=right_order,
    ).double()
    # Reference [16], Eq. (2)/(3): learned a_0 supplements the identity term.
    past = torch.tensor([[0.25, 2., -1., 0.5], [-0.75, -0.5, 3., 1.]], dtype=torch.float64)
    future = torch.tensor([[1.5, -2.], [0.25, 0.5]], dtype=torch.float64)
    with torch.no_grad():
        memory.left_memory.copy_(past)
        if right_order:
            memory.right_memory.copy_(future)
    features = torch.tensor([[[1., 2., 4., 8., 16.], [-2., 1., 3., -1., 5.]]], dtype=torch.float64)
    expected = features.clone()
    for channel in range(2):
        for time in range(5):
            for delay in range(4):
                if time - delay >= 0:
                    expected[0, channel, time] += past[channel, delay] * features[0, channel, time - delay]
            for advance in range(1, right_order + 1):
                if time + advance < 5:
                    expected[0, channel, time] += future[channel, advance - 1] * features[0, channel, time + advance]
    torch.testing.assert_close(memory._memory_filter(features), expected, rtol=0, atol=0)


def test_reference_16_dfsmn_projection_memory_skip_and_output_nonlinearity():
    # https://arxiv.org/pdf/1802.09194, Eqs. (1), (4), (5).
    # Order/projection choices below are test fixtures, not claimed paper values.
    memory = SharedDFSMN(channels=2, hidden_units=2, memory_size=1).double()
    projection = np.array([[0.5, -0.25], [0.75, 0.4]])
    projection_bias = np.array([0.2, -0.3])
    output_weight = np.array([[0.6, -0.5], [0.3, 0.8]])
    output_bias = np.array([-0.1, 0.25])
    coefficients = np.array([[0.2, 0.3], [-0.1, 0.4]])
    with torch.no_grad():
        memory.in_conv.weight.copy_(torch.from_numpy(projection[:, :, None]))
        memory.in_conv.bias.copy_(torch.from_numpy(projection_bias))
        memory.out_conv.weight.copy_(torch.from_numpy(output_weight[:, :, None]))
        memory.out_conv.bias.copy_(torch.from_numpy(output_bias))
        memory.left_memory.copy_(torch.from_numpy(coefficients))
    features = torch.tensor(
        [[[[1., -2.], [0.5, 3.], [-1., 2.]],
          [[-0.5, 1.], [2., -3.], [1.5, 0.5]]]], dtype=torch.float64,
    )
    expected = features.numpy().copy()
    expected_memory = np.zeros_like(expected)
    actual = features
    actual_memory = None
    for _ in range(3):
        projected = np.einsum("oc,bctf->botf", projection, expected)
        projected += projection_bias[None, :, None, None]
        new_memory = expected_memory + projected * (1 + coefficients[:, 0])[None, :, None, None]
        new_memory[:, :, 1:, :] += projected[:, :, :-1, :] * coefficients[:, 1][None, :, None, None]
        expected = np.einsum("oc,bctf->botf", output_weight, new_memory)
        expected = np.maximum(expected + output_bias[None, :, None, None], 0)
        expected_memory = new_memory
        actual, actual_memory = memory(actual, actual_memory, return_memory=True)
        torch.testing.assert_close(actual, torch.from_numpy(expected), rtol=1e-12, atol=1e-12)


def test_existing_complex_magnitude_loss_value_and_padding_gradient():
    # Four valid TF cells, with target 3+4j and estimate zero:
    # magnitude MSE=25, mean over real/imaginary components=12.5.
    estimate = torch.zeros(2, 2, 3, 1, dtype=torch.float64, requires_grad=True)
    label = torch.tensor([3., 4.], dtype=torch.float64)[None, :, None, None].expand_as(estimate).clone()
    label[0, :, 1:] = 1000  # Padded values must not contribute to the loss.
    loss = com_mag_mse_loss(estimate, label, [1, 3])
    assert loss.item() == pytest.approx(18.75)
    loss.backward()
    expected_grad = torch.tensor([-3. / 8, -4. / 8], dtype=torch.float64)[None, :, None, None].expand_as(estimate).clone()
    expected_grad[0, :, 1:] = 0
    torch.testing.assert_close(estimate.grad, expected_grad, rtol=0, atol=0)
    assert torch.isfinite(estimate.grad).all()


def test_full_default_model_finite_gradients_and_fixed_batch_learning():
    model = EaBNet().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mixture = torch.randn(1, 5, 257, 8, 2)
    target = mixture[..., 0, :].permute(0, 3, 1, 2).contiguous() * 0.7
    losses = []
    for step in range(10):
        optimizer.zero_grad(set_to_none=True)
        output = model(mixture)
        loss = com_mag_mse_loss(output, target, [5])
        assert torch.isfinite(loss)
        losses.append(loss.item())
        loss.backward()
        if step == 0:
            for name, parameter in model.named_parameters():
                assert parameter.grad is not None, name
                assert torch.isfinite(parameter.grad).all(), name
            # Each major component must have a nonzero learning signal.
            for component in [*model.cred.encoder, model.cred.dfsmn,
                              *model.cred.skip_attention, *model.cred.decoder,
                              model.bf_map]:
                assert sum(p.grad.abs().sum().item() for p in component.parameters()) > 0
        optimizer.step()
    assert losses[-1] < losses[0] * 0.8, losses


@pytest.mark.parametrize("shape", [(1, 3, 256, 8, 2), (1, 3, 257, 7, 2), (1, 3, 257, 8, 3)])
def test_invalid_stft_geometry_is_rejected_without_silent_adjustment(shape):
    with pytest.raises(ValueError):
        EaBNet()(torch.zeros(shape))


@pytest.mark.parametrize("device_type,dtype", [
    ("cpu", torch.bfloat16),
    pytest.param("cuda", torch.float16, marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA unavailable: GPU AMP has not been verified")),
])
def test_autocast_odd_frequency_fft_and_backward_are_finite(device_type, dtype):
    model = EaBNet().to(device_type).train()
    mixture = torch.randn(1, 5, 257, 8, 2, device=device_type)
    target = torch.randn(1, 2, 5, 257, device=device_type)
    with torch.autocast(device_type, dtype=dtype):
        estimate = model(mixture)
    loss = com_mag_mse_loss(estimate.float(), target, [5])
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

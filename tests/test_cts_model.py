import torch
import pytest

from CTSNet import CTSNet, SqueezedTCM, cts_loss


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.set_num_threads(2)
    torch.manual_seed(417)


def test_table_i_shapes_and_independent_parameters():
    model = CTSNet(return_auxiliary=True)
    shapes = []
    handles = [layer.register_forward_hook(lambda _m, _a, y: shapes.append(tuple(y.shape)))
               for layer in model.me_net.encoder.layers]
    output = model(torch.randn(2, 13, 161, 1, 2))
    for h in handles:
        h.remove()
    assert shapes == [(2, 64, 13, f) for f in (79, 39, 19, 9, 4)]
    assert output.shape == (2, 3, 13, 161)
    assert torch.all(output[:, 2] > 0)
    blocks = [m for m in model.modules() if isinstance(m, SqueezedTCM)]
    assert len(blocks) == 36
    assert len({id(m.input_conv.weight) for m in blocks}) == 36
    assert [m.main_branch.conv.dilation[0] for m in blocks] == [1, 2, 4, 8, 16, 32] * 6
    assert all(m.main_branch.conv.groups == 1 and m.gate_branch.conv.groups == 1 for m in blocks)
    assert len(model.me_net.decoders) == 1 and len(model.cs_net.decoders) == 2
    assert all(d.linear.in_features == d.linear.out_features == 161
               for net in (model.me_net, model.cs_net) for d in net.decoders)


def test_joint_gradient_reaches_both_stages_and_is_finite():
    model = CTSNet(return_auxiliary=True)
    x = torch.randn(1, 9, 161, 1, 2)
    target = torch.randn(1, 2, 9, 161)
    loss = cts_loss(model(x), target, [9], lambda_me=0.0)
    loss.backward()
    for net in (model.me_net, model.cs_net):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
        assert sum(float(p.grad.abs().sum()) for p in net.parameters()) > 0


def test_zero_noisy_phase_has_a_finite_inference_value():
    model = CTSNet().eval()
    with torch.no_grad():
        magnitude, coarse, final = model.forward_stages(torch.zeros(1, 9, 161, 1, 2))
    assert torch.isfinite(final).all()
    assert torch.equal(coarse[:, :1], magnitude)
    assert torch.count_nonzero(coarse[:, 1:]) == 0


def test_pretraining_does_not_train_or_execute_cs():
    model = CTSNet(return_auxiliary=True)
    model.set_stage("me")
    def forbidden(*_):
        raise AssertionError("CS must not execute during ME pretraining")
    handle = model.cs_net.register_forward_pre_hook(forbidden)
    x = torch.randn(1, 8, 161, 1, 2)
    output = model(x)
    cts_loss(output, torch.randn(1, 2, 8, 161), [8], stage="me").backward()
    handle.remove()
    assert all(p.grad is None and not p.requires_grad for p in model.cs_net.parameters())
    assert torch.allclose(torch.linalg.vector_norm(output[:, :2], dim=1), output[:, 2], atol=1e-6)
    model.set_stage("joint")
    assert all(p.requires_grad for p in model.cs_net.parameters())


def test_equations_7_to_11_and_global_residual():
    model = CTSNet()
    x = torch.randn(1, 8, 161, 1, 2)
    with torch.no_grad():
        for parameter in model.cs_net.parameters():
            parameter.zero_()
        mag, coarse, final = model.forward_stages(x)
    noisy_ri = x[..., 0, :].permute(0, 3, 1, 2)
    expected = mag * noisy_ri / torch.linalg.vector_norm(noisy_ri, dim=1, keepdim=True)
    assert torch.allclose(coarse, expected, atol=1e-6)
    assert torch.equal(final, coarse)


def test_equations_17_to_21_ri_sum_and_padding_gradient():
    # Reference is 3+4j (magnitude 5); refined estimate is 0+0j,
    # ME magnitude is 2. One cell: Lme=9, LRI=25, Lmag=25, L=25.9.
    prediction = torch.tensor([[[[0.0], [float("nan")]],
                                [[0.0], [float("nan")]],
                                [[2.0], [float("nan")]]]], requires_grad=True)
    target = torch.tensor([[[[3.0], [float("nan")]], [[4.0], [float("nan")]]]])
    assert cts_loss(prediction, target, [1], stage="me").item() == 9
    loss = cts_loss(prediction, target, [1])
    assert loss.item() == pytest.approx(25.9)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert torch.equal(prediction.grad[:, :, 1:], torch.zeros_like(prediction.grad[:, :, 1:]))


def test_variable_length_batch_matches_independent_inference():
    model = CTSNet(return_auxiliary=True).eval()
    x = torch.randn(2, 15, 161, 1, 2)
    with torch.no_grad():
        batched = model(x, frame_lengths=torch.tensor([8, 15]))
        short = model(x[:1, :8])
        long = model(x[1:])
    assert torch.equal(batched[:1, :, :8], short)
    assert torch.equal(batched[1:], long)
    assert torch.count_nonzero(batched[0, :, 8:]) == 0


def test_cumulative_variant_is_causal_at_spectral_frame_level():
    model = CTSNet(norm_type="cIN").eval()
    x = torch.randn(1, 21, 161, 1, 2)
    future_changed = x.clone()
    future_changed[:, 10:] = 20 * torch.randn_like(x[:, 10:])
    with torch.no_grad():
        before, after = model(x), model(future_changed)
    assert torch.allclose(before[:, :, :10], after[:, :, :10], atol=1e-5, rtol=1e-5)


def test_standard_in_cannot_support_a_strict_causal_claim():
    model = CTSNet(norm_type="IN", is_causal=True).eval()
    x = torch.randn(1, 21, 161, 1, 2)
    changed = x.clone()
    changed[:, 10:] *= 20
    with torch.no_grad():
        difference = (model(x)[:, :, :10] - model(changed)[:, :, :10]).abs().max()
    assert difference > 1e-3


def test_small_fixed_batch_can_be_optimized():
    model = CTSNet(return_auxiliary=True)
    x = torch.randn(1, 8, 161, 1, 2)
    target = x[..., 0, :].permute(0, 3, 1, 2) * 0.6
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        loss = cts_loss(model(x), target, [8])
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    assert losses[-1] < 0.65 * losses[0]


@pytest.mark.parametrize("shape", [(1, 8, 257, 1, 2), (1, 8, 161, 8, 2)])
def test_rejects_wrong_fft_or_multichannel_input(shape):
    with pytest.raises(ValueError, match="320-point FFT"):
        CTSNet()(torch.randn(*shape))

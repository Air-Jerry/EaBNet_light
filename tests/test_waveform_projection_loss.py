import torch

from experiments.waveform_projection_loss import waveform_projection_loss


def test_projection_loss_is_scale_invariant_and_penalizes_residual():
    target = torch.tensor([[1., 1., -1., -1.]])
    interference = torch.tensor([[1., -1., 1., -1.]])
    assert waveform_projection_loss(-3 * target + 2, target) == 0
    torch.testing.assert_close(waveform_projection_loss(target + interference, target), torch.tensor(.5))
    torch.testing.assert_close(waveform_projection_loss(2 * (target + interference), target), torch.tensor(.5))
    scale = torch.tensor(.8, requires_grad=True)
    loss = waveform_projection_loss(target + scale * interference, target)
    loss.backward()
    assert scale.grad > 0


def test_projection_loss_finite_for_silent_estimate_and_target():
    estimate = torch.zeros(2, 8, requires_grad=True)
    target = torch.stack((torch.arange(8.), torch.zeros(8)))
    loss = waveform_projection_loss(estimate, target)
    loss.backward()
    assert loss == 1
    assert torch.isfinite(estimate.grad).all()

import torch

from experiments.bark_envelope_loss import bark_envelope_loss


def test_bark_envelope_ignores_padding_and_penalizes_band_magnitude_error():
    target = torch.tensor([[[1., .25], [2., .1], [3., .5]]])
    frames = torch.tensor([[True, True, False]])
    matrix = torch.eye(2)
    prediction = target.clone()
    prediction[:, 2] *= 50
    assert bark_envelope_loss(prediction, target, frames, matrix) == 0
    prediction[:, :2, 0] *= 2
    prediction.requires_grad_()
    loss = bark_envelope_loss(prediction, target, frames, matrix)
    assert loss > 0
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert (prediction.grad[:, :2, 0] > 0).all()
    assert torch.count_nonzero(prediction.grad[:, 2]) == 0

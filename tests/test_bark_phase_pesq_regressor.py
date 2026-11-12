import torch

from experiments.bark_phase_pesq_regressor import BarkPhasePESQRegressor, bark_phase_score_features


def test_bark_phase_features_distinguish_equal_magnitude_signals():
    torch.set_num_threads(2)
    angle = torch.linspace(-3., 3., 2 * 12 * 6).reshape(2, 12, 6)
    ref = torch.stack((angle.cos(), angle.sin()), dim=1)
    est = ref[:, [1, 0]].clone()
    est[:, 0].neg_()
    est.requires_grad_()
    matrix = torch.eye(6)
    same = bark_phase_score_features(ref, ref, matrix)
    shifted = bark_phase_score_features(est, ref, matrix)
    torch.testing.assert_close(same[:, :3], shifted[:, :3])
    assert same[:, 3].mean() > .9
    assert shifted[:, 3].abs().max() < 1e-6
    assert shifted[:, 4].mean() > .9
    result = BarkPhasePESQRegressor()(shifted)
    result.mean().backward()
    assert result.shape == (2,)
    assert torch.isfinite(est.grad).all()
    assert torch.count_nonzero(est.grad)

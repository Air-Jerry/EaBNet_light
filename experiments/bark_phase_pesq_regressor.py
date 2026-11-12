"""Training-only PESQ regressor with Bark-band complex agreement features."""
import torch
from torch import nn

from experiments.bark_pesq_regressor import BarkPESQRegressor, bark_score_features


def bark_phase_score_features(estimate, reference, matrix):
    magnitude = bark_score_features(estimate, reference, matrix)
    estimate = estimate.float()
    reference = reference.float()
    estimate = estimate * torch.linalg.vector_norm(estimate, dim=1, keepdim=True)
    reference = reference * torch.linalg.vector_norm(reference, dim=1, keepdim=True)
    er, ei = estimate.unbind(dim=1)
    rr, ri = reference.unbind(dim=1)
    estimated_power = (er.square() + ei.square()) @ matrix
    reference_power = (rr.square() + ri.square()) @ matrix
    floor = (.01 * reference_power.mean((1, 2), keepdim=True)).clamp_min(1e-9)
    denominator = ((estimated_power + floor) * (reference_power + floor)).sqrt()
    real = ((er * rr + ei * ri) @ matrix) / denominator
    imaginary = ((ei * rr - er * ri) @ matrix) / denominator
    return torch.cat((magnitude, real.unsqueeze(1), imaginary.unsqueeze(1)), dim=1)


class BarkPhasePESQRegressor(BarkPESQRegressor):
    def __init__(self):
        super().__init__()
        self.features[0] = nn.Conv2d(5, 16, (5, 5), stride=(2, 1), padding=(2, 2))

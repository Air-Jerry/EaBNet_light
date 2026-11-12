"""Training-only score regressor for reference and enhanced Bark spectra."""
import torch
from torch import nn

from experiments.causal_pmsqe_objective import physical_power


def bark_score_features(estimate, reference, matrix):
    """Return raw band levels relative to one common reference power."""
    target_power = physical_power(reference) @ matrix
    estimate_power = physical_power(estimate) @ matrix
    floor = (.05 * target_power.mean((1, 2), keepdim=True)).clamp_min(1e-10)
    target = .5 * (target_power / floor).log1p()
    enhanced = .5 * (estimate_power / floor).log1p()
    return torch.stack((target, enhanced, (target - enhanced).abs()), dim=1)


class BarkPESQRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, (5, 5), stride=(2, 1), padding=(2, 2)),
            nn.SiLU(),
            nn.Conv2d(16, 24, (5, 5), stride=(3, 2), padding=(2, 2)),
            nn.SiLU(),
            nn.Conv2d(24, 32, (3, 3), stride=(2, 2), padding=(1, 1)),
            nn.SiLU(),
        )
        self.score = nn.Sequential(nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, value):
        hidden = self.features(value)
        pooled = torch.cat((hidden.mean((2, 3)), hidden.amax((2, 3))), dim=1)
        return self.score(pooled).squeeze(-1)

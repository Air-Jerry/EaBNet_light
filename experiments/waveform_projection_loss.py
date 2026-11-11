"""Training loss for residual waveform components orthogonal to the target."""
import torch


def waveform_projection_loss(estimate, target):
    """Mean fraction of estimate energy outside the zero-mean target direction."""
    estimate = estimate.float()
    target = target.float()
    estimate = estimate - estimate.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    target_energy = target.square().mean(dim=1)
    estimate_energy = estimate.square().mean(dim=1)
    cross = (target * estimate).mean(dim=1)
    # The floor caps the gradient at near-silence. Silent references already use the original loss.
    residual_fraction = 1 - cross.square() / (target_energy * estimate_energy).clamp_min(1e-12)
    audible = target_energy > 1e-10
    return (residual_fraction.clamp_min(0) * audible).sum() / audible.sum().clamp_min(1)

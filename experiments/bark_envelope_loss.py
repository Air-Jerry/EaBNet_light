"""Bandwise spectral-envelope supervision before perceptual equalization."""
import torch


def bark_envelope_loss(estimate_power, target_power, valid_frames, bark_matrix):
    """Absolute log-amplitude error in each Bark band of the valid frames."""
    estimate = estimate_power.float() @ bark_matrix
    target = target_power.float() @ bark_matrix
    mask = valid_frames.unsqueeze(-1)
    counts = mask.sum(dim=1, keepdim=True).clamp_min(1)
    average = target.masked_fill(~mask, 0).sum(dim=1, keepdim=True) / counts
    floor = (.05 * average + .0001 * average.mean(dim=-1, keepdim=True)).clamp_min(1e-10)
    error = .5 * ((estimate + floor).log() - (target + floor).log()).abs()
    per_sample = error.masked_fill(~mask, 0).sum(dim=(1, 2))
    return (per_sample / (counts.flatten() * bark_matrix.shape[-1])).mean()

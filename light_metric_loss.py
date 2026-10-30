"""Training-only WB-PESQ supervision; inference still uses the mixture-only model.

Discriminator architecture adapted from yxlu-0102/MP-SENet, models/discriminator.py
and utils.py. Copyright (c) 2023 Yexin Lu; see third_party/MP-SENet-LICENSE.
"""

import math
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from light_quality_loss import QualityObjective


class LearnedSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.slope = nn.Parameter(torch.ones(1))

    def forward(self, value):
        return torch.sigmoid(self.slope * value)


class MetricDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        channels = 2
        for width in (16, 32, 64, 128):
            layers.extend((spectral_norm(nn.Conv2d(channels, width, 4, 2, 1, bias=False)),
                           nn.InstanceNorm2d(width, affine=True), nn.PReLU(width)))
            channels = width
        layers.extend((nn.AdaptiveMaxPool2d(1), nn.Flatten(),
                       spectral_norm(nn.Linear(128, 64)), nn.Dropout(0.3), nn.PReLU(64),
                       spectral_norm(nn.Linear(64, 1)), LearnedSigmoid()))
        self.layers = nn.Sequential(*layers)

    def forward(self, reference, estimate):
        return self.layers(torch.stack((reference, estimate), dim=1)).flatten()


def real_pesq(reference, estimate):
    from pesq import BufferTooShortError, NoUtterancesError, pesq
    if not np.isfinite(reference).all() or not np.isfinite(estimate).all():
        raise FloatingPointError("Non-finite waveform passed to PESQ")
    if not reference.any() or not estimate.any():
        return None
    try:
        score = float(pesq(16000, reference, estimate, "wb"))
    except (BufferTooShortError, NoUtterancesError):
        return None
    if not math.isfinite(score):
        raise FloatingPointError("PESQ returned a non-finite training label")
    return score


class MetricObjective(QualityObjective):
    """Update D per microbatch, retaining the original accumulated G update.

    D gradients are explicitly averaged between ranks. Its parameters are not in
    the generator optimizer, and stay constant during the subsequent G backward.
    """

    def __init__(self, device, weight=0.05, discriminator_lr=1e-4, score_fn=real_pesq,
                 accelerated=False, **frontend):
        super().__init__(**frontend)
        if self.n_fft != 512 or self.hop_length != 160 or self.win_length != 320 or self.power != 0.5:
            raise ValueError("Metric experiment requires the existing 512/160/320/power0.5 frontend")
        if not math.isfinite(weight) or weight <= 0 or not math.isfinite(discriminator_lr) or discriminator_lr <= 0:
            raise ValueError("Metric weight and discriminator LR must be finite and positive")
        self.device = torch.device(device)
        self.discriminator = MetricDiscriminator().to(self.device)
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=discriminator_lr)
        self.weight = float(weight)
        self.score_fn = score_fn
        self.training = False
        self.updates = 0
        self.accelerated = accelerated
        self._reconstructions = None
        self._score_pool = None
        self.reset_metric_stats()

    def close(self):
        if self._score_pool is not None:
            self._score_pool.shutdown(wait=True)
            self._score_pool = None

    def reconstruct(self, estimate_ri, length):
        cache = self._reconstructions
        if cache is None:
            return super().reconstruct(estimate_ri, length)
        key = (estimate_ri.data_ptr(), tuple(estimate_ri.shape), tuple(estimate_ri.stride()), length)
        if key not in cache:
            cache[key] = super().reconstruct(estimate_ri, length)
        return cache[key]

    def reset_metric_stats(self):
        # valid labels, invalid labels, score sum, prediction squared error, G term sum
        self._metric_totals = torch.zeros(5, device=getattr(self, "device", "cpu"), dtype=torch.float64)

    def state_dict(self):
        return dict(discriminator=self.discriminator.state_dict(),
                    optimizer=self.optimizer.state_dict(), updates=self.updates)

    def load_state_dict(self, state):
        self.discriminator.load_state_dict(state["discriminator"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = int(state["updates"])

    def _magnitude(self, waveform):
        spectrum = torch.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length,
                              win_length=self.win_length, window=self._window(self.win_length, self.device),
                              center=True, pad_mode="reflect", normalized=False,
                              onesided=True, return_complex=True)
        magnitude = spectrum.abs()
        return (magnitude * magnitude.clamp_min(1e-8).pow(self.power - 1)).transpose(-1, -2)

    def __call__(self, estimate, target_stft, frame_list):
        self._reconstructions = {} if self.accelerated and self.training else None
        try:
            return self._compute(estimate, target_stft, frame_list)
        finally:
            self._reconstructions = None

    def _compute(self, estimate, target_stft, frame_list):
        baseline_loss = None if self.accelerated and self.training else super().__call__(estimate, target_stft, frame_list)
        if not self.training:
            return baseline_loss
        if self.accelerated and self.device.type == "cuda" and self._score_pool is None:
            self._score_pool = ProcessPoolExecutor(max_workers=min(4, estimate.shape[0]),
                mp_context=multiprocessing.get_context("spawn"))
        self.discriminator.train()
        examples = []
        host_target = host_waveform = None
        if self.accelerated and len(set(self._lengths)) == len(set(frame_list)) == 1:
            length, frames = self._lengths[0], frame_list[0]
            reconstructed = super().reconstruct(estimate[:, :, :frames], length)
            for index in range(estimate.shape[0]):
                view = estimate[index:index + 1, :, :frames]
                key = (view.data_ptr(), tuple(view.shape), tuple(view.stride()), length)
                self._reconstructions[key] = reconstructed[index:index + 1]
            host_target = self._target[:, :length].detach().cpu().numpy()
            host_waveform = reconstructed.detach().cpu().numpy()
        for index, (length, frames) in enumerate(zip(self._lengths, frame_list)):
            waveform = self.reconstruct(estimate[index:index + 1, :, :frames], length)[0]
            reference = self._target[index, :length]
            waveforms = ((host_target[index], host_waveform[index]) if host_target is not None
                         else (reference.detach().cpu().numpy(), waveform.detach().cpu().numpy()))
            score = (self._score_pool.submit(self.score_fn, *waveforms) if self._score_pool
                     else self.score_fn(*waveforms))
            examples.append((reference, waveform, score))
        examples = [(self._magnitude(reference), self._magnitude(waveform), score)
                    for reference, waveform, score in examples]
        if baseline_loss is None:
            baseline_loss = super().__call__(estimate, target_stft, frame_list)

        predictions = []
        real_predictions = []
        if self.accelerated and len(examples) == 1:
            for clean, enhanced, _ in examples:
                real_predictions.append(self.discriminator(clean[None].detach(), clean[None].detach()))
                predictions.append(self.discriminator(clean[None].detach(), enhanced[None].detach()))
        resolved = []
        for clean, enhanced, score in examples:
            if self._score_pool:
                score = score.result()
            if score is not None and not math.isfinite(score):
                raise FloatingPointError("Non-finite PESQ supervision")
            resolved.append((clean, enhanced, score))
        examples = resolved

        if self.accelerated and len(examples) > 1:
            if len(set(self._lengths)) != 1 or len(set(frame_list)) != 1:
                raise ValueError("Merged metric microbatches require equal waveform lengths")
            # Capture each input gradient before the next D update changes its weights.
            gradient = torch.zeros_like(estimate)
            value = estimate.new_zeros(())
            for example in examples:
                term = self._metric_term(estimate, [example], [], [])
                gradient.add_(torch.autograd.grad(term, estimate, retain_graph=True)[0].detach())
                value.add_(term.detach())
            gradient.div_(len(examples))
            value.div_(len(examples))
            surrogate = (estimate * gradient).sum()
            return baseline_loss + self.weight * (surrogate + (value - surrogate.detach()))
        return baseline_loss + self.weight * self._metric_term(
            estimate, examples, real_predictions, predictions)

    def _metric_term(self, estimate, examples, real_predictions, predictions):

        count = torch.tensor(sum(score is not None for _, _, score in examples),
                             device=self.device, dtype=torch.float32)
        distributed = dist.is_available() and dist.is_initialized()
        if distributed:
            dist.all_reduce(count)
        world = dist.get_world_size() if distributed else 1
        self.optimizer.zero_grad(set_to_none=True)
        discriminator_loss = estimate.new_zeros(())
        for index, (clean, enhanced, score) in enumerate(examples):
            if real_predictions:
                real, predicted = real_predictions[index], predictions[index]
                predictions[index] = predicted.detach()
            else:
                real = self.discriminator(clean[None].detach(), clean[None].detach())
                predicted = self.discriminator(clean[None].detach(), enhanced[None].detach())
                predictions.append(predicted.detach())
            label = 0.0 if score is None else (score - 1.0) / 3.5
            valid = float(score is not None)
            discriminator_loss = discriminator_loss + valid * (
                F.mse_loss(real, torch.ones_like(real)) + (predicted - label).square().mean())
        discriminator_loss = discriminator_loss * world / count.clamp_min(1)
        discriminator_loss.backward()
        parameters = list(self.discriminator.parameters())
        if self.accelerated:
            if any(p.grad is None for p in parameters):
                raise RuntimeError("Missing discriminator gradient")
            flat_gradient = torch.cat([p.grad.reshape(-1) for p in parameters])
            finite = torch.isfinite(flat_gradient).all().to(torch.int32)
        else:
            finite = torch.tensor(int(all(p.grad is not None and torch.isfinite(p.grad).all()
                                          for p in parameters)), device=self.device)
        if distributed:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError("Non-finite discriminator gradient; optimizer step aborted")
        if self.accelerated and distributed:
            dist.all_reduce(flat_gradient)
            flat_gradient.div_(world)
            offset = 0
            for parameter in parameters:
                size = parameter.numel()
                parameter.grad.copy_(flat_gradient[offset:offset + size].view_as(parameter))
                offset += size
        elif distributed:
            for parameter in parameters:
                dist.all_reduce(parameter.grad)
                parameter.grad.div_(world)
        if count.item() > 0:
            self.optimizer.step()
            self.updates += 1

        for parameter in self.discriminator.parameters():
            parameter.requires_grad_(False)
        terms = []
        try:
            for (clean, enhanced, score), predicted in zip(examples, predictions):
                metric = (self.discriminator(clean[None].detach(), enhanced[None]) - 1).square().mean()
                valid = score is not None
                terms.append(metric * float(valid))
                label = 0.0 if not valid else (score - 1.0) / 3.5
                self._metric_totals += torch.stack((
                    metric.new_tensor(float(valid)), metric.new_tensor(float(not valid)),
                    metric.new_tensor(0.0 if not valid else score),
                    (predicted.mean() - label).square() * float(valid), metric.detach() * float(valid),
                )).detach().to(torch.float64)
        finally:
            for parameter in self.discriminator.parameters():
                parameter.requires_grad_(True)
        return torch.stack(terms).mean()

    def metric_stats(self):
        values = self._metric_totals.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(values)
        valid, invalid, scores, error, metric = values.cpu().tolist()
        return dict(valid_pesq_labels=int(valid), invalid_pesq_labels=int(invalid),
                    mean_training_pesq=scores / max(valid, 1),
                    discriminator_calibration_rmse=math.sqrt(error / max(valid, 1)),
                    generator_metric_term=metric / max(valid, 1), discriminator_updates_per_rank=self.updates)

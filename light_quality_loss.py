"""Waveform-aware objectives for the separate quality refinement experiment."""

import math

import torch
import torch.nn.functional as F


class QualityObjective:
    """Pair the existing trainer's feature and loss hooks without changing its defaults.

    ``build_features`` records the original target waveforms for the immediately
    following loss call. A separate instance belongs to each training process.
    Target normalization weights the loss only; model inputs retain their gain.
    """

    COMPONENTS = ("spectral", "mrstft", "waveform", "total")

    def __init__(self, n_fft=512, hop_length=160, win_length=320, power=0.5):
        if not (n_fft > 0 and 0 < win_length <= n_fft and hop_length > 0):
            raise ValueError("Invalid STFT dimensions")
        if not math.isfinite(power) or power <= 0:
            raise ValueError("power must be finite and positive")
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.power = float(power)
        self._windows = {}
        self._target = None
        self._lengths = None
        self.reset_stats()

    def _window(self, size, device):
        key = (size, device.type, device.index)
        if key not in self._windows:
            self._windows[key] = torch.hann_window(size, device=device, dtype=torch.float32)
        return self._windows[key]

    def build_features(self, mixture, target, n_fft, hop_length, win_length,
                       power, device, lengths):
        """Use the original frontend, reflecting each utterance at its true end."""
        from train_light import build_stft_batch

        if (n_fft, hop_length, win_length, power) != (
                self.n_fft, self.hop_length, self.win_length, self.power):
            raise ValueError("Quality objective and training frontend differ")
        device = torch.device(device)
        if mixture.ndim != 3 or target.ndim != 2 or mixture.shape[:2] != target.shape:
            raise ValueError("Expected mixture (B, samples, microphones) and target (B, samples)")
        if isinstance(lengths, torch.Tensor):
            lengths = lengths.detach().cpu().tolist()
        lengths = [int(length) for length in lengths]
        if len(lengths) != target.shape[0] or not lengths:
            raise ValueError("One nonempty length is required per utterance")
        if any(length <= self.n_fft // 2 or length > target.shape[1] for length in lengths):
            raise ValueError("Lengths must exceed n_fft/2 for reflect padding and fit the waveforms")

        with torch.autocast(device_type=device.type, enabled=False):
            self._target = target.detach().to(device=device, dtype=torch.float32, non_blocking=True)
            self._lengths = lengths
            kwargs = dict(n_fft=n_fft, hop_length=hop_length,
                          win_length=win_length, power=power, device=device)
            if all(length == lengths[0] for length in lengths):
                return build_stft_batch(mixture[:, :lengths[0]].float(),
                                        target[:, :lengths[0]].float(), **kwargs)
            max_frames = max(lengths) // self.hop_length + 1
            mixtures, targets = [], []
            for index, length in enumerate(lengths):
                mix, reference = build_stft_batch(
                    mixture[index:index + 1, :length].float(),
                    target[index:index + 1, :length].float(), **kwargs)
                padding = max_frames - mix.shape[1]
                mixtures.append(F.pad(mix, (0, 0, 0, 0, 0, 0, 0, padding)))
                targets.append(F.pad(reference, (0, 0, 0, padding)))
            return torch.cat(mixtures, dim=0), torch.cat(targets, dim=0)

    def reconstruct(self, estimate_ri, length):
        """Invert the training compression, including its epsilon region, in FP32."""
        with torch.autocast(device_type=estimate_ri.device.type, enabled=False):
            estimate_ri = estimate_ri.float()
            compressed = torch.complex(estimate_ri[:, 0].transpose(1, 2),
                                       estimate_ri[:, 1].transpose(1, 2))
            spectrum = compressed * compressed.abs().clamp_min(
                1e-8 ** self.power).pow(1.0 / self.power - 1.0)
            return torch.istft(spectrum, n_fft=self.n_fft,
                               hop_length=self.hop_length, win_length=self.win_length,
                               window=self._window(self.win_length, estimate_ri.device),
                               center=True, normalized=False, onesided=True, length=int(length))

    def _multi_resolution(self, estimate, target):
        terms = []
        for size in (256, 512, 1024):
            kwargs = dict(n_fft=size, hop_length=size // 4, win_length=size,
                          window=self._window(size, estimate.device), center=True,
                          pad_mode="reflect" if target.numel() > size // 2 else "constant",
                          normalized=False, onesided=True, return_complex=True)
            estimated_mag = torch.stft(estimate, **kwargs).abs()
            target_mag = torch.stft(target, **kwargs).abs()
            denominator = torch.linalg.vector_norm(target_mag).clamp_min(
                math.sqrt(target_mag.numel()) * 1e-3)
            convergence = torch.linalg.vector_norm(estimated_mag - target_mag) / denominator
            log_magnitude = (torch.log(estimated_mag + 1e-5)
                             - torch.log(target_mag + 1e-5)).abs().mean()
            terms.append(convergence + log_magnitude)
        return torch.stack(terms).mean()

    def __call__(self, estimate, target_stft, frame_list):
        """Combine gain-weighted compressed spectra, multiscale spectra and waveform L1."""
        if self._target is None or self._lengths is None:
            raise RuntimeError("Call build_features before the quality loss")
        if (estimate.ndim != 4 or estimate.shape != target_stft.shape
                or estimate.shape[1] != 2 or estimate.shape[-1] != self.n_fft // 2 + 1
                or estimate.shape[0] != len(self._lengths)):
            raise ValueError("Estimate and target must share the expected (B, 2, frames, frequencies) shape")
        expected_frames = [length // self.hop_length + 1 for length in self._lengths]
        if [int(value) for value in frame_list] != expected_frames:
            raise ValueError("Loss frame counts differ from the captured waveform lengths")
        if estimate.device != self._target.device or target_stft.device != estimate.device:
            raise ValueError("Features, estimate and target must use the same device")
        if estimate.shape[2] < max(expected_frames):
            raise ValueError("Estimate omits valid frames")

        with torch.autocast(device_type=estimate.device.type, enabled=False):
            estimate, target_stft = estimate.float(), target_stft.float()
            spectral, multiscale, waveform = [], [], []
            for index, (length, frames) in enumerate(zip(self._lengths, expected_frames)):
                estimated = estimate[index, :, :frames]
                reference = target_stft[index, :, :frames]
                estimated_mag = torch.linalg.vector_norm(estimated, dim=0)
                target_mag = torch.linalg.vector_norm(reference, dim=0)
                magnitude_loss = (estimated_mag - target_mag).abs().mean() / target_mag.mean().clamp_min(1e-3)
                complex_loss = (estimated - reference).abs().mean() / reference.abs().mean().clamp_min(1e-3)
                spectral.append(0.5 * (magnitude_loss + complex_loss))
                reconstructed = self.reconstruct(estimated.unsqueeze(0), length)[0]
                raw_target = self._target[index, :length]
                multiscale.append(self._multi_resolution(reconstructed, raw_target))
                target_rms = raw_target.square().mean().sqrt().clamp_min(1e-3)
                waveform.append((reconstructed - raw_target).abs().mean() / target_rms)
            spectral = torch.stack(spectral).mean()
            multiscale = torch.stack(multiscale).mean()
            waveform = torch.stack(waveform).mean()
            total = spectral + 0.1 * multiscale + 0.1 * waveform
        components = torch.stack((spectral, multiscale, waveform, total)).detach()
        weighted = components * len(self._lengths)
        if self._component_totals is None:
            self._component_totals = weighted
        else:
            self._component_totals.add_(weighted)
        self._sample_count += len(self._lengths)
        self._batch_count += 1
        return total

    def reset_stats(self):
        """Clear detached running component means before an epoch or split."""
        self._component_totals = None
        self._sample_count = 0
        self._batch_count = 0

    def stats(self):
        """Return local-rank sample means; callers may aggregate them across DDP ranks."""
        means = ([0.0] * len(self.COMPONENTS) if self._component_totals is None
                 else (self._component_totals / self._sample_count).cpu().tolist())
        return dict(samples=self._sample_count, batches=self._batch_count,
                    **dict(zip(self.COMPONENTS, means)))

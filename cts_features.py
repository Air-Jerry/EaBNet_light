"""CTS-Net waveform/STFT conversion, shared by training and inference.

The paper specifies 16 kHz, a 320-sample Hann window, 160-sample hop,
and 320-point FFT. It does not specify magnitude compression; power=1
is the paper-text configuration. Centered, periodic-Hann, unnormalized
STFT with reflection padding is an explicit implementation assumption.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


def _validate(n_fft: int, hop_length: int, win_length: int, power: float) -> None:
    if (n_fft, hop_length, win_length) != (320, 160, 320):
        raise ValueError("CTS-Net Table I requires n_fft=320, hop_length=160, win_length=320 (161 bins)")
    if not (0 < power <= 1):
        raise ValueError("power must be in (0, 1]; the paper-text setting is 1.0")


def _stft(wave: Tensor, n_fft: int, hop_length: int, win_length: int, power: float) -> Tensor:
    if wave.shape[-1] <= n_fft // 2:
        raise ValueError(f"Audio needs more than {n_fft // 2} samples for centered reflection STFT")
    if not torch.isfinite(wave).all():
        raise ValueError("Audio contains NaN or Inf")
    spectrum = torch.stft(
        wave.float(), n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=torch.hann_window(win_length, periodic=True, device=wave.device),
        center=True, pad_mode="reflect", normalized=False, onesided=True, return_complex=True,
    )
    if power != 1.0:
        spectrum = spectrum * spectrum.abs().clamp_min(1e-12).pow(power - 1.0)
    return torch.view_as_real(spectrum)


def build_mixture_stft(
    mixture: Tensor, n_fft: int = 320, hop_length: int = 160,
    win_length: int = 320, power: float = 1.0,
    device: torch.device = torch.device("cpu"), lengths: Optional[Tensor] = None,
) -> Tensor:
    """Convert (B,N,1) waveforms to (B,T,161,1,2), preserving each boundary.

    A short member of a padded batch must be reflected at its own endpoint,
    not at the longest utterance's endpoint. Masking a loss alone cannot fix
    that discrepancy between batched training and individual inference.
    """
    _validate(n_fft, hop_length, win_length, power)
    if mixture.ndim != 3 or mixture.shape[-1] != 1:
        raise ValueError("CTS-Net is monaural: mixture must have shape (B,N,1)")
    mixture = mixture.to(device=device, dtype=torch.float32)
    if lengths is None:
        lengths = [mixture.shape[1]] * mixture.shape[0]
    else:
        lengths = torch.as_tensor(lengths).cpu().tolist()
    if len(lengths) != mixture.shape[0] or any(int(n) != n or n <= 0 or n > mixture.shape[1] for n in lengths):
        raise ValueError("lengths must contain one valid sample count per waveform")
    spectra = [
        _stft(wave[:int(n), 0], n_fft, hop_length, win_length, power).permute(1, 0, 2)
        for wave, n in zip(mixture, lengths)
    ]
    return pad_sequence(spectra, batch_first=True).unsqueeze(3).contiguous()


def build_stft_batch(
    mixture: Tensor, target: Tensor, n_fft: int, hop_length: int,
    win_length: int, power: float, device: torch.device,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Keep the existing training data/tensor interface; no target level matching."""
    if mixture.ndim != 3 or mixture.shape[-1] != 1:
        raise ValueError("CTS-Net is monaural: mixture must have shape (B,N,1)")
    if target.ndim != 2 or target.shape != mixture.shape[:2]:
        raise ValueError("target must have shape (B,N) matching mixture")
    valid_lengths = ([mixture.shape[1]] * mixture.shape[0] if lengths is None
                     else torch.as_tensor(lengths).cpu().tolist())
    if len(valid_lengths) != mixture.shape[0] or any(int(n) != n or n <= 0 or n > mixture.shape[1] for n in valid_lengths):
        raise ValueError("lengths must contain one valid sample count per waveform")
    for i, n in enumerate(valid_lengths):
        wave = mixture[i, :int(n), 0]
        if wave.numel() and torch.all(wave == wave[0]):
            # Repeated IN layers at zero variance can overflow FP32 gradients.
            # Fail visibly rather than injecting noise, changing epsilon, or
            # silently skipping the optimizer update. Silent clean targets are
            # allowed when the noisy mixture contains a real signal.
            raise ValueError(f"Degenerate constant/silent mixture at batch index {i}; verify the audio/crop")
    mixed = build_mixture_stft(mixture, n_fft, hop_length, win_length, power, device, lengths)
    clean = build_mixture_stft(target.unsqueeze(-1), n_fft, hop_length, win_length, power, device, lengths)
    return mixed, clean.squeeze(3).permute(0, 3, 1, 2).contiguous()


def reconstruct_waveform(
    estimate_ri: Tensor, length: int, n_fft: int = 320, hop_length: int = 160,
    win_length: int = 320, power: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Invert exactly the front end above, without access to a clean reference."""
    _validate(n_fft, hop_length, win_length, power)
    if estimate_ri.ndim != 4 or estimate_ri.shape[1] != 2 or estimate_ri.shape[-1] != 161:
        raise ValueError("estimate_ri must have shape (B,2,T,161)")
    if length <= 0:
        raise ValueError("length must be positive")
    estimate_ri = estimate_ri.to(device=device, dtype=torch.float32)
    spectrum = torch.complex(estimate_ri[:, 0], estimate_ri[:, 1]).transpose(1, 2)
    if power != 1.0:
        spectrum = spectrum * spectrum.abs().pow(1.0 / power - 1.0)
    return torch.istft(
        spectrum, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=torch.hann_window(win_length, periodic=True, device=device),
        center=True, normalized=False, onesided=True, length=length,
    )

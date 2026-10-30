"""Bounded waveform streaming for the unchanged 512/160/320 centered frontend.

Frames are built only from received samples. The 20 ms analysis/synthesis
buffer budget is separate from device, compute and audio-driver latency.
"""

import torch
import torch.nn.functional as F


class CausalWaveformStream:
    sample_rate = 16000
    hop_length = 160
    win_length = 320
    n_fft = 512
    buffering_delay_samples = 320

    def __init__(self, model):
        if model.training or not getattr(model, "is_causal", False):
            raise ValueError("Use an eval-mode causal model")
        self.model = model
        self.state = None
        self.pending = self.previous = self.tail = self.overlap = None
        self.total_samples = self.emitted_samples = 0
        self.closed = False
        self.window = None

    @torch.inference_mode()
    def warmup(self, batch_size=1):
        """Initialize GPU kernels before opening audio; discard all synthetic state."""
        if self.pending is not None or self.total_samples:
            raise RuntimeError("Warm up before receiving audio")
        device = next(self.model.parameters()).device
        temporary = type(self)(self.model)
        silence = torch.zeros(batch_size, 160, 8, device=device, dtype=torch.float32)
        for _ in range(self.model.context + 1):
            temporary.push(silence)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def _frame(self, hop):
        if self.window is None:
            self.window = torch.hann_window(self.win_length, device=hop.device, dtype=torch.float32)
        if self.previous is None:
            # The unobserved sample at -160 has zero Hann weight.
            previous = torch.cat((torch.zeros_like(hop[:, :1]), hop[:, 1:].flip(1)), dim=1)
        else:
            previous = self.previous
        frame = torch.cat((previous, hop), dim=1).transpose(1, 2)
        spectrum = torch.fft.rfft(F.pad(frame * self.window, (96, 96)), n=self.n_fft)
        compressed = spectrum * spectrum.abs().clamp_min(1e-8).pow(-0.5)
        features = torch.view_as_real(compressed.transpose(1, 2)).unsqueeze(1)
        estimate, self.state = self.model.forward_stream(features, self.state)
        value = torch.complex(estimate[:, 0, 0], estimate[:, 1, 0])
        value = value * value.abs().clamp_min(1e-4)
        waveform = torch.fft.irfft(value, n=self.n_fft)[:, 96:416] * self.window
        output = hop.new_empty((hop.shape[0], 0))
        if self.overlap is not None:
            denominator = self.window[:160].square() + self.window[160:].square()
            output = (self.overlap + waveform[:, :160]) / denominator
        self.previous = hop.detach().clone()
        self.overlap = waveform[:, 160:].detach().clone()
        return output

    @torch.inference_mode()
    def push(self, samples):
        if self.closed:
            raise RuntimeError("Stream is already flushed")
        if samples.ndim != 3 or samples.shape[-1] != 8 or samples.dtype != torch.float32:
            raise ValueError("Expected float32 waveform (B,samples,8)")
        if self.pending is not None and (samples.shape[0] != self.pending.shape[0] or samples.device != self.pending.device):
            raise ValueError("Batch and device must remain fixed for a stream")
        self.total_samples += samples.shape[1]
        self.tail = samples if self.tail is None else torch.cat((self.tail, samples), dim=1)
        self.tail = self.tail[:, -320:].clone()
        pending = samples if self.pending is None else torch.cat((self.pending, samples), dim=1)
        outputs = []
        offset = 0
        while offset + 160 <= pending.shape[1]:
            outputs.append(self._frame(pending[:, offset:offset + 160]))
            offset += 160
        self.pending = pending[:, offset:].clone()
        output = torch.cat(outputs, dim=1) if outputs else samples.new_empty((samples.shape[0], 0))
        self.emitted_samples += output.shape[1]
        return output

    @torch.inference_mode()
    def flush(self):
        if self.closed:
            raise RuntimeError("Stream is already flushed")
        if self.total_samples <= 256:
            raise ValueError("Reflect-padded training frontend requires more than n_fft/2 samples")
        remaining = self.pending.shape[1]
        padding = self.tail.flip(1)[:, 1:161]
        hop = torch.cat((self.pending, padding[:, :160 - remaining]), dim=1)
        output = self._frame(hop)
        if remaining:
            final = self.overlap[:, :remaining] / self.window[160:160 + remaining].square()
            output = torch.cat((output, final), dim=1)
        self.emitted_samples += output.shape[1]
        if self.emitted_samples != self.total_samples:
            raise RuntimeError("Streaming output length differs from received audio")
        self.closed = True
        return output

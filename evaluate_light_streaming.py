"""Audit all 600 comparison utterances using actual 10 ms waveform input blocks.

Reuse the existing evaluator and metrics without modifying evaluate_light.py.
Run on an idle device: simultaneous training invalidates the latency measurement.
Pass --reference-summary from the identical checkpoint's complete offline run.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

import evaluate_light as evaluator
from light_causal_stream import CausalWaveformStream


IDENTITY_FIELDS = ('checkpoint_sha256', 'structure_candidate', 'candidate_implementation_sha256',
    'candidate_evidence', 'frontend', 'frontend_protocol', 'num_mics', 'mixture_ref_mic',
    'selected_pairs_sha256', 'metadata_sha256', 'metric_protocol', 'checkpoint_model_config', 'device')


def require_full_comparison(report):
    if report.get('status') != 'complete' or report.get('completed_samples') != 600 or report.get('expected_samples') != 600:
        raise ValueError('Streaming audit requires the same complete 600-utterance comparison')
    if report['metric_protocol']['target_gain_matching'] or report['metric_protocol']['external_normalization_clipping_or_alignment']:
        raise ValueError('Streaming audit forbids gain matching, external normalization or alignment')
    if report['structure_candidate'] not in ('causal_unet_spatial_v1', 'causal_unet_global_v1', 'causal_recurrent_spatial_v1', 'causal_recurrent_unet65_v1', 'causal_recurrent_tcn_v1', 'causal_frequency_fourier_v1'):
        raise ValueError('Streaming audit requires a supported causal checkpoint')
    if report['candidate_evidence'].get('model_causal') is not True or report['candidate_evidence'].get('future_frames') != 0:
        raise ValueError('Checkpoint evidence must identify a causal model with zero future frames')


class WaveformAdapter(nn.Module):
    def __init__(self, model, build_features, reconstruct):
        super().__init__()
        self.model, self.M = model, model.M
        self.build_features, self.reconstruct = build_features, reconstruct
        self.device = next(model.parameters()).device
        self.hop_ms = []
        self.flush_ms = []
        self.utterances = self.samples = 0
        self.waveform_max_error = 0.0
        self.frontend = dict(n_fft=512, hop_length=160, win_length=320, power=.5)
        CausalWaveformStream(model).warmup()

    def synchronize(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def forward(self, mixture):
        if mixture.device.type != 'cpu' or mixture.ndim != 3 or mixture.shape[0] != 1 or mixture.shape[-1] != 8:
            raise ValueError('Expected one CPU float32 waveform with eight ordered channels')
        features = self.build_features(mixture, device=self.device, **self.frontend)
        offline = self.reconstruct(self.model(features), length=mixture.shape[1], device=self.device, **self.frontend).cpu()
        stream = CausalWaveformStream(self.model)
        parts = []
        for start in range(0, mixture.shape[1], 160):
            self.synchronize()
            began = time.perf_counter()
            parts.append(stream.push(mixture[:, start:start + 160].to(self.device)).cpu())
            self.synchronize()
            self.hop_ms.append(1000 * (time.perf_counter() - began))
        began = time.perf_counter()
        parts.append(stream.flush().cpu())
        self.synchronize()
        self.flush_ms.append(1000 * (time.perf_counter() - began))
        streamed = torch.cat(parts, dim=1)
        if streamed.shape != offline.shape or not torch.isfinite(streamed).all():
            raise ValueError('Streaming output has an invalid length or non-finite samples')
        error = (streamed - offline).abs().max().item()
        self.waveform_max_error = max(self.waveform_max_error, error)
        if not torch.allclose(streamed, offline, rtol=1e-3, atol=1e-5):
            raise ValueError(f'Streaming waveform differs from the saved model: max error {error}')
        self.utterances += 1
        self.samples += mixture.shape[1]
        return streamed

    def evidence(self):
        values = np.asarray(self.hop_ms)
        return dict(input_hop_samples=160, input_hop_ms=10, future_stft_frames=0,
            buffering_delay_samples=320, buffering_delay_ms=20, stream_reset_per_utterance=True,
            warmup_state_discarded=True, completed_utterances=self.utterances, audio_samples=self.samples,
            measured_hops=len(values), all_hop_ms=dict(mean=float(values.mean()),
                p99=float(np.percentile(values, 99)), max=float(values.max())),
            deadline_miss_fraction=float((values >= 10).mean()), all_hops_below_10ms=bool(values.max() < 10),
            flush_max_ms=max(self.flush_ms), full_vs_stream_waveform_max_error=self.waveform_max_error,
            latency_scope='CPU input transfer + streaming frontend/model/synthesis + CPU output transfer; excludes audio driver',
            end_of_stream='Reflect only samples already received; no delay correction or clean-reference access')


def run(argv=None):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('--reference-summary')
    selected, remaining = selector.parse_known_args(argv)
    args = evaluator.parse_args(remaining)
    if not selected.reference_summary:
        selector.error('--reference-summary from the complete offline run is required')
    if getattr(args, 'checkpoint_dir', None) or args.max_samples != 0 or args.match_estimate_level != 'no':
        raise ValueError('Use one checkpoint, all 600 records and no gain matching')
    reference_path = Path(selected.reference_summary).expanduser().resolve()
    baseline = json.loads(reference_path.read_text())
    require_full_comparison(baseline)
    device = evaluator.get_device(args.device)
    if baseline.get('device') != str(device):
        raise ValueError('Use an offline reference summary from the same device; CPU/GPU numerical drift is a separate comparison')
    if baseline['checkpoint_sha256'] != evaluator.sha256_file(args.checkpoint):
        raise ValueError('Reference summary belongs to a different checkpoint')
    records, info, protected = evaluator.resolve_evaluation_inputs(args)
    if len(records) != 600 or evaluator.pairs_sha256(records) != baseline['selected_pairs_sha256'] or info['metadata_sha256'] != baseline['metadata_sha256']:
        raise ValueError('Streaming inputs differ from the complete offline comparison')
    _, _, metadata_path, csv_path, result_path = evaluator._output_plan(args, records, protected | {reference_path}, info)
    if any(path.exists() for path in (metadata_path, csv_path, result_path) if path is not None):
        raise FileExistsError('Use new output files for the streaming audit')
    load, build, reconstruct = evaluator.load_model, evaluator.build_stft_batch, evaluator.reconstruct_waveform
    wrapped = []

    def streaming_load(path, device, expected=None):
        model, saved = load(path, device, expected)
        config = evaluator.resolve_frontend(args, saved)
        if config != dict(sample_rate=16000, n_fft=512, hop_length=160, win_length=320, power=.5, target_ref_mic=0):
            raise ValueError('Streaming frontend differs from the trained 512/160/320, power0.5, ref0 protocol')
        wrapped.append(WaveformAdapter(model, build, reconstruct).eval())
        return wrapped[-1], saved

    evaluator.load_model = streaming_load
    evaluator.build_stft_batch = lambda mixture, **kwargs: mixture
    evaluator.reconstruct_waveform = lambda estimate, **kwargs: estimate
    report = None
    try:
        report = evaluator.evaluate(args)
        require_full_comparison(report)
        if len(wrapped) != 1 or wrapped[0].utterances != 600:
            raise ValueError('Not every comparison utterance was streamed')
        report['waveform_streaming'] = wrapped[0].evidence()
        report['waveform_streaming']['adapter_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report['waveform_streaming']['stream_source_sha256'] = evaluator.sha256_file(Path(__file__).with_name('light_causal_stream.py'))
        report['waveform_streaming']['reference_summary_sha256'] = evaluator.sha256_file(reference_path)
        report['waveform_streaming']['inference_backend'] = dict(device=str(device),
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
        for name in IDENTITY_FIELDS:
            if report[name] != baseline[name]:
                raise ValueError(f'Streaming comparison identity differs: {name}')
        for name in evaluator.METRICS:
            if abs(report['mean']['noisy'][name] - baseline['mean']['noisy'][name]) > 1e-9:
                raise ValueError(f'Streaming noisy baseline differs: {name}')
        if abs(report['mean']['enhanced']['pesq'] - baseline['mean']['enhanced']['pesq']) > 1e-4:
            raise ValueError('Full600 streaming PESQ differs from offline PESQ')
        report['streaming_quality_target_reached'] = report['mean']['enhanced']['pesq'] >= 3.4
        print(json.dumps(report['waveform_streaming'], indent=2), flush=True)
        return report
    except Exception as error:
        if result_path.exists():
            report = json.loads(result_path.read_text()) if report is None else report
            report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        evaluator.load_model, evaluator.build_stft_batch, evaluator.reconstruct_waveform = load, build, reconstruct
        if report is not None:
            result_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run()

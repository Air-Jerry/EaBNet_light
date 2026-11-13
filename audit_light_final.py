"""Final same600 offline/streaming quality, causal latency and epoch-budget audit."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

import benchmark_light_causal as benchmark
import evaluate_light as evaluator
import evaluate_light_streaming as streaming
from light_candidate_runtime import PARALLEL_PHASE_FILTER_CANDIDATE, DOUBLE_PHASE_FILTER_CANDIDATE, PHASE_FILTER_CANDIDATE, read_candidate_checkpoint


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def require_protocol(report):
    streaming.require_full_comparison(report)
    if report['frontend'] != dict(sample_rate=16000, n_fft=512, hop_length=160,
                                  win_length=320, power=.5, target_ref_mic=0):
        raise ValueError('Final frontend differs from training')
    if report['num_mics'] != 8 or report['mixture_ref_mic'] != 0:
        raise ValueError('Final microphone count or reference changed')
    if report['metric_protocol'].get('pesq') != 'pesq wideband' or report['metric_protocol'].get('gain_rule') != 'none':
        raise ValueError('Require wideband PESQ without reference gain correction')
    expected=dict(window='periodic Hann', center=True, pad_mode='reflect', normalized=False,
        onesided=True, waveform_normalization='none', mixture_channels='first 8 channels, original order',
        evaluation_extent='full common-length utterance; no training crop or batch padding', inference_precision='float32')
    if any(report['frontend_protocol'].get(name) != value for name, value in expected.items()):
        raise ValueError('Window, normalization, channel order or evaluation extent changed')


def require_deadline(record):
    if (not record['all_hops_below_10ms'] or not all(math.isfinite(value) for value in record['all_hop_ms'].values())
        or not 0 < record['all_hop_ms']['max'] < 10 or record['measured_hops'] <= 0):
        raise ValueError('A measured waveform hop missed the10ms deadline')


def validate_latency(latency, checkpoint_sha):
    require_deadline(latency)
    if (latency.get('status') != 'measured' or latency['checkpoint_sha256'] != checkpoint_sha
        or not math.isfinite(latency['future_perturbation_max_error']) or latency['future_perturbation_max_error'] >= 1e-6
        or not math.isfinite(latency['full_vs_stream_waveform_max_error']) or latency['full_vs_stream_waveform_max_error'] >= 1e-5
        or not latency['warmup_state_discarded'] or latency['bounded_model_cache_bytes'] <= 0):
        raise ValueError('Final trained-model causality, waveform or bounded-state check failed')


def validate_training(state, report, timing, checkpoint_sha):
    require_protocol(report)
    epoch = int(state['epoch'])
    if epoch < 1 or report['checkpoint_sha256'] != checkpoint_sha:
        raise ValueError('Require the evaluated trained checkpoint, not an initial model')
    if not math.isfinite(report['mean']['enhanced']['pesq']) or report['mean']['enhanced']['pesq'] < 3.4:
        raise ValueError('Full600 trained PESQ has not reached3.4')
    if (timing.get('status'), timing.get('epoch'), timing.get('training_samples'),
        timing.get('validation_samples'), timing.get('full_metric_samples'),
        timing.get('optimizer_updates_this_epoch'), timing.get('global_batch_size')) != (
            'complete', epoch, 53600, 600, 600, 6700, 8):
        raise ValueError('Training evidence does not cover a complete53600/600/600 epoch')
    seconds = timing['total_seconds']
    if not math.isfinite(seconds) or not 0 < seconds <= 3600:
        raise ValueError('Complete training epoch exceeded one hour')
    optimizer = state['optimizer_state_dict']['state']
    if state.get('args', {}).get('structure_candidate') in (PHASE_FILTER_CANDIDATE, DOUBLE_PHASE_FILTER_CANDIDATE, PARALLEL_PHASE_FILTER_CANDIDATE):
        groups = state['optimizer_state_dict']['param_groups']
        source_epoch = state['args']['phase_filter_source_epoch']
        base, added = map(lambda group: set(group['params']), groups)
        valid_updates = (0 < source_epoch < epoch and not (base & added)
            and set(optimizer) <= base | added and len(base & optimizer.keys()) == 235
            and len(added) == 6 and added <= optimizer.keys()
            and all(int(optimizer[i]['step']) == epoch*6700 for i in base & optimizer.keys())
            and all(int(optimizer[i]['step']) == (epoch-source_epoch)*6700 for i in added))
    else:
        valid_updates = bool(optimizer) and {int(item['step']) for item in optimizer.values()} == {epoch * 6700}
    if not valid_updates:
        raise ValueError('Saved Adam updates do not match the complete epoch budget')
    if not all(torch.isfinite(value).all() for value in state['model_state_dict'].values()):
        raise ValueError('Model contains non-finite values')
    if not all(torch.isfinite(value).all() for item in optimizer.values()
               for value in item.values() if torch.is_tensor(value)):
        raise ValueError('Adam state contains non-finite values')


def validate_final(training, reference, streamed, latency):
    for report in (training, reference, streamed):
        require_protocol(report)
        if not all(math.isfinite(value) for metrics in report['mean'].values() for value in metrics.values()):
            raise ValueError('Evaluation contains non-finite metrics')
        if not math.isfinite(report['mean']['enhanced']['pesq']) or report['mean']['enhanced']['pesq'] < 3.4:
            raise ValueError('Every complete quality evaluation must reach3.4')
    for key in streaming.IDENTITY_FIELDS:
        if reference[key] != streamed[key] or reference[key] != training[key]:
            raise ValueError(f'Final evaluation identity differs: {key}')
    backend = reference['inference_backend']
    if backend != streamed['waveform_streaming']['inference_backend']:
        raise ValueError('Offline and streaming numerical backends differ')
    if backend['matmul_allow_tf32'] or backend['cudnn_allow_tf32']:
        raise ValueError('Final reference and stream must both use FP32 without TF32')
    for key, value in reference['mean']['noisy'].items():
        if abs(value - streamed['mean']['noisy'][key]) > 1e-9 or abs(value - training['mean']['noisy'][key]) > 1e-9:
            raise ValueError('Raw noisy baseline changed')
    if abs(reference['mean']['enhanced']['pesq'] - streamed['mean']['enhanced']['pesq']) > 1e-4:
        raise ValueError('Full600 offline and streaming PESQ disagree')
    evidence = streamed['waveform_streaming']
    if (evidence['completed_utterances'], evidence['input_hop_samples'], evidence['future_stft_frames'],
        evidence['buffering_delay_samples'], evidence['stream_reset_per_utterance'], evidence['warmup_state_discarded']) != (
            600, 160, 0, 320, True, True):
        raise ValueError('Waveform streaming extent, buffering or reset differs')
    require_deadline(evidence)
    validate_latency(latency, reference['checkpoint_sha256'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--training-summary', type=Path, required=True)
    parser.add_argument('--training-timings', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--evaluator-sha256', required=True)
    parser.add_argument('--launcher-pid', type=int, required=True)
    parser.add_argument('--launcher-cookie', required=True)
    parser.add_argument('--train-dir', type=Path, default=Path('/data/ssd1/jinrui.yang/training_set'))
    parser.add_argument('--val-dir', type=Path, default=Path('/data/ssd1/jinrui.yang/newset_rt60_005-015_on_NOISEX92/babble/training_set'))
    args = parser.parse_args(argv)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        fields = Path(f'/proc/{args.launcher_pid}/stat').read_text().rsplit(') ', 1)[1].split()
        if fields[19] == args.launcher_cookie and fields[0] != 'Z':
            raise ValueError('Training is still live; do not share its GPU for latency measurements')
    except FileNotFoundError:
        pass
    evaluator_path = Path(evaluator.__file__)
    if sha(evaluator_path) != args.evaluator_sha256:
        raise ValueError('Original evaluator changed; do not overwrite it')
    checkpoint = args.checkpoint.expanduser().resolve()
    state, candidate = read_candidate_checkpoint(checkpoint)
    training = json.loads(args.training_summary.read_text())
    timings = [json.loads(line) for line in args.training_timings.read_text().splitlines() if line.strip()]
    timing = next(row for row in timings if row['epoch'] == state['epoch'])
    validate_training(state, training, timing, sha(checkpoint))
    if Path(state['args']['train_dir']).resolve() != args.train_dir.resolve():
        raise ValueError('Checkpoint training data differs from the requested training set')
    output = args.output_dir.expanduser().resolve()
    protected = (checkpoint.parent, args.train_dir.resolve(), args.val_dir.resolve(),
                 args.training_summary.resolve(), args.training_timings.resolve())
    if any(output == path or output in path.parents or path in output.parents for path in protected):
        raise ValueError('Final audit outputs overlap protected inputs')
    if not torch.cuda.is_available() or torch.cuda.mem_get_info(0)[0] <= 10 * 2**30:
        raise ValueError('Require an idle CUDA GPU with more than10GiB free')
    output.mkdir(parents=True, exist_ok=False)
    base = ['--checkpoint', str(checkpoint), '--candidate', candidate, '--val-dir', str(args.val_dir),
            '--device', 'cuda', '--max-samples', '0', '--save-samples', 'no', '--match-estimate-level', 'no']
    result = dict(status='running', checkpoint_sha256=sha(checkpoint), epoch=state['epoch'],
                  evaluator_sha256=args.evaluator_sha256, device=torch.cuda.get_device_name(0),
                  audit_runner_sha256=sha(__file__), streaming_adapter_sha256=sha(streaming.__file__),
                  torch_version=torch.__version__)
    report_path = output / 'final_audit.json'
    started = time.perf_counter()
    try:
        offline_path = output / 'offline/summary.json'
        offline_args = evaluator.parse_args([*base, '--estimate-dir', str(output / 'offline'), '--save-json', str(offline_path)])
        records, _, _ = evaluator.resolve_evaluation_inputs(offline_args)
        private_hashes = {path:sha(path) for record in records for path in (record.mixture_path, record.target_path)}
        with (output / 'audit_console.log').open('x') as handle, redirect_stdout(handle), redirect_stderr(handle):
            reference = evaluator.evaluate(offline_args)
            reference['inference_backend'] = dict(device=str(evaluator.get_device('cuda')),
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
            offline_path.write_text(json.dumps(reference, indent=2, allow_nan=False) + '\n')
            require_protocol(reference)
            if reference['mean']['enhanced']['pesq'] < 3.4:
                raise ValueError('Same-backend FP32 offline PESQ is below3.4')
            # A separate process keeps the benchmark's one-time thread setup independent.
            torch.cuda.empty_cache()
            measured = subprocess.run([sys.executable, '-u', benchmark.__file__, '--candidate', candidate,
                '--checkpoint', str(checkpoint), '--train-dir', str(args.train_dir),
                '--output-dir', str(output / 'latency')], stdout=handle, stderr=subprocess.STDOUT, timeout=180)
            if measured.returncode != 0:
                raise ValueError('Trained-model latency benchmark failed; inspect audit_console.log')
            latency = json.loads((output / 'latency/streaming_benchmark.json').read_text())
            validate_latency(latency, reference['checkpoint_sha256'])
            streamed = streaming.run([*base, '--reference-summary', str(offline_path),
                '--estimate-dir', str(output / 'streamed'), '--save-json', str(output / 'streamed/summary.json')])
        validate_final(training, reference, streamed, latency)
        if sha(evaluator_path) != args.evaluator_sha256 or sha(checkpoint) != result['checkpoint_sha256']:
            raise ValueError('Evaluator or checkpoint changed during final audit')
        if not all(sha(path) == digest for path, digest in private_hashes.items()):
            raise ValueError('Input audio changed during final audit')
        result.update(status='passed', training_epoch_seconds=timing['total_seconds'],
            offline_full600_pesq=reference['mean']['enhanced']['pesq'], streamed_full600_pesq=streamed['mean']['enhanced']['pesq'],
            same_numerical_backend=True, all600_waveforms_match=True, audio_files_checked=len(private_hashes),
            input_audio_unchanged=True, private_audio_hashes_exported=False,
            waveform_streaming={key:streamed['waveform_streaming'][key] for key in (
                'all_hop_ms','measured_hops','deadline_miss_fraction','buffering_delay_ms','full_vs_stream_waveform_max_error')},
            bounded_cache_bytes=latency['bounded_model_cache_bytes'], future_error=latency['future_perturbation_max_error'])
    except Exception as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        result['elapsed_seconds'] = time.perf_counter() - started
        report_path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()

"""Check whether gains from fixed8 training fits transfer to held-out mixtures.

This is a small diagnostic on the comparison set, not the full600 goal metric or
an independent test set. It must never replace full-data training/evaluation.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from pesq import pesq
import soundfile as sf
import torch

import evaluate_light as evaluator
from experiments.causal_frequency_fourier import CausalFrequencyFourierTCN
from experiments.causal_tcn_mapping import CausalTCNMapping
from light_candidate_runtime import read_candidate_checkpoint
from train_light import resolve_record_path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def target_data(path):
    value, sample_rate = sf.read(path, dtype='float32', always_2d=True)
    assert sample_rate == 16000 and np.isfinite(value).all()
    return value[:, 0].copy()


def select_records(rows, directory, excluded_targets):
    groups = {}
    for row in rows:
        groups.setdefault(float(row['snr_db']), []).append(row)
    randomizer = random.Random(7321)
    for records in groups.values():
        randomizer.shuffle(records)
    selected, seen = [], set()
    while len(selected) < 32:
        available = False
        for snr in sorted(groups):
            if not groups[snr]:
                continue
            available = True
            row = groups[snr].pop()
            target = resolve_record_path(directory, row['target_path'])
            fingerprint = hashlib.sha256(target_data(target).tobytes()).hexdigest()
            if fingerprint in excluded_targets or row['sample_id'] in seen:
                continue
            mixture = resolve_record_path(directory, row['mixture_path'])
            selected.append((row['sample_id'], snr, mixture, target)); seen.add(row['sample_id'])
            if len(selected) == 32:
                break
        if not available:
            raise ValueError('Not enough held-out records')
    assert len(selected) == len(seen) == 32
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--val-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--full-source-checkpoint', type=Path, required=True)
    parser.add_argument('--full-pmsqe-checkpoint', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.manual_seed(1234)
    root = Path(__file__).resolve().parents[1]
    eight = root / 'logs_cbam_flat_projection64/overfit8_epoch40_v2/dataset'
    excluded = {hashlib.sha256(target_data(eight / f'target_{index}.wav').tobytes()).hexdigest() for index in range(1, 9)}
    with (args.val_dir / 'metadata.csv').open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 600
    records = select_records(rows, args.val_dir, excluded)
    pair_hash = hashlib.sha256(json.dumps([(sid, str(mix), str(target)) for sid, _, mix, target in records]).encode()).hexdigest()
    audio_hashes = {path:sha(path) for _, _, mixture, target in records for path in (mixture, target)}
    assert not any(hashlib.sha256(target_data(target).tobytes()).hexdigest() in excluded for _, _, _, target in records)
    control_root = root / 'logs_causal_recurrent_tcn_v1/frequency_fourier_probe_20261031/controlled_cpu'
    control = json.loads((control_root / 'probe.json').read_text())
    pmsqe_root = root / 'logs_causal_recurrent_tcn_v1/pmsqe_probe_20261031/controlled_cpu'
    combined_root = root / 'logs_causal_recurrent_tcn_v1/fourier_pmsqe_probe_20261031/controlled_cpu'
    for folder in (control_root, pmsqe_root, combined_root):
        report = json.loads((folder / 'probe.json').read_text())
        assert report['status'] == 'complete' and report['updates'] == 200
        assert report['audio_file_occurrences_unchanged'] == 16
    constructors = [
        ('full_source_epoch1', Path(control['warmstart']['checkpoint']), CausalTCNMapping, True),
        ('full_source_best', args.full_source_checkpoint, CausalTCNMapping, True),
        ('full_pmsqe_epoch1', args.full_pmsqe_checkpoint, CausalTCNMapping, True),
        ('physical_fit8', control_root / 'tcn_control.pt', CausalTCNMapping, False),
        ('pmsqe_fit8', pmsqe_root / 'pmsqe_training_only_fit.pt', CausalTCNMapping, False),
        ('fourier_pmsqe_fit8', combined_root / 'pmsqe_training_only_fit.pt', CausalFrequencyFourierTCN, False)]
    output = args.output_dir.expanduser().resolve()
    for protected in (args.val_dir.resolve(), eight, *(path.parent.resolve() for _, path, _, _ in constructors)):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError('Diagnostic outputs overlap protected inputs')
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status='running', scope='32 fixed held-out mixtures; comparison-set diagnostic, not full600 or independent-test quality',
        selection_seed=7321, selection='SNR-stratified, excludes exact decoded targets from the8-fit cohort',
        holdout_unit='mixture/target records; not a speaker-disjoint split',
        held_out_samples=32, selected_pairs_sha256=pair_hash,
        snr_counts={str(snr):sum(row[1] == snr for row in records) for snr in sorted({row[1] for row in records})},
        metadata_sha256=sha(args.val_dir / 'metadata.csv'), private_audio_hashes_exported=False,
        frontend=dict(sample_rate=16000, n_fft=512, hop_length=160, win_length=320, power=.5, microphones=8, reference=0),
        normalization='none', gain_matching=False, pesq_mode='wb', results=[])
    started = time.perf_counter()
    try:
        for name, checkpoint, constructor, complete in constructors:
            model = constructor().eval()
            if complete:
                state, _ = read_candidate_checkpoint(checkpoint, 'causal_recurrent_tcn_v1')
                weights = state['model_state_dict']
            else:
                weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
            assert all(torch.isfinite(value).all() for value in weights.values())
            model.load_state_dict(weights, strict=True)
            scores, noisy, per_snr = [], [], {}
            with torch.inference_mode():
                for _, snr, mixture_path, target_path in records:
                    mixture, sample_rate = sf.read(mixture_path, dtype='float32', always_2d=True)
                    target = target_data(target_path); length = min(len(mixture), len(target))
                    assert sample_rate == 16000 and mixture.shape[1] >= 8
                    mixture = mixture[:length, :8].copy(); target = target[:length]
                    features = evaluator.build_stft_batch(torch.from_numpy(mixture)[None], device=torch.device('cpu'),
                        n_fft=512, hop_length=160, win_length=320, power=.5)
                    estimate = evaluator.reconstruct_waveform(model(features), length=length, device=torch.device('cpu'),
                        n_fft=512, hop_length=160, win_length=320, power=.5)[0].numpy()
                    assert np.isfinite(estimate).all()
                    value = float(pesq(16000, target, estimate, 'wb'))
                    scores.append(value); per_snr.setdefault(str(snr), []).append(value)
                    noisy.append(float(pesq(16000, target, mixture[:, 0], 'wb')))
            if result['results']:
                assert noisy == result['results'][0]['per_sample_noisy_pesq']
            row = dict(model=name, checkpoint_sha256=sha(checkpoint), mean_pesq=float(np.mean(scores)),
                per_sample_pesq=scores, per_sample_noisy_pesq=noisy,
                per_snr={key:float(np.mean(values)) for key, values in per_snr.items()})
            result['results'].append(row)
            (output / 'transfer.json').write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(dict(model=name, held_out32_pesq=row['mean_pesq'], elapsed_seconds=time.perf_counter() - started)), flush=True)
            del model
        assert all(sha(path) == digest for path, digest in audio_hashes.items())
        result.update(status='complete', audio_files_unchanged=True, checked_audio_file_occurrences=len(audio_hashes),
            exact8_fit_targets_excluded=True, elapsed_seconds=time.perf_counter() - started)
    except Exception as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        (output / 'transfer.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()

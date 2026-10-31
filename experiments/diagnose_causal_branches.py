"""Paired branch ablation on the existing fixed32 diagnostic cohort, without training."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from pesq import pesq
import soundfile as sf
import torch

import evaluate_light as evaluator
from experiments.diagnose_fit_transfer import select_records, sha, target_data
from experiments.causal_tcn_mapping import CausalTCNMapping
from light_causal_recurrent65_model import CausalRecurrentUNet65
from light_candidate_runtime import read_candidate_checkpoint


def spatial_only(model, features):
    return CausalRecurrentUNet65.process(model, features)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--val-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    root = Path(__file__).resolve().parents[1]
    eight = root / 'logs_cbam_flat_projection64/overfit8_epoch40_v2/dataset'
    excluded = {hashlib.sha256(target_data(eight / f'target_{i}.wav').tobytes()).hexdigest() for i in range(1, 9)}
    with (args.val_dir / 'metadata.csv').open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 600
    records = select_records(rows, args.val_dir, excluded)
    pair_sha = hashlib.sha256(json.dumps([(sid, str(mix), str(target)) for sid, _, mix, target in records]).encode()).hexdigest()
    prior = json.loads((root / 'logs_causal_recurrent_tcn_v1/fit_transfer32_20261101/transfer.json').read_text())
    assert pair_sha == prior['selected_pairs_sha256'] and prior['status'] == 'complete'
    original_sha = sha(args.checkpoint)
    evaluator_sha = sha(evaluator.__file__)
    state, _ = read_candidate_checkpoint(args.checkpoint, 'causal_recurrent_tcn_v1')
    assert state['epoch'] >= 1 and all(torch.isfinite(v).all() for v in state['model_state_dict'].values())
    model = CausalTCNMapping().eval()
    model.load_state_dict(state['model_state_dict'], strict=True)
    output = args.output_dir.resolve()
    for protected in (args.checkpoint.resolve().parent, args.val_dir.resolve(), eight):
        assert output != protected and output not in protected.parents and protected not in output.parents
    output.mkdir(parents=True, exist_ok=False)
    audio_hashes = {p:sha(p) for _, _, mixture, target in records for p in (mixture, target)}
    result = dict(status='running', scope='Paired32-record branch ablation; not full600 goal or separately retrained branch performance',
        checkpoint_sha256=original_sha, epoch=state['epoch'], cohort_pairs_sha256=pair_sha,
        metadata_sha256=sha(args.val_dir / 'metadata.csv'), evaluator_sha256=evaluator_sha,
        implementation_sha256=sha(__file__), device='cpu', precision='float32', training_updates=0,
        pesq_mode='wb', sample_rate=16000, normalization='none', gain_matching=False,
        frontend=dict(n_fft=512, hop_length=160, win_length=320, power=.5, microphones=8, reference=0),
        source_training_objective=state['args']['causal_training_objective'],
        interpretation='Removing a jointly trained branch changes its operating point. Ablation scores do not bound separately retrained model capacity.',
        results=[])
    started = time.perf_counter()
    scores = {'full': [], 'spatial_only': []}
    try:
        with torch.inference_mode():
            for index, (_, snr, mix_path, target_path) in enumerate(records):
                mixture, sample_rate = sf.read(mix_path, dtype='float32', always_2d=True)
                target = target_data(target_path)
                assert sample_rate == 16000 and mixture.shape[1] >= 8 and len(mixture) == len(target)
                features = evaluator.build_stft_batch(torch.from_numpy(mixture[:, :8].copy())[None],
                    device=torch.device('cpu'), n_fft=512, hop_length=160, win_length=320, power=.5)
                full = model(features)
                spatial = spatial_only(model, features)
                row = dict(snr=snr)
                for name, value in (('full', full), ('spatial_only', spatial)):
                    waveform = evaluator.reconstruct_waveform(value, length=len(target), device=torch.device('cpu'),
                        n_fft=512, hop_length=160, win_length=320, power=.5)[0].numpy()
                    assert np.isfinite(waveform).all()
                    score = float(pesq(16000, target, waveform, 'wb'))
                    scores[name].append(score)
                    row[name] = score
                result['results'].append(row)
                if (index + 1) % 8 == 0:
                    print(json.dumps(dict(completed=index + 1, elapsed_seconds=time.perf_counter() - started)), flush=True)
        assert sha(args.checkpoint) == original_sha and sha(evaluator.__file__) == evaluator_sha
        assert all(sha(path) == value for path, value in audio_hashes.items())
        result.update(status='complete', checked_samples=len(records), input_audio_unchanged=True,
            private_audio_hashes_exported=False, mean_pesq={key:float(np.mean(value)) for key, value in scores.items()},
            full_better_count=sum(a > b for a, b in zip(scores['full'], scores['spatial_only'])),
            elapsed_seconds=time.perf_counter() - started)
        print(json.dumps({key:value for key, value in result.items() if key != 'results'}, indent=2), flush=True)
    except Exception as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        (output / 'branches.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()

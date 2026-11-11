"""Summarize paired PESQ changes by requested SNR and free-field geometry."""
import argparse
import cmath
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean


def read_metadata(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def geometry_band(row):
    x, y, z = json.loads(row['array_center'])
    mics = [(x + (i - 4) * .04, y, z) for i in range(8)]
    steering = []
    for field in ('target_pos', 'noise_pos'):
        position = json.loads(row[field])
        distances = [math.dist(mic, position) for mic in mics]
        vector = [cmath.exp(-2j * math.pi * 1000 * d / 343) / d for d in distances]
        norm = math.sqrt(sum(abs(v)**2 for v in vector))
        steering.append([v / norm for v in vector])
    target, noise = steering
    coherence = abs(sum(a.conjugate() * b for a, b in zip(target, noise, strict=True)))
    if coherence < .8:
        return 'below_0.8'
    return '0.8_to_0.95' if coherence < .95 else 'at_least_0.95'


def summarize(metadata, control, variant):
    expected = set(range(600))
    if len(metadata) != 600 or set(control) != expected or set(variant) != expected:
        raise ValueError('Expected both scores for each of the 600 original records')
    if len({row['sample_id'] for row in metadata}) != 600:
        raise ValueError('Duplicate sample IDs in comparison metadata')
    groups = {'all': {'all': list(range(600))}, 'requested_snr_db': defaultdict(list),
              'free_field_1khz_geometry': defaultdict(list), 'geometry_and_snr': defaultdict(list)}
    for index, row in enumerate(metadata):
        snr = float(row['snr_db'])
        geometry = geometry_band(row)
        groups['requested_snr_db'][snr].append(index)
        groups['free_field_1khz_geometry'][geometry].append(index)
        snr_group = 'snr_le_minus2' if snr <= -2 else 'snr_gt_minus2'
        groups['geometry_and_snr'][f'{geometry}:{snr_group}'].append(index)
    if not all(math.isfinite(value) for scores in (control, variant) for value in scores.values()):
        raise ValueError('Expected finite per-sample PESQ scores')
    result = {}
    for dimension, cohorts in groups.items():
        result[dimension] = {}
        for name, indices in cohorts.items():
            deltas = [variant[index] - control[index] for index in indices]
            result[dimension][name] = dict(
                samples=len(indices), control_pesq=fmean(control[index] for index in indices),
                variant_pesq=fmean(variant[index] for index in indices), delta=fmean(deltas),
                improved=sum(d > 0 for d in deltas), worse=sum(d < 0 for d in deltas))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial_summary', type=Path)
    parser.add_argument('comparison_dir', type=Path)
    args = parser.parse_args()
    from light_eval_inputs import load_records, pairs_sha256

    trial = json.loads(args.trial_summary.read_text())
    if trial['status'] != 'matched_full600_complete':
        raise ValueError('The matched trial has not completed both full evaluations')
    metadata_path = args.comparison_dir / 'metadata.csv'
    metadata_hash = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    records = load_records(args.comparison_dir)
    pair_hash = pairs_sha256(records)
    metadata = read_metadata(metadata_path)
    if [int(row['sample_id']) for row in metadata] != [record.sample_id for record in records]:
        raise ValueError('Comparison metadata order changed')
    control, variant = (trial['results'][branch] for branch in ('control', 'beam_cues'))
    if any(control['protocol'][key] != variant['protocol'][key]
           for key in ('original_protected_evaluator_sha256', 'source_checkpoint_sha256', 'inference_precision')):
        raise ValueError('The branches have different source checkpoints or scoring backends')
    for row in (control, variant):
        protocol = row['protocol']
        if (row['samples'] != 600 or protocol['selected_pairs_sha256'] != pair_hash
                or protocol['metadata_sha256'] != metadata_hash or protocol['target_gain_matching']
                or protocol['external_normalization_clipping_or_alignment']):
            raise ValueError('The trial used a different dataset or scoring protocol')
    if control['noisy'] != variant['noisy']:
        raise ValueError('The branches have different mixture baselines')
    grouped = summarize(metadata, *({int(k): v for k, v in row['scores'].items()} for row in (control, variant)))
    if not math.isclose(grouped['all']['all']['delta'], trial['delta'], abs_tol=1e-12):
        raise ValueError('Paired records disagree with the reported overall gain')
    print(json.dumps(dict(source_epoch=trial['source_epoch'], metadata_sha256=metadata_hash,
                          selected_pairs_sha256=pair_hash, groups=grouped), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

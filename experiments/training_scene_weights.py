"""Fixed training weights from original scene metadata."""
import csv
import math
from statistics import fmean

from experiments.analyze_beam_cues_trial import geometry_band


def scene_weight(row):
    rt60 = float(row['rt60'])
    snr = float(row['snr_db'])
    if not math.isfinite(rt60) or not math.isfinite(snr):
        raise ValueError('Expected finite training scene metadata')
    short_room = .05 <= rt60 <= .15
    difficult_scene = snr <= -2 and geometry_band(row) == 'at_least_0.95'
    return 1 + 2 * short_room + 2 * difficult_scene


def training_scene_weights(metadata):
    with metadata.open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    weights = {int(row['sample_id']): scene_weight(row) for row in rows}
    if not weights or len(weights) != len(rows):
        raise ValueError('Expected unique training sample IDs')
    mean_weight = fmean(weights.values())
    return {sample_id: weight / mean_weight for sample_id, weight in weights.items()}

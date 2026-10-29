#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ $# -ne 2 ]]; then
    printf 'Usage: bash run_light_metric.sh QUALITY_CHECKPOINT MATCHING_600_SAMPLE_SUMMARY\n' >&2
    exit 2
fi
train_py="${TRAIN_PY:-$HOME/miniconda3/envs/EaBNet/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
source_checkpoint="$1"
source_summary="$2"
train_dir="/data/ssd1/jinrui.yang/training_set"
val_dir="/data/ssd1/jinrui.yang/newset_rt60_005-015_on_NOISEX92/babble/training_set"
mkdir -p ./logs_cbam_metric_v1
exec 9>./logs_cbam_metric_v1/.launcher.lock
flock -n 9 || { printf 'Another metric launcher is running.\n' >&2; exit 1; }

"$train_py" - "$source_checkpoint" "$source_summary" "$val_dir" <<'PY_PREFLIGHT'
import hashlib
import json
import os
from pathlib import Path
import sys
import torch
from light_candidate_runtime import read_candidate_checkpoint

repository = Path.cwd()
names = {'run_light_quality.sh', 'continue_light_quality.sh', 'run_light_metric.sh',
         'train_light_quality.py', 'train_light_metric.py'}
for process in Path('/proc').glob('[0-9]*'):
    if int(process.name) in (os.getpid(), os.getppid()):
        continue
    try:
        command = (process / 'cmdline').read_bytes().split(b'\0')
        if ((process / 'cwd').resolve() == repository and
                any(Path(value.decode(errors='replace')).name in names for value in command if value)):
            raise RuntimeError(f'Training runner {process.name} in this repository must finish first')
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
checkpoint, _ = read_candidate_checkpoint(sys.argv[1], 'cbam_refiner_v1')
report = json.loads(Path(sys.argv[2]).read_text())
assert checkpoint['epoch'] >= 0 and checkpoint['args']['quality_objective'] == 'spectral_mrstft_waveform_v1'
assert report['status'] == 'complete' and report['completed_samples'] == report['expected_samples'] == 600
assert report['metric_protocol']['target_gain_matching'] is False
assert report['checkpoint_sha256'] == hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest()
assert report['metadata_sha256'] == hashlib.sha256((Path(sys.argv[3]) / 'metadata.csv').read_bytes()).hexdigest()
if report['mean']['enhanced']['pesq'] >= 3.4:
    raise RuntimeError('Source already reaches 3.4; verify it before starting another experiment')
if torch.cuda.device_count() < 2:
    raise RuntimeError('Metric training requires two visible CUDA GPUs')
PY_PREFLIGHT

run_name="metric_$(date +%Y%m%d_%H%M%S)_$$"
checkpoint_dir="./checkpoints_cbam_metric_v1_${run_name}"
best_dir="./bestmodels_cbam_metric_v1_${run_name}"
log_dir="./logs_cbam_metric_v1/${run_name}"
mkdir -p "$log_dir"
printf 'Run: %s\nSource: %s\nCheckpoint directory: %s\nLog directory: %s\n' \
    "$run_name" "$source_checkpoint" "$checkpoint_dir" "$log_dir"

evaluate_epoch() {
    local epoch="$1"
    local result_dir="$log_dir/pesq_selection/model_epoch_${epoch}"
    "$train_py" -u evaluate_light.py \
        --checkpoint "$checkpoint_dir/model_epoch_${epoch}.pt" --candidate cbam_refiner_v1 \
        --val-dir "$val_dir" --device cuda --max-samples 0 --save-samples no \
        --match-estimate-level no --estimate-dir "$result_dir" --save-json "$result_dir/summary.json" \
        2>&1 | tee "$log_dir/evaluation_epoch_${epoch}.log"
}

"$train_py" -u train_light_metric.py --warmstart "$source_checkpoint" \
    --train-dir "$train_dir" --val-dir "$val_dir" \
    --checkpoint-dir "$checkpoint_dir" --best-dir "$best_dir" --log-dir "$log_dir" \
    --num-epochs 0 --parallel-mode none --num-workers 0 --cpu-threads 2 --cpu-thread-scale 1 \
    2>&1 | tee "$log_dir/initialization.log"
evaluate_epoch 0
"$train_py" - "$source_summary" "$log_dir/pesq_selection/model_epoch_0/summary.json" <<'PY_BASELINE'
import json
from pathlib import Path
import sys
old, new = [json.loads(Path(path).read_text()) for path in sys.argv[1:]]
assert old['status'] == new['status'] == 'complete'
assert old['completed_samples'] == new['completed_samples'] == 600
for key in ('frontend', 'frontend_protocol', 'num_mics', 'selected_pairs_sha256', 'metadata_sha256',
            'metric_protocol', 'model_parameters', 'candidate_implementation_sha256'):
    assert old[key] == new[key], key
for kind in ('enhanced', 'noisy'):
    for metric, value in old['mean'][kind].items():
        assert abs(new['mean'][kind][metric] - value) <= 0.001, (kind, metric)
print('Initialized generator matches its source on all 600 samples.', flush=True)
PY_BASELINE

for epoch in {1..6}; do
    "$train_py" -u -m torch.distributed.run --standalone --nproc_per_node=2 \
        train_light_metric.py --warmstart "$source_checkpoint" \
        --train-dir "$train_dir" --val-dir "$val_dir" \
        --checkpoint-dir "$checkpoint_dir" --best-dir "$best_dir" --log-dir "$log_dir" \
        --num-epochs "$epoch" --parallel-mode ddp --batch-size 1 --grad-accum-steps 4 \
        --learning-rate 0.0001 --metric-weight 0.05 --metric-lr 0.0001 --segment-seconds 6 \
        --num-workers 4 --strict-memory no --cpu-threads 2 --cpu-thread-scale 1 \
        --use-amp no --model-amp no --allow-tf32 no --resume yes --resume-reset-lr no \
        --stop-on-non-finite yes --skip-non-finite-batches no --save-every 1 \
        2>&1 | tee -a "$log_dir/training.log"
    evaluate_epoch "$epoch"
    reached="$("$train_py" - "$log_dir/pesq_selection/model_epoch_${epoch}/summary.json" "$source_summary" <<'PY_TARGET'
import json
import math
from pathlib import Path
import sys
report, source = [json.loads(Path(path).read_text()) for path in sys.argv[1:]]
assert report['status'] == 'complete' and report['completed_samples'] == report['expected_samples'] == 600
for key in ('frontend', 'frontend_protocol', 'num_mics', 'selected_pairs_sha256', 'metadata_sha256', 'metric_protocol'):
    assert report[key] == source[key], key
assert report['metric_protocol']['target_gain_matching'] is False
score = report['mean']['enhanced']['pesq']
assert math.isfinite(score)
print(int(score >= 3.4))
PY_TARGET
)"
    if [[ "$reached" == 1 ]]; then
        printf 'Full 600-sample PESQ reached 3.4 at metric epoch %s; stopping for verification.\n' "$epoch"
        break
    fi
done

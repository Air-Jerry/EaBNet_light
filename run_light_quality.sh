#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
train_py="${TRAIN_PY:-$HOME/miniconda3/envs/EaBNet/bin/python}"
train_dir="/data/ssd1/jinrui.yang/training_set"
val_dir="/data/ssd1/jinrui.yang/newset_rt60_005-015_on_NOISEX92/babble/training_set"
source_checkpoint="./checkpoints_cbam_flat_projection64/model_epoch_40.pt"
baseline_summary="./logs_cbam_flat_projection64/restart_lr1e3_20261029_093641_1016691/pesq_selection/model_epoch_40/summary.json"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

evaluate_epoch() {
    local epoch="$1"
    local result_dir="$log_dir/pesq_selection/model_epoch_${epoch}"
    "$train_py" -u evaluate_light.py \
        --checkpoint "$checkpoint_dir/model_epoch_${epoch}.pt" \
        --candidate cbam_refiner_v1 --val-dir "$val_dir" --device cuda \
        --max-samples 0 --save-samples no --match-estimate-level no \
        --estimate-dir "$result_dir" --save-json "$result_dir/summary.json" \
        2>&1 | tee "$log_dir/evaluation_epoch_${epoch}.log"
}

if [[ $# -eq 2 && "$1" == "--evaluate-only" ]]; then
    checkpoint_dir="$2"
    run_name="$(basename -- "$checkpoint_dir")"
    run_name="${run_name#checkpoints_cbam_refiner_v1_}"
    log_dir="./logs_cbam_refiner_v1/${run_name}"
    first_eval_epoch=0
elif [[ $# -eq 0 ]]; then
    if [[ ! -f "$baseline_summary" ]]; then
        printf 'Missing historical baseline summary: %s\n' "$baseline_summary" >&2
        exit 1
    fi
    "$train_py" - "$source_checkpoint" "$train_dir" "$val_dir" <<'PY'
import sys
from pathlib import Path
import torch
import evaluate_light
from light_candidate_runtime import all_model_candidates, read_candidate_checkpoint
from light_quality_model import BASE_CANDIDATE, QUALITY_CANDIDATE
from train_light import EnhancementDataset

checkpoint, _ = read_candidate_checkpoint(sys.argv[1], BASE_CANDIDATE)
if checkpoint.get("epoch") != 40:
    raise ValueError("The quality experiment requires the original epoch-40 checkpoint")
if QUALITY_CANDIDATE not in all_model_candidates():
    raise ValueError("The installed runtime does not support the quality model; pull the current code")
evaluate_light.parse_args(["--candidate", QUALITY_CANDIDATE, "--checkpoint", sys.argv[1],
                           "--val-dir", sys.argv[3]])
for directory in sys.argv[2:]:
    dataset = EnhancementDataset(Path(directory), checkpoint["args"]["sample_rate"],
                                 target_ref_mic=checkpoint["args"]["target_ref_mic"],
                                 num_mics=checkpoint["args"]["num_mics"])
    dataset[0]
    print(f"Dataset preflight: {directory}; samples={len(dataset)}", flush=True)
evaluation_records = evaluate_light.load_records(Path(sys.argv[3]))
training_pairs = [(row.sample_id, row.mixture_path, row.target_path) for row in dataset.records]
evaluation_pairs = [(row.sample_id, row.mixture_path, row.target_path) for row in evaluation_records]
if training_pairs != evaluation_pairs:
    raise ValueError("Training and evaluation resolve the validation metadata to different audio files")
if torch.cuda.device_count() < 2:
    raise RuntimeError("This training command requires two visible CUDA GPUs")
PY
    run_name="quality_$(date +%Y%m%d_%H%M%S)_$$"
    checkpoint_dir="./checkpoints_cbam_refiner_v1_${run_name}"
    best_dir="./bestmodels_cbam_refiner_v1_${run_name}"
    log_dir="./logs_cbam_refiner_v1/${run_name}"
    mkdir -p -- "$log_dir"
    printf 'Run: %s\nCheckpoint directory: %s\nLog directory: %s\n' "$run_name" "$checkpoint_dir" "$log_dir"
    "$train_py" -u train_light_quality.py \
        --warmstart "$source_checkpoint" \
        --train-dir "$train_dir" --val-dir "$val_dir" \
        --checkpoint-dir "$checkpoint_dir" --best-dir "$best_dir" --log-dir "$log_dir" \
        --num-epochs 0 --parallel-mode none --num-workers 0 --allow-tf32 no \
        2>&1 | tee "$log_dir/initialization.log"
    evaluate_epoch 0
    "$train_py" - "$baseline_summary" "$log_dir/pesq_selection/model_epoch_0/summary.json" "$source_checkpoint" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

baseline, initialized = [json.loads(Path(name).read_text(encoding="utf-8")) for name in sys.argv[1:3]]
source_hash = hashlib.sha256(Path(sys.argv[3]).read_bytes()).hexdigest()
if baseline.get("checkpoint_sha256") != source_hash:
    raise ValueError("Historical baseline does not describe the original epoch-40 checkpoint")
if baseline.get("status") != "complete" or initialized.get("status") != "complete":
    raise ValueError("Baseline verification requires complete evaluation reports")
if initialized.get("completed_samples") != 600:
    raise ValueError("Quality baseline must evaluate all 600 comparison samples")
for field in ("frontend", "num_mics", "selected_pairs_sha256", "metadata_sha256", "metric_protocol",
              "expected_samples", "completed_samples"):
    if baseline.get(field) != initialized.get(field):
        raise ValueError(f"Quality baseline comparison differs in {field}")
for kind in ("enhanced", "noisy"):
    for metric, old_value in baseline["mean"][kind].items():
        new_value = initialized["mean"][kind][metric]
        if abs(new_value - old_value) > 0.001:
            raise ValueError(f"Initial quality model changed baseline {kind}.{metric}: {old_value} -> {new_value}")
print(f"Baseline verified on 600 samples: PESQ={initialized['mean']['enhanced']['pesq']:.6f}", flush=True)
PY
    for epoch in {1..10}; do
        "$train_py" -u -m torch.distributed.run --standalone --nproc_per_node=2 \
            train_light_quality.py \
            --warmstart "$source_checkpoint" \
            --train-dir "$train_dir" --val-dir "$val_dir" \
            --checkpoint-dir "$checkpoint_dir" --best-dir "$best_dir" --log-dir "$log_dir" \
            --num-epochs "$epoch" --batch-size 1 --grad-accum-steps 4 \
            --segment-seconds 6 --learning-rate 0.0003 \
            --train-lr-patience 3 --train-lr-factor 0.5 --lr-reduce-metric val \
            --parallel-mode ddp --num-workers 4 --strict-memory no \
            --use-amp no --model-amp no --allow-tf32 no \
            --stop-on-non-finite yes --skip-non-finite-batches no --save-every 1 \
            2>&1 | tee -a "$log_dir/training.log"
        evaluate_epoch "$epoch"
    done
    exit 0
else
    printf 'Usage: bash run_light_quality.sh [--evaluate-only QUALITY_CHECKPOINT_DIRECTORY]\n' >&2
    exit 2
fi

mkdir -p -- "$log_dir"
for epoch in {0..10}; do
    if [[ ! -f "$checkpoint_dir/model_epoch_${epoch}.pt" ]]; then
        printf 'Missing checkpoint: %s/model_epoch_%s.pt\n' "$checkpoint_dir" "$epoch" >&2
        exit 1
    fi
done
for ((epoch=first_eval_epoch; epoch<=10; epoch++)); do
    evaluate_epoch "$epoch"
done
printf 'Finished. Evaluation summaries: %s/pesq_selection/model_epoch_*/summary.json\n' "$log_dir"

#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
train_py="${TRAIN_PY:-$HOME/miniconda3/envs/EaBNet/bin/python}"
train_dir="/data/ssd1/jinrui.yang/training_set"
val_dir="/data/ssd1/jinrui.yang/newset_rt60_005-015_on_NOISEX92/babble/training_set"
source_checkpoint="./checkpoints_cbam_flat_projection64/model_epoch_40.pt"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

for required in "$source_checkpoint" "$train_dir/metadata.csv" "$val_dir/metadata.csv"; do
    if [[ ! -f "$required" ]]; then
        printf 'Missing required input: %s\n' "$required" >&2
        exit 1
    fi
done

"$train_py" - "$source_checkpoint" <<'PY'
import sys
import torch
from light_candidate_runtime import read_candidate_checkpoint

checkpoint, _ = read_candidate_checkpoint(sys.argv[1], "cbam_flat_projection64")
if checkpoint.get("epoch") != 40:
    raise ValueError("This five-epoch experiment requires the original epoch-40 checkpoint")
if not checkpoint.get("optimizer_state_dict"):
    raise ValueError("The source checkpoint must include the original Adam state")
if torch.cuda.device_count() < 2:
    raise RuntimeError("This experiment requires two visible CUDA GPUs")
PY

run_name="restart_lr1e3_$(date +%Y%m%d_%H%M%S)_$$"
checkpoint_dir="./checkpoints_cbam_flat_projection64_${run_name}"
best_dir="./bestmodels_cbam_flat_projection64_${run_name}"
log_dir="./logs_cbam_flat_projection64/${run_name}"
mkdir -- "$checkpoint_dir" "$best_dir"
mkdir -p -- "$log_dir"
cp -- "$source_checkpoint" "$checkpoint_dir/model_epoch_40.pt"
cp -- "$source_checkpoint" "$checkpoint_dir/checkpoint_latest.pt"

printf 'Run: %s\nSource: %s\nTraining log: %s/training.log\n' \
    "$run_name" "$source_checkpoint" "$log_dir"

"$train_py" -u -m torch.distributed.run --standalone --nproc_per_node=2 \
    train_light_candidate.py \
    --candidate cbam_flat_projection64 \
    --train-dir "$train_dir" \
    --val-dir "$val_dir" \
    --checkpoint-dir "$checkpoint_dir" \
    --best-dir "$best_dir" \
    --log-dir "$log_dir" \
    --num-epochs 45 \
    --resume yes \
    --resume-reset-lr yes \
    --learning-rate 0.001 \
    --train-lr-reduce-on-plateau no \
    --parallel-mode ddp \
    --batch-size 4 \
    --grad-accum-steps 1 \
    --num-workers 4 \
    --segment-seconds 6 \
    --sample-rate 16000 \
    --n-fft 512 \
    --hop-length 160 \
    --win-length 320 \
    --power 0.5 \
    --target-ref-mic 0 \
    --num-mics 8 \
    --use-amp yes \
    --model-amp yes \
    --allow-tf32 yes \
    --stop-on-non-finite yes \
    --save-every 1 \
    --strict-memory no \
    2>&1 | tee "$log_dir/training.log"

# Compare every saved epoch with the untouched epoch-40 baseline.
# The validation set differs from the old run, so do not rely on inherited best_val_loss.
"$train_py" -u evaluate_light.py \
    --candidate cbam_flat_projection64 \
    --checkpoint-dir "$checkpoint_dir" \
    --val-dir "$val_dir" \
    --device cuda \
    --max-samples 0 \
    --save-samples no \
    --match-estimate-level no \
    --estimate-dir "$log_dir/pesq_selection" \
    2>&1 | tee "$log_dir/evaluation.log"

printf 'Finished. Comparison: %s/pesq_selection/checkpoint_ranking.json\n' "$log_dir"

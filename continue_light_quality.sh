#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ $# -ne 2 || ! "$2" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Usage: bash continue_light_quality.sh QUALITY_CHECKPOINT_DIRECTORY FINAL_EPOCH\n' >&2
    exit 2
fi
train_py="${TRAIN_PY:-$HOME/miniconda3/envs/EaBNet/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
checkpoint_dir="$1"
final_epoch="$2"

plan="$("$train_py" - "$checkpoint_dir" "$final_epoch" <<'PY_PLAN'
import hashlib
import json
from pathlib import Path
import sys

from light_candidate_runtime import read_candidate_checkpoint
from light_quality_model import QUALITY_CANDIDATE

directory = Path(sys.argv[1]).resolve()
repository = Path.cwd()
prefix = "checkpoints_cbam_refiner_v1_quality_"
if directory.parent != repository or not directory.name.startswith(prefix):
    raise ValueError("Use this repository's checkpoints_cbam_refiner_v1_quality_* directory")
checkpoint, _ = read_candidate_checkpoint(directory / "checkpoint_latest.pt", QUALITY_CANDIDATE)
saved = checkpoint["args"]
epoch = checkpoint["epoch"]
if type(epoch) is not int or not 0 < epoch < int(sys.argv[2]):
    raise ValueError("FINAL_EPOCH must exceed the saved positive epoch; refusing to repeat completed training")
run_name = directory.name.removeprefix("checkpoints_cbam_refiner_v1_")
expected = {
    "checkpoint_dir": directory,
    "best_dir": repository / f"bestmodels_cbam_refiner_v1_{run_name}",
    "log_dir": repository / "logs_cbam_refiner_v1" / run_name,
}
for name, path in expected.items():
    if Path(saved[name]).resolve() != path or not path.is_dir():
        raise ValueError(f"Saved {name} does not belong to this existing run")
values = [str(directory), str(expected["best_dir"]), str(expected["log_dir"]),
          str(saved["quality_source_checkpoint"]), str(saved["train_dir"]),
          str(saved["val_dir"]), str(epoch + 1)]
for name in ("seed", "learning_rate", "weight_decay", "grad_clip", "segment_seconds",
             "train_random_crop", "train_lr_reduce_on_plateau", "train_lr_patience",
             "train_lr_factor", "lr_reduce_metric", "train_lr_min_delta", "min_learning_rate",
             "batch_size", "grad_accum_steps", "num_workers", "strict_memory", "pin_memory",
             "persistent_workers", "prefetch_factor", "cudnn_benchmark", "cpu_threads",
             "interop_threads", "use_amp", "model_amp", "allow_tf32"):
    values.extend(("--" + name.replace("_", "-"), str(saved[name])))
values.extend(("--cpu-thread-scale", "1.0"))
if any("\n" in value or "\r" in value for value in values):
    raise ValueError("Checkpoint paths and settings must not contain newlines")
summary = json.loads((expected["log_dir"] / f"pesq_selection/model_epoch_{epoch}/summary.json").read_text())
baseline = json.loads((expected["log_dir"] / "pesq_selection/model_epoch_0/summary.json").read_text())
assert summary["status"] == "complete" and summary["completed_samples"] == summary["expected_samples"] == 600
for key in ("frontend", "frontend_protocol", "num_mics", "selected_pairs_sha256", "metadata_sha256", "metric_protocol"):
    assert summary[key] == baseline[key], key
assert summary["metric_protocol"]["target_gain_matching"] is False
assert summary["checkpoint_sha256"] == hashlib.sha256((directory / f"model_epoch_{epoch}.pt").read_bytes()).hexdigest()
assert baseline["metadata_sha256"] == hashlib.sha256((Path(saved["val_dir"]) / "metadata.csv").read_bytes()).hexdigest()
for value in values:
    print(value)
PY_PLAN
)"
mapfile -t saved_values <<< "$plan"
checkpoint_dir="${saved_values[0]}"
best_dir="${saved_values[1]}"
log_dir="${saved_values[2]}"
source_checkpoint="${saved_values[3]}"
train_dir="${saved_values[4]}"
val_dir="${saved_values[5]}"
first_epoch="${saved_values[6]}"
training_args=("${saved_values[@]:7}")
exec 9>"$checkpoint_dir/.quality_continuation.lock"
flock -n 9 || { printf 'This quality continuation is already running.\n' >&2; exit 1; }

for ((epoch=first_epoch; epoch<=final_epoch; epoch++)); do
    "$train_py" -u -m torch.distributed.run --standalone --nproc_per_node=2 \
        train_light_quality.py --warmstart "$source_checkpoint" \
        --train-dir "$train_dir" --val-dir "$val_dir" \
        --checkpoint-dir "$checkpoint_dir" --best-dir "$best_dir" --log-dir "$log_dir" \
        --num-epochs "$epoch" --parallel-mode ddp --resume yes --resume-reset-lr no \
        --stop-on-non-finite yes --skip-non-finite-batches no --save-every 1 \
        "${training_args[@]}" 2>&1 | tee -a "$log_dir/training.log"
    result_dir="$log_dir/pesq_selection/model_epoch_${epoch}"
    "$train_py" -u evaluate_light.py \
        --checkpoint "$checkpoint_dir/model_epoch_${epoch}.pt" --candidate cbam_refiner_v1 \
        --val-dir "$val_dir" --device cuda --max-samples 0 --save-samples no \
        --match-estimate-level no --estimate-dir "$result_dir" --save-json "$result_dir/summary.json" \
        2>&1 | tee "$log_dir/evaluation_epoch_${epoch}.log"
    reached="$("$train_py" - "$result_dir/summary.json" "$log_dir/pesq_selection/model_epoch_0/summary.json" <<'PY_RESULT'
import json
import math
from pathlib import Path
import sys
report, baseline = [json.loads(Path(path).read_text()) for path in sys.argv[1:]]
assert report["status"] == "complete" and report["completed_samples"] == report["expected_samples"] == 600
for key in ("frontend", "frontend_protocol", "num_mics", "selected_pairs_sha256", "metadata_sha256", "metric_protocol"):
    assert report[key] == baseline[key], key
assert report["metric_protocol"]["target_gain_matching"] is False
score = report["mean"]["enhanced"]["pesq"]
assert math.isfinite(score)
print(int(score >= 3.4))
PY_RESULT
)"
    if [[ "$reached" == 1 ]]; then
        printf 'Full 600-sample PESQ reached 3.4 at epoch %s; stopping for verification.\n' "$epoch"
        break
    fi
done

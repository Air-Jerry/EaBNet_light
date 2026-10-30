#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ $# -ne 1 ]]; then
    printf 'Usage: bash run_light_causal.sh VERIFIED_PREFLIGHT_DIRECTORY\n' >&2
    exit 2
fi
train_py="${TRAIN_PY:-$HOME/miniconda3/envs/EaBNet/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
preflight="$1"
train_dir="/data/ssd1/jinrui.yang/training_set"
val_dir="/data/ssd1/jinrui.yang/newset_rt60_005-015_on_NOISEX92/babble/training_set"
mkdir -p logs_causal_spatial_v1
exec 9>logs_causal_spatial_v1/.launcher.lock
flock -n 9 || { printf 'Another causal launcher is running.\n' >&2; exit 1; }
"$train_py" - "$preflight" "$val_dir" <<'PY_PREFLIGHT'
import hashlib,json,os,sys
from pathlib import Path
import torch
from light_candidate_runtime import read_candidate_checkpoint
root=Path(sys.argv[1]);check=json.loads((root/'cuda_updates_final/cuda_check.json').read_text())
bench=json.loads((root/'streaming_final/streaming_benchmark.json').read_text())
assert check['status']=='passed' and check['generator_adam_step']==8
assert check['all_weights_and_adam_finite'] and check['noncausal_frozen_backbones']==0
p=root/'cuda_updates_final/checkpoints/model_epoch_2.pt'
assert hashlib.sha256(p.read_bytes()).hexdigest()==check['checkpoint_sha256']==bench['checkpoint_sha256']
state,_=read_candidate_checkpoint(p,'causal_spatial_v1')
for name,value in state['args']['causal_training_implementation_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==value,name
assert bench['status']=='measured' and bench['future_perturbation_max_error']<1e-6
assert bench['warmup_state_discarded'] and bench['mean_rtf_below_one'] and bench['p99_below_10ms']
assert bench['all_hops_below_10ms'],'Investigate startup/live deadlines before full training'
assert bench['full_vs_stream_waveform_max_error']<1e-5
assert hashlib.sha256((Path(sys.argv[2])/'metadata.csv').read_bytes()).hexdigest()=='8d0b4d229ebc42cd9892e6d0e0d36416007ded01fbc0b753df4bf7e6b734eb02'
names={'run_light_quality.sh','continue_light_quality.sh','run_light_metric.sh','run_light_spatial.sh',
       'run_light_causal.sh','train_light_quality.py','train_light_metric.py','train_light_spatial.py',
       'train_light_causal.py','check_light_causal_cuda.py','benchmark_light_causal.py'}
for process in Path('/proc').glob('[0-9]*'):
    if int(process.name) in (os.getpid(),os.getppid()):continue
    try:
        if (process/'cwd').resolve()==Path.cwd() and any(Path(x.decode(errors='replace')).name in names
                for x in (process/'cmdline').read_bytes().split(b'\0') if x):
            raise RuntimeError(f'Existing job {process.name} must finish first')
    except (FileNotFoundError,PermissionError,ProcessLookupError):pass
assert torch.cuda.device_count()==2 and torch.cuda.is_bf16_supported()
assert all(torch.cuda.mem_get_info(i)[0]>10*2**30 for i in range(2))
print('Real causal streaming and dual-CUDA preflight passed.',flush=True)
PY_PREFLIGHT
run="causal_$(date +%Y%m%d_%H%M%S)_$$"
checkpoints="./checkpoints_causal_spatial_v1_${run}"
best="./bestmodels_causal_spatial_v1_${run}"
logs="./logs_causal_spatial_v1/${run}"
mkdir -p "$logs"
printf 'Run: %s\nCheckpoints: %s\nLogs: %s\n' "$run" "$checkpoints" "$logs"
cp "$preflight/cuda_updates_final/cuda_check.json" "$logs/cuda_preflight.json"
cp "$preflight/streaming_final/streaming_benchmark.json" "$logs/streaming_preflight.json"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv,noheader,nounits -l 10 >> "$logs/gpu_utilization.log" &
sampler=$!
trap 'kill "$sampler" 2>/dev/null || true' EXIT

evaluate_epoch() {
    local epoch="$1"
    local result="$logs/pesq_selection/model_epoch_${epoch}"
    "$train_py" -u evaluate_light.py --checkpoint "$checkpoints/model_epoch_${epoch}.pt" \
        --candidate causal_spatial_v1 --val-dir "$val_dir" --device cuda \
        --max-samples 0 --save-samples no --match-estimate-level no \
        --estimate-dir "$result" --save-json "$result/summary.json" \
        2>&1 | tee "$logs/evaluation_epoch_${epoch}.log"
}

check_results() {
    "$train_py" - "$logs" <<'PY_RESULTS'
import json,math,sys
from pathlib import Path
root=Path(sys.argv[1])/'pesq_selection';baseline=json.loads((root/'model_epoch_0/summary.json').read_text())
old=json.loads(Path('logs_cbam_metric_v1/metric_20261030_052113_1573812/pesq_selection/model_epoch_0/summary.json').read_text())
keys=('frontend','frontend_protocol','num_mics','selected_pairs_sha256','metadata_sha256','metric_protocol')
for key in keys:assert baseline[key]==old[key],key
for name,value in old['mean']['noisy'].items():assert abs(baseline['mean']['noisy'][name]-value)<1e-10,name
for name,value in baseline['mean']['noisy'].items():assert abs(baseline['mean']['enhanced'][name]-value)<1e-3,name
rows=[]
for p in root.glob('model_epoch_*/summary.json'):
    r=json.loads(p.read_text());epoch=int(p.parent.name.rsplit('_',1)[1])
    assert r['status']=='complete' and r['completed_samples']==r['expected_samples']==600
    assert r['structure_candidate']=='causal_spatial_v1'
    assert r['candidate_evidence']['model_causal'] is True and r['candidate_evidence']['future_frames']==0
    assert r['candidate_implementation_sha256']==baseline['candidate_implementation_sha256']
    for key in keys:assert r[key]==baseline[key],key
    assert r['metric_protocol']['target_gain_matching'] is False
    for name,value in baseline['mean']['noisy'].items():assert abs(r['mean']['noisy'][name]-value)<1e-10,name
    score=r['mean']['enhanced']['pesq'];assert math.isfinite(score)
    assert all(math.isfinite(v) for metrics in r['mean'].values() for v in metrics.values())
    rows.append(dict(epoch=epoch,pesq=score,checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256']))
rows.sort(key=lambda r:r['epoch']);best=max(rows,key=lambda r:r['pesq'])
result=dict(status='target_reached_pending_streaming_audit' if best['pesq']>=3.4 else ('complete' if rows[-1]['epoch']==30 else 'partial'),
    selection_scope='Same full600 comparison samples; not an independent final test',
    model_is_causal=True,results=rows,best_completed_epoch=best['epoch'],best_completed_pesq=best['pesq'],
    target_reached=best['pesq']>=3.4,final_full600_waveform_streaming_audit_required=True)
(root/'checkpoint_ranking.json').write_text(json.dumps(result,indent=2))
print(int(result['target_reached']))
PY_RESULTS
}

"$train_py" -u train_light_causal.py --train-dir "$train_dir" --val-dir "$val_dir" \
    --checkpoint-dir "$checkpoints" --best-dir "$best" --log-dir "$logs" \
    --num-epochs 0 --parallel-mode none --num-workers 0 --cpu-threads 2 \
    2>&1 | tee "$logs/initialization.log"
evaluate_epoch 0
check_results >/dev/null
for epoch in {1..30}; do
    printf 'phase=train epoch=%s\n' "$epoch" >> "$logs/gpu_utilization.log"
    "$train_py" -u -m torch.distributed.run --standalone --nproc_per_node=2 \
        train_light_causal.py --train-dir "$train_dir" --val-dir "$val_dir" \
        --checkpoint-dir "$checkpoints" --best-dir "$best" --log-dir "$logs" \
        --num-epochs "$epoch" --parallel-mode ddp --batch-size 1 --grad-accum-steps 4 \
        --segment-seconds 6 --num-workers 4 --strict-memory no --cpu-threads 2 --cpu-thread-scale 1 \
        --learning-rate 0.001 --grad-clip 5 --use-amp no --model-amp no --allow-tf32 no \
        --resume yes --resume-reset-lr no --stop-on-non-finite yes --skip-non-finite-batches no \
        --save-every 1 2>&1 | tee -a "$logs/training.log"
    printf 'phase=evaluation epoch=%s\n' "$epoch" >> "$logs/gpu_utilization.log"
    evaluate_epoch "$epoch"
    if [[ "$(check_results)" == 1 ]]; then
        printf 'Full600 PESQ reached 3.4; stop for final waveform-streaming verification.\n'
        break
    fi
done

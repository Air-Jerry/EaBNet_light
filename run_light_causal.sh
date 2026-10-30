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
candidate="${CAUSAL_MODEL_CANDIDATE:-causal_unet_spatial_v1}"
case "$candidate" in
    causal_unet_spatial_v1) entry=train_light_causal_fast.py; run_prefix=causal_fast ;;
    causal_recurrent_spatial_v1) entry=train_light_causal_recurrent.py; run_prefix=causal_recurrent ;;
    *) printf 'Unsupported cold-start causal candidate: %s\n' "$candidate" >&2; exit 2 ;;
esac
train_dir="/data/ssd1/jinrui.yang/training_set"
val_dir="/data/ssd1/jinrui.yang/newset_rt60_005-015_on_NOISEX92/babble/training_set"
mkdir -p "logs_${candidate}"
exec 9>"logs_${candidate}/.launcher.lock"
flock -n 9 || { printf 'Another causal launcher is running.\n' >&2; exit 1; }
"$train_py" - "$preflight" "$val_dir" "$candidate" <<'PY_PREFLIGHT'
import hashlib,json,os,re,sys
from pathlib import Path
import torch
from light_candidate_runtime import read_candidate_checkpoint
root=Path(sys.argv[1]);check=json.loads((root/'cuda_final/cuda_check.json').read_text())
bench=json.loads((root/'streaming_final/streaming_benchmark.json').read_text())
assert check['status']=='passed' and check['generator_adam_step']==64
assert check['all_weights_and_adam_finite'] and check['noncausal_frozen_backbones']==0
assert check['effective_global_batch']==8 and check['batch_size_per_gpu']*check['grad_accum_steps']==4
p=root/'cuda_final/checkpoints/model_epoch_2.pt'
assert hashlib.sha256(p.read_bytes()).hexdigest()==check['checkpoint_sha256']==bench['checkpoint_sha256']
state,_=read_candidate_checkpoint(p,sys.argv[3])
for name,value in state['args']['causal_training_implementation_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==value,name
assert bench['status']=='measured' and bench['future_perturbation_max_error']<1e-6
assert bench['warmup_state_discarded'] and bench['mean_rtf_below_one'] and bench['p99_below_10ms']
assert bench['all_hops_below_10ms'],'Investigate startup/live deadlines before full training'
assert bench['full_vs_stream_waveform_max_error']<1e-5
speeds=re.findall(r'\[PERF\]\[train\].*?samples/s\(local/global\)=[\d.]+/([\d.]+)',(root/'cuda_final/training.log').read_text())
assert speeds and float(speeds[-1])>=20,'Hot global throughput is insufficient for the one-hour budget'
assert hashlib.sha256((Path(sys.argv[2])/'metadata.csv').read_bytes()).hexdigest()=='8d0b4d229ebc42cd9892e6d0e0d36416007ded01fbc0b753df4bf7e6b734eb02'
names={'run_light_quality.sh','continue_light_quality.sh','run_light_metric.sh','run_light_spatial.sh',
       'run_light_causal.sh','train_light_quality.py','train_light_metric.py','train_light_spatial.py',
       'train_light_causal.py','train_light_causal_fast.py','check_light_causal_cuda.py','benchmark_light_causal.py',
       'run_light_causal_global.sh','train_light_causal_global.py','train_light_causal_recurrent.py'}
ancestors=set();cursor=os.getpid()
while cursor>1:
    ancestors.add(cursor)
    try:cursor=int(Path(f'/proc/{cursor}/stat').read_text().rsplit(')',1)[1].split()[1])
    except FileNotFoundError:break
for process in Path('/proc').glob('[0-9]*'):
    if int(process.name) in ancestors:continue
    try:
        if (process/'cwd').resolve()==Path.cwd() and any(Path(x.decode(errors='replace')).name in names
                for x in (process/'cmdline').read_bytes().split(b'\0') if x):
            raise RuntimeError(f'Existing job {process.name} must finish first')
    except (FileNotFoundError,PermissionError,ProcessLookupError):pass
assert torch.cuda.device_count()==2 and torch.cuda.is_bf16_supported()
assert all(torch.cuda.mem_get_info(i)[0]>10*2**30 for i in range(2))
print('Real causal streaming and dual-CUDA preflight passed.',flush=True)
PY_PREFLIGHT
read -r batch_size grad_accum_steps < <("$train_py" - "$preflight/cuda_final/cuda_check.json" <<'PY_BATCH'
import json,sys
check=json.load(open(sys.argv[1]))
print(check['batch_size_per_gpu'],check['grad_accum_steps'])
PY_BATCH
)
run="${run_prefix}_$(date +%Y%m%d_%H%M%S)_$$"
checkpoints="./checkpoints_${candidate}_${run}"
best="./bestmodels_${candidate}_${run}"
logs="./logs_${candidate}/${run}"
mkdir -p "$logs"
printf 'Run: %s\nCheckpoints: %s\nLogs: %s\n' "$run" "$checkpoints" "$logs"
cp "$preflight/cuda_final/cuda_check.json" "$logs/cuda_preflight.json"
cp "$preflight/streaming_final/streaming_benchmark.json" "$logs/streaming_preflight.json"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv,noheader,nounits -l 10 >> "$logs/gpu_utilization.log" &
sampler=$!
phase=initialization
finish() {
    local code=$?
    kill "$sampler" 2>/dev/null || true
    "$train_py" - "$logs" "$phase" "$code" <<'PY_STATUS'
import json,sys,time
from pathlib import Path
Path(sys.argv[1],'run_status.json').write_text(json.dumps(dict(
    status='complete' if int(sys.argv[3])==0 else 'failed',
    final_phase=sys.argv[2],exit_code=int(sys.argv[3]),finished_unix=time.time()),indent=2))
PY_STATUS
    exit "$code"
}
trap finish EXIT

evaluate_epoch() {
    local epoch="$1"
    local result="$logs/pesq_selection/model_epoch_${epoch}"
    timeout --kill-after=15s "${2:-3600}s" "$train_py" -u evaluate_light.py --checkpoint "$checkpoints/model_epoch_${epoch}.pt" \
        --candidate "$candidate" --val-dir "$val_dir" --device cuda \
        --max-samples 0 --save-samples no --match-estimate-level no \
        --estimate-dir "$result" --save-json "$result/summary.json" \
        2>&1 | tee "$logs/evaluation_epoch_${epoch}.log"
}

check_results() {
    "$train_py" - "$logs" "$candidate" <<'PY_RESULTS'
import json,math,sys
from pathlib import Path
root=Path(sys.argv[1])/'pesq_selection';baseline=json.loads((root/'model_epoch_0/summary.json').read_text())
candidate=sys.argv[2] if len(sys.argv)>2 else 'causal_unet_spatial_v1'
old=json.loads(Path('logs_cbam_metric_v1/metric_20261030_052113_1573812/pesq_selection/model_epoch_0/summary.json').read_text())
keys=('frontend','frontend_protocol','num_mics','selected_pairs_sha256','metadata_sha256','metric_protocol')
for key in keys:assert baseline[key]==old[key],key
for name,value in old['mean']['noisy'].items():assert abs(baseline['mean']['noisy'][name]-value)<1e-10,name
for name,value in baseline['mean']['noisy'].items():assert abs(baseline['mean']['enhanced'][name]-value)<1e-3,name
rows=[]
for p in root.glob('model_epoch_*/summary.json'):
    r=json.loads(p.read_text());epoch=int(p.parent.name.rsplit('_',1)[1])
    assert r['status']=='complete' and r['completed_samples']==r['expected_samples']==600
    assert r['structure_candidate']==candidate
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

"$train_py" -u "$entry" --train-dir "$train_dir" --val-dir "$val_dir" \
    --checkpoint-dir "$checkpoints" --best-dir "$best" --log-dir "$logs" \
    --num-epochs 0 --parallel-mode none --num-workers 0 --cpu-threads 2 \
    2>&1 | tee "$logs/initialization.log"
baseline_start=$(date +%s)
phase=baseline_full600
evaluate_epoch 0
evaluation_reserve=$(( $(date +%s) - baseline_start + 60 ))
(( evaluation_reserve < 1200 )) || { printf 'Full600 evaluation needs too much of the one-hour budget.\n' >&2; exit 1; }
check_results >/dev/null
for epoch in {1..30}; do
    phase="training_epoch_${epoch}"
    epoch_start=$(date +%s)
    train_budget=$(( 3600 - evaluation_reserve ))
    printf 'phase=train epoch=%s\n' "$epoch" >> "$logs/gpu_utilization.log"
    timeout --kill-after=15s "${train_budget}s" "$train_py" -u -m torch.distributed.run --standalone --nproc_per_node=2 \
        "$entry" --train-dir "$train_dir" --val-dir "$val_dir" \
        --checkpoint-dir "$checkpoints" --best-dir "$best" --log-dir "$logs" \
        --num-epochs "$epoch" --parallel-mode ddp --batch-size "$batch_size" --grad-accum-steps "$grad_accum_steps" \
        --segment-seconds 6 --num-workers 4 --strict-memory no --cpu-threads 2 --cpu-thread-scale 1 \
        --learning-rate 0.001 --grad-clip 5 --use-amp no --model-amp no --allow-tf32 no \
        --resume yes --resume-reset-lr no --stop-on-non-finite yes --skip-non-finite-batches no \
        --save-every 1 2>&1 | tee -a "$logs/training.log"
    printf 'phase=evaluation epoch=%s\n' "$epoch" >> "$logs/gpu_utilization.log"
    phase="full600_epoch_${epoch}"
    evaluation_start=$(date +%s)
    remaining=$(( 3600 - evaluation_start + epoch_start - 10 ))
    (( remaining > 0 )) || { printf 'Epoch exceeded one hour before full600 evaluation.\n' >&2; exit 1; }
    evaluate_epoch "$epoch" "$remaining"
    reached=$(check_results)
    "$train_py" - "$logs" "$checkpoints" "$epoch" "$epoch_start" "$evaluation_start" <<'PY_TIMING'
import json,re,sys,time
from pathlib import Path
import torch
root=Path(sys.argv[1]);epoch=int(sys.argv[3]);start=int(sys.argv[4]);evaluation=int(sys.argv[5])
state=torch.load(Path(sys.argv[2])/f'model_epoch_{epoch}.pt',map_location='cpu',weights_only=False)
assert state['epoch']==epoch
assert {int(v['step']) for v in state['optimizer_state_dict']['state'].values()}=={epoch*6700}
text=(root/'training.log').read_text()
for phase,count in [('train',53600),('val',600)]:
    perf=re.findall(r'\[PERF\]\['+phase+r'\] samples\(local/global\)=\d+/(\d+).*?non_finite_skips\(local\)=(\d+)',text)
    assert perf and perf[-1]==(str(count),'0'),(phase,perf[-1:])
end=time.time();seconds=end-start
row=dict(epoch=epoch,status='complete' if seconds<=3600 else 'exceeded_one_hour',
    started_unix=start,finished_unix=end,total_seconds=seconds,
    training_validation_checkpoint_seconds=evaluation-start,evaluation_audit_seconds=end-evaluation,
    training_samples=53600,validation_samples=600,full_metric_samples=600,
    optimizer_updates_this_epoch=6700,global_batch_size=8,limit_seconds=3600)
with (root/'epoch_timing.jsonl').open('a') as handle:handle.write(json.dumps(row)+'\n')
(root/'latest_epoch_timing.json').write_text(json.dumps(row,indent=2))
print('Complete epoch timing:',json.dumps(row),flush=True)
assert seconds<=3600,'The complete epoch exceeded one hour; stop for optimization'
PY_TIMING
    if [[ "$reached" == 1 ]]; then
        printf 'Full600 PESQ reached 3.4; stop for final waveform-streaming verification.\n'
        break
    fi
done
phase=complete

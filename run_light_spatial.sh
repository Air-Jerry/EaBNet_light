#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ $# -ne 2 ]]; then
    printf 'Usage: bash run_light_spatial.sh QUALITY_CHECKPOINT MATCHING_600_SAMPLE_SUMMARY\n' >&2
    exit 2
fi
train_py="${TRAIN_PY:-$HOME/miniconda3/envs/EaBNet/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
source_checkpoint="$1"
source_summary="$2"
train_dir="/data/ssd1/jinrui.yang/training_set"
val_dir="/data/ssd1/jinrui.yang/newset_rt60_005-015_on_NOISEX92/babble/training_set"
mkdir -p logs_cbam_spatial_refiner_v1
exec 9>logs_cbam_spatial_refiner_v1/.launcher.lock
flock -n 9 || { printf 'Another spatial launcher is running.\n' >&2; exit 1; }
"$train_py" - "$source_checkpoint" "$source_summary" "$val_dir" <<'PY_PREFLIGHT'
import hashlib,json,os,sys
from pathlib import Path
import torch
from light_candidate_runtime import read_candidate_checkpoint
names={'run_light_quality.sh','continue_light_quality.sh','run_light_metric.sh','run_light_spatial.sh',
       'train_light_quality.py','train_light_metric.py','train_light_spatial.py','benchmark_light_metric.py'}
for p in Path('/proc').glob('[0-9]*'):
    if int(p.name) in (os.getpid(),os.getppid()):continue
    try:
        if (p/'cwd').resolve()==Path.cwd() and any(Path(x.decode(errors='replace')).name in names
                for x in (p/'cmdline').read_bytes().split(b'\0') if x):
            raise RuntimeError(f'Existing GPU job {p.name} must finish first')
    except (FileNotFoundError,PermissionError,ProcessLookupError):pass
state,_=read_candidate_checkpoint(sys.argv[1],'cbam_refiner_v1')
report=json.loads(Path(sys.argv[2]).read_text())
assert report['status']=='complete' and report['expected_samples']==report['completed_samples']==600
assert report['metric_protocol']['target_gain_matching'] is False
assert report['checkpoint_sha256']==hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest()
assert report['metadata_sha256']==hashlib.sha256((Path(sys.argv[3])/'metadata.csv').read_bytes()).hexdigest()
assert report['mean']['enhanced']['pesq']<3.4,'Source already reaches the target'
assert torch.cuda.device_count()==2 and torch.cuda.is_bf16_supported()
for index in range(2):
    assert torch.cuda.mem_get_info(index)[0]>10*2**30,'Both GPUs must be free before the new experiment'
PY_PREFLIGHT
run="spatial_$(date +%Y%m%d_%H%M%S)_$$"
checkpoints="./checkpoints_cbam_spatial_refiner_v1_${run}"
best="./bestmodels_cbam_spatial_refiner_v1_${run}"
logs="./logs_cbam_spatial_refiner_v1/${run}"
mkdir -p "$logs"
printf 'Run: %s\nSource: %s\nCheckpoints: %s\nLogs: %s\n' "$run" "$source_checkpoint" "$checkpoints" "$logs"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv,noheader,nounits -l 10 >> "$logs/gpu_utilization.log" &
sampler=$!
trap 'kill "$sampler" 2>/dev/null || true' EXIT

evaluate_epoch() {
    local epoch="$1"
    local result="$logs/pesq_selection/model_epoch_${epoch}"
    "$train_py" -u evaluate_light.py --checkpoint "$checkpoints/model_epoch_${epoch}.pt" \
        --candidate cbam_spatial_refiner_v1 --val-dir "$val_dir" --device cuda \
        --max-samples 0 --save-samples no --match-estimate-level no \
        --estimate-dir "$result" --save-json "$result/summary.json" \
        2>&1 | tee "$logs/evaluation_epoch_${epoch}.log"
}

check_results() {
    "$train_py" - "$logs" "$source_summary" <<'PY_RESULTS'
import json,math,sys
from pathlib import Path
root=Path(sys.argv[1])/'pesq_selection'
source=json.loads(Path(sys.argv[2]).read_text())
rows=[]
for p in root.glob('model_epoch_*/summary.json'):
    r=json.loads(p.read_text());epoch=int(p.parent.name.rsplit('_',1)[1])
    assert r['status']=='complete' and r['expected_samples']==r['completed_samples']==600
    for key in ('frontend','frontend_protocol','num_mics','selected_pairs_sha256','metadata_sha256','metric_protocol'):
        assert r[key]==source[key],key
    assert r['metric_protocol']['target_gain_matching'] is False
    score=r['mean']['enhanced']['pesq'];assert math.isfinite(score)
    if epoch==0:
        for kind in ('enhanced','noisy'):
            for metric,value in source['mean'][kind].items():
                assert abs(r['mean'][kind][metric]-value)<1e-5,(kind,metric)
    rows.append(dict(epoch=epoch,pesq=score,checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256']))
assert any(r['epoch']==0 for r in rows),'Missing full-600 initialization gate'
rows.sort(key=lambda x:x['epoch']);best=max(rows,key=lambda x:x['pesq'])
result=dict(status='complete' if rows[-1]['epoch']==10 or best['pesq']>=3.4 else 'partial',
    selection_scope='Same 600 comparison samples; not an independent final test',
    results=rows,best_completed_epoch=best['epoch'],best_completed_pesq=best['pesq'],target_reached=best['pesq']>=3.4)
(root/'checkpoint_ranking.json').write_text(json.dumps(result,indent=2))
print(int(result['target_reached']))
PY_RESULTS
}

"$train_py" -u train_light_spatial.py --warmstart "$source_checkpoint" \
    --train-dir "$train_dir" --val-dir "$val_dir" --checkpoint-dir "$checkpoints" \
    --best-dir "$best" --log-dir "$logs" --num-epochs 0 --parallel-mode none --num-workers 0 \
    --cpu-threads 2 --cpu-thread-scale 1 2>&1 | tee "$logs/initialization.log"
evaluate_epoch 0
check_results >/dev/null
for epoch in {1..10}; do
    printf 'phase=train epoch=%s\n' "$epoch" >> "$logs/gpu_utilization.log"
    "$train_py" -u -m torch.distributed.run --standalone --nproc_per_node=2 \
        train_light_spatial.py --warmstart "$source_checkpoint" \
        --train-dir "$train_dir" --val-dir "$val_dir" --checkpoint-dir "$checkpoints" \
        --best-dir "$best" --log-dir "$logs" --num-epochs "$epoch" \
        --parallel-mode ddp --batch-size 1 --grad-accum-steps 4 --segment-seconds 6 \
        --num-workers 4 --strict-memory no --cpu-threads 2 --cpu-thread-scale 1 \
        --learning-rate 0.001 --grad-clip 5 --use-amp no --model-amp no --allow-tf32 no \
        --resume yes --resume-reset-lr no --stop-on-non-finite yes --skip-non-finite-batches no \
        --save-every 1 2>&1 | tee -a "$logs/training.log"
    printf 'phase=evaluation epoch=%s\n' "$epoch" >> "$logs/gpu_utilization.log"
    evaluate_epoch "$epoch"
    if [[ "$(check_results)" == 1 ]]; then
        printf 'Full 600-sample PESQ reached 3.4; stop for verification.\n'
        break
    fi
done

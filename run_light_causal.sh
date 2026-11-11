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
max_epochs="${CAUSAL_MAX_EPOCHS:-30}"
if [[ ! "$max_epochs" =~ ^[1-9][0-9]{0,8}$ ]]; then
    printf 'CAUSAL_MAX_EPOCHS must be a positive integer of at most nine digits.\n' >&2
    exit 2
fi
export CAUSAL_MAX_EPOCHS="$max_epochs"
candidate="${CAUSAL_MODEL_CANDIDATE:-causal_unet_spatial_v1}"
learning_rate=0.001
case "$candidate" in
    causal_unet_spatial_v1) entry=train_light_causal_fast.py; run_prefix=causal_fast ;;
    causal_recurrent_spatial_v1) entry=train_light_causal_recurrent.py; run_prefix=causal_recurrent ;;
    causal_recurrent_unet65_v1) entry=train_light_causal_recurrent65.py; run_prefix=causal_recurrent65 ;;
    causal_fullband_tcn_v1) entry=train_light_causal_fullband.py; run_prefix=fullband ;;
    causal_fullband_spatial_tcn_v1) entry=train_light_causal_fullband_spatial.py; run_prefix=fullband_spatial; learning_rate=0.0003 ;;
    causal_retention_subband_gru_v1) entry=train_light_causal_subband.py; run_prefix=retention_subband; learning_rate=0.0003 ;;
    causal_fullband_conditioned_subband_v1) entry=train_light_causal_conditioned_subband.py; run_prefix=conditioned_subband; learning_rate=0.0003 ;;
    causal_frozen_conditioned_subband_v1) entry=train_light_causal_frozen_conditioned_subband.py; run_prefix=frozen_conditioned; learning_rate=0.001 ;;
    causal_spatial_conditioned_subband_v1) entry=train_light_causal_spatial_conditioned_subband.py; run_prefix=spatial_conditioned; learning_rate=0.001 ;;
    causal_gated_multichannel_crn_v1) entry=train_light_causal_gated_multichannel.py; run_prefix=gated_multichannel; learning_rate=0.001 ;;
    causal_fullchannel_dualpath_crn_v1) entry=train_light_causal_fullchannel_dualpath.py; run_prefix=fullchannel_dualpath; learning_rate=0.001 ;;
    causal_folded_retention_v1) entry=train_light_causal_folded_retention.py; run_prefix=folded_retention; learning_rate=0.001 ;;
    causal_joint_conditioned_subband_v1) entry=train_light_causal_joint_conditioned_subband.py; run_prefix=joint_conditioned; learning_rate=0.0001 ;;
    causal_wide_conditioned_subband_v1) entry=train_light_causal_wide_conditioned_subband.py; run_prefix=wide_conditioned; learning_rate=0.001 ;;
    causal_monaural_spectral_v1) entry=train_light_causal_monaural.py; run_prefix=monaural_spectral; learning_rate=0.001 ;;
    causal_decoder_monaural_spectral_v1) entry=train_light_causal_decoder_monaural.py; run_prefix=decoder_monaural; learning_rate=0.0001 ;;
    causal_recurrent_adapt_monaural_v1) entry=train_light_causal_recurrent_adapt.py; run_prefix=recurrent_adapt; learning_rate=0.0001 ;;
    causal_global_spectral_unet_v1) entry=train_light_causal_global_spectral.py; run_prefix=global_spectral; learning_rate=0.001 ;;
    causal_global_spectral_beamformer_v1) entry=train_light_causal_global_beam.py; run_prefix=global_beam; learning_rate=0.001 ;;
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
assert speeds and float(speeds[-1])>0
if sys.argv[3] not in ('causal_retention_subband_gru_v1','causal_fullband_conditioned_subband_v1','causal_frozen_conditioned_subband_v1','causal_spatial_conditioned_subband_v1','causal_fullchannel_dualpath_crn_v1','causal_folded_retention_v1','causal_joint_conditioned_subband_v1','causal_wide_conditioned_subband_v1','causal_monaural_spectral_v1','causal_decoder_monaural_spectral_v1','causal_recurrent_adapt_monaural_v1'):
    assert float(speeds[-1])>=20,'Hot global throughput is insufficient for the one-hour budget'
if sys.argv[3] in ('causal_fullband_tcn_v1','causal_fullband_spatial_tcn_v1','causal_retention_subband_gru_v1','causal_fullband_conditioned_subband_v1','causal_frozen_conditioned_subband_v1','causal_spatial_conditioned_subband_v1','causal_gated_multichannel_crn_v1','causal_fullchannel_dualpath_crn_v1','causal_folded_retention_v1','causal_joint_conditioned_subband_v1','causal_wide_conditioned_subband_v1','causal_monaural_spectral_v1','causal_decoder_monaural_spectral_v1','causal_recurrent_adapt_monaural_v1','causal_global_spectral_unet_v1','causal_global_spectral_beamformer_v1'):
    assert state['args']['causal_training_objective']=='spectral_mrstft_waveform_pmsqe_v1'
    assert state['args']['pmsqe_weight']==.1
    expected_entry={'causal_fullband_spatial_tcn_v1':'train_light_causal_fullband_spatial.py',
        'causal_retention_subband_gru_v1':'train_light_causal_subband.py',
        'causal_fullband_conditioned_subband_v1':'train_light_causal_conditioned_subband.py',
        'causal_frozen_conditioned_subband_v1':'train_light_causal_frozen_conditioned_subband.py',
        'causal_spatial_conditioned_subband_v1':'train_light_causal_spatial_conditioned_subband.py',
        'causal_gated_multichannel_crn_v1':'train_light_causal_gated_multichannel.py',
        'causal_fullchannel_dualpath_crn_v1':'train_light_causal_fullchannel_dualpath.py',
        'causal_folded_retention_v1':'train_light_causal_folded_retention.py',
        'causal_joint_conditioned_subband_v1':'train_light_causal_joint_conditioned_subband.py',
        'causal_wide_conditioned_subband_v1':'train_light_causal_wide_conditioned_subband.py',
        'causal_global_spectral_unet_v1':'train_light_causal_global_spectral.py',
        'causal_global_spectral_beamformer_v1':'train_light_causal_global_beam.py',
        'causal_monaural_spectral_v1':'train_light_causal_monaural.py',
        'causal_decoder_monaural_spectral_v1':'train_light_causal_decoder_monaural.py',
        'causal_recurrent_adapt_monaural_v1':'train_light_causal_recurrent_adapt.py'}.get(sys.argv[3],'train_light_causal_fullband.py')
    assert expected_entry in state['args']['causal_training_implementation_sha256']
    if sys.argv[3]=='causal_fullband_spatial_tcn_v1':
        warm=Path(state['args']['spatial_fullband_warmstart_checkpoint'])
        assert warm.is_absolute() and hashlib.sha256(warm.read_bytes()).hexdigest()==state['args']['spatial_fullband_warmstart_sha256']
        source,_=read_candidate_checkpoint(warm,'causal_fullband_tcn_v1')
        assert int(source['epoch'])==state['args']['spatial_fullband_sample_epoch_offset']>=1
    if sys.argv[3]=='causal_retention_subband_gru_v1':
        warm=Path(state['args']['subband_warmstart_checkpoint'])
        assert warm.is_absolute() and hashlib.sha256(warm.read_bytes()).hexdigest()==state['args']['subband_warmstart_sha256']
        source,_=read_candidate_checkpoint(warm,'causal_frequency_attention_retention_v1')
        assert int(source['epoch'])>=1
        assert source['args']['attention_sample_epoch_offset']+int(source['epoch'])==state['args']['subband_sample_epoch_offset']
    if sys.argv[3]=='causal_fullband_conditioned_subband_v1':
        warm=Path(state['args']['conditioned_subband_warmstart_checkpoint'])
        assert warm.is_absolute() and hashlib.sha256(warm.read_bytes()).hexdigest()==state['args']['conditioned_subband_warmstart_sha256']
        source,_=read_candidate_checkpoint(warm,'causal_retention_subband_gru_v1')
        assert int(source['epoch'])>=1
        assert source['args']['subband_sample_epoch_offset']+int(source['epoch'])==state['args']['conditioned_subband_sample_epoch_offset']
    if sys.argv[3]=='causal_frozen_conditioned_subband_v1':
        warm=Path(state['args']['frozen_subband_warmstart_checkpoint'])
        assert warm.is_absolute() and hashlib.sha256(warm.read_bytes()).hexdigest()==state['args']['frozen_subband_warmstart_sha256']
        source_id=state['args']['frozen_subband_warmstart_candidate']
        assert source_id in ('causal_retention_subband_gru_v1','causal_fullband_conditioned_subband_v1')
        source,_=read_candidate_checkpoint(warm,source_id)
        assert int(source['epoch'])>=1
        offset='subband_sample_epoch_offset' if source_id=='causal_retention_subband_gru_v1' else 'conditioned_subband_sample_epoch_offset'
        assert source['args'][offset]+int(source['epoch'])==state['args']['frozen_subband_sample_epoch_offset']
        assert state['args']['candidate_evidence']['frozen_backbone_is_causal'] is True
        assert check['frozen_causal_weights_unchanged'] is True
    if sys.argv[3]=='causal_spatial_conditioned_subband_v1':
        warm=Path(state['args']['spatial_conditioned_warmstart_checkpoint'])
        assert warm.is_absolute() and hashlib.sha256(warm.read_bytes()).hexdigest()==state['args']['spatial_conditioned_warmstart_sha256']
        source,_=read_candidate_checkpoint(warm,'causal_frozen_conditioned_subband_v1')
        assert int(source['epoch'])>=1
        assert source['args']['frozen_subband_sample_epoch_offset']+int(source['epoch'])==state['args']['spatial_conditioned_sample_epoch_offset']
        assert state['args']['candidate_evidence']['frozen_backbone_is_causal'] is True
        assert check['frozen_causal_weights_unchanged'] is True
    if sys.argv[3]=='causal_joint_conditioned_subband_v1':
        warm=Path(state['args']['joint_conditioned_warmstart_checkpoint'])
        assert warm.is_absolute() and hashlib.sha256(warm.read_bytes()).hexdigest()==state['args']['joint_conditioned_warmstart_sha256']
        source,_=read_candidate_checkpoint(warm,'causal_spatial_conditioned_subband_v1')
        assert int(source['epoch'])>=1
        assert source['args']['spatial_conditioned_sample_epoch_offset']+int(source['epoch'])==state['args']['joint_conditioned_sample_epoch_offset']
        assert state['args']['gru_training_precision']=='fp16'
        assert state['args']['candidate_evidence']['frozen_backbone'] is False
    if sys.argv[3]=='causal_wide_conditioned_subband_v1':
        warm=Path(state['args']['capacity_warmstart_checkpoint'])
        assert warm.is_absolute() and hashlib.sha256(warm.read_bytes()).hexdigest()==state['args']['capacity_warmstart_sha256']
        source,_=read_candidate_checkpoint(warm,'causal_joint_conditioned_subband_v1')
        assert int(source['epoch'])>=1
        assert source['args']['joint_conditioned_sample_epoch_offset']+int(source['epoch'])==state['args']['capacity_sample_epoch_offset']
        assert state['args']['capacity_head_width']==128 and state['args']['gru_training_precision']=='fp16'
        assert state['args']['candidate_evidence']['frozen_backbone_is_causal'] is True
        assert check['frozen_causal_weights_unchanged'] is True
    if sys.argv[3]=='causal_monaural_spectral_v1':
        warm=Path(state['args']['monaural_warmstart_checkpoint'])
        assert warm.is_absolute() and hashlib.sha256(warm.read_bytes()).hexdigest()==state['args']['monaural_warmstart_sha256']
        source,_=read_candidate_checkpoint(warm,'causal_wide_conditioned_subband_v1')
        assert int(source['epoch'])==state['args']['monaural_warmstart_epoch']>=1
        assert source['args']['capacity_sample_epoch_offset']+int(source['epoch'])==state['args']['monaural_sample_epoch_offset']
        assert state['args']['candidate_evidence']['frozen_backbone_is_causal'] is True
        assert state['args']['candidate_evidence']['source_head_trainable'] is True
        assert state['args']['candidate_evidence']['frozen_parameter_prefix']=='base.base.'
        assert check['frozen_causal_parameter_prefix']=='base.base.' and check['frozen_causal_weights_unchanged'] is True
        assert check['changed_source_head_state_tensors']>0
    if sys.argv[3]=='causal_decoder_monaural_spectral_v1':
        warm=Path(state['args']['decoder_monaural_warmstart_checkpoint'])
        assert warm.is_absolute() and hashlib.sha256(warm.read_bytes()).hexdigest()==state['args']['decoder_monaural_warmstart_sha256']
        source,_=read_candidate_checkpoint(warm,'causal_monaural_spectral_v1')
        assert not source.get('diagnostic_only',False) and not source['args'].get('diagnostic_only',False)
        assert int(source['epoch'])==state['args']['decoder_monaural_warmstart_epoch']>=1
        assert source['args']['monaural_sample_epoch_offset']+int(source['epoch'])==state['args']['decoder_monaural_sample_epoch_offset']
        assert state['args']['gru_training_precision']=='fp16' and state['args']['cudnn_training_benchmark'] is True
        assert state['args']['candidate_evidence']['partially_frozen_backbone'] is True
        assert check['frozen_causal_weights_unchanged'] is True and check['changed_causal_decoder_state_tensors']>0
        assert check['frozen_causal_parameter_rule']=='All base.base parameters except decoders,upsamplers,beamformer'
    elif sys.argv[3]=='causal_recurrent_adapt_monaural_v1':
        warm=Path(state['args']['recurrent_adapt_warmstart_checkpoint'])
        assert warm.is_absolute() and hashlib.sha256(warm.read_bytes()).hexdigest()==state['args']['recurrent_adapt_warmstart_sha256']
        source,_=read_candidate_checkpoint(warm,'causal_decoder_monaural_spectral_v1')
        assert not source.get('diagnostic_only',False) and not source['args'].get('diagnostic_only',False)
        assert int(source['epoch'])==state['args']['recurrent_adapt_warmstart_epoch']>=1
        assert source['args']['decoder_monaural_sample_epoch_offset']+int(source['epoch'])==state['args']['recurrent_adapt_sample_epoch_offset']
        assert state['args']['gru_training_precision']=='fp16' and state['args']['cudnn_training_benchmark'] is True
        assert state['args']['candidate_evidence']['partially_frozen_backbone'] is True
        assert check['frozen_causal_weights_unchanged'] is True and check['changed_causal_recurrent_state_tensors']>0
        assert check['frozen_causal_parameter_rule']=='Freeze root stem,encoders,first three recurrent blocks and shared frequency matrix'
    projection=json.loads((root/'timing_projection.json').read_text())
    baseline=json.loads((root/'zero_mapping_full600/summary.json').read_text())
    assert projection['status']=='measured'
    assert (projection['training_samples'],projection['validation_samples'],projection['metric_samples'])==(53600,600,600)
    assert projection['cuda_checkpoint_sha256']==check['checkpoint_sha256']
    assert baseline['status']=='complete' and baseline['completed_samples']==baseline['expected_samples']==600
    zero=root/'cuda_final/checkpoints/model_epoch_0.pt'
    assert projection['evaluation_checkpoint_sha256']==baseline['checkpoint_sha256']==hashlib.sha256(zero.read_bytes()).hexdigest()
    val_rates=re.findall(r'\[PERF\]\[val\].*?samples/s\(local/global\)=[\d.]+/([\d.]+)',(root/'cuda_final/training.log').read_text())
    assert val_rates
    predicted=53600/float(speeds[-1])+600/float(val_rates[-1])+projection['full600_evaluation_seconds']+60
    assert 0<predicted<=3600,'Measured complete-epoch projection exceeds one hour'
    assert abs(predicted-projection['projected_complete_epoch_seconds'])<1e-6
assert hashlib.sha256((Path(sys.argv[2])/'metadata.csv').read_bytes()).hexdigest()=='8d0b4d229ebc42cd9892e6d0e0d36416007ded01fbc0b753df4bf7e6b734eb02'
names={'run_light_quality.sh','continue_light_quality.sh','run_light_metric.sh','run_light_spatial.sh',
       'run_light_causal.sh','train_light_quality.py','train_light_metric.py','train_light_spatial.py',
       'train_light_causal.py','train_light_causal_fast.py','check_light_causal_cuda.py','benchmark_light_causal.py',
       'run_light_causal_global.sh','train_light_causal_global.py','train_light_causal_recurrent.py','train_light_causal_recurrent65.py'}
names.update({'run_light_causal_fullband.sh','train_light_causal_fullband.py',
    'run_light_causal_retention.sh','train_light_causal_retention.py','run_light_causal_frequency_attention.sh',
    'run_light_causal_fullband_spatial.sh','train_light_causal_fullband_spatial.py',
    'run_light_causal_subband.sh','train_light_causal_subband.py'})
names.update({'run_light_causal_conditioned_subband.sh','train_light_causal_conditioned_subband.py'})
names.update({'run_light_causal_frozen_conditioned_subband.sh','train_light_causal_frozen_conditioned_subband.py'})
names.update({'run_light_causal_spatial_conditioned_subband.sh','train_light_causal_spatial_conditioned_subband.py'})
names.update({'run_light_causal_gated_multichannel.sh','train_light_causal_gated_multichannel.py'})
names.update({'run_light_causal_fullchannel_dualpath.sh','train_light_causal_fullchannel_dualpath.py'})
names.update({'run_light_causal_folded_retention.sh','train_light_causal_folded_retention.py'})
names.update({'run_light_causal_joint_conditioned_subband.sh','train_light_causal_joint_conditioned_subband.py'})
names.update({'run_light_causal_wide_conditioned_subband.sh','train_light_causal_wide_conditioned_subband.py'})
names.update({'run_light_causal_monaural.sh','train_light_causal_monaural.py'})
names.update({'run_light_causal_decoder_monaural.sh','train_light_causal_decoder_monaural.py',
    'run_light_causal_recurrent_adapt.sh','train_light_causal_recurrent_adapt.py'})
names.update({'run_light_causal_global_spectral.sh','train_light_causal_global_spectral.py',
              'run_light_causal_global_beam.sh','train_light_causal_global_beam.py'})
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
warm=''
warm_args=()
if [[ "$candidate" == causal_fullband_spatial_tcn_v1 || "$candidate" == causal_retention_subband_gru_v1 || "$candidate" == causal_fullband_conditioned_subband_v1 || "$candidate" == causal_frozen_conditioned_subband_v1 || "$candidate" == causal_spatial_conditioned_subband_v1 || "$candidate" == causal_joint_conditioned_subband_v1 || "$candidate" == causal_wide_conditioned_subband_v1 || "$candidate" == causal_monaural_spectral_v1 || "$candidate" == causal_decoder_monaural_spectral_v1 || "$candidate" == causal_recurrent_adapt_monaural_v1 ]]; then
    warm=$("$train_py" - "$preflight/cuda_final/checkpoints/model_epoch_2.pt" "$candidate" <<'PY_WARM'
import sys,torch
state=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
key={'causal_retention_subband_gru_v1':'subband_warmstart_checkpoint',
     'causal_fullband_conditioned_subband_v1':'conditioned_subband_warmstart_checkpoint',
     'causal_frozen_conditioned_subband_v1':'frozen_subband_warmstart_checkpoint',
      'causal_spatial_conditioned_subband_v1':'spatial_conditioned_warmstart_checkpoint',
      'causal_joint_conditioned_subband_v1':'joint_conditioned_warmstart_checkpoint',
      'causal_wide_conditioned_subband_v1':'capacity_warmstart_checkpoint',
      'causal_monaural_spectral_v1':'monaural_warmstart_checkpoint',
      'causal_decoder_monaural_spectral_v1':'decoder_monaural_warmstart_checkpoint',
      'causal_recurrent_adapt_monaural_v1':'recurrent_adapt_warmstart_checkpoint'}.get(sys.argv[2],'spatial_fullband_warmstart_checkpoint')
print(state['args'][key])
PY_WARM
)
    warm_args=(--warmstart "$warm")
fi
run="${run_prefix}_$(date +%Y%m%d_%H%M%S)_$$"
if [[ -n "${CAUSAL_RESUME_RUN:-}" ]]; then
    [[ "$candidate" == causal_fullchannel_dualpath_crn_v1 || "$candidate" == causal_folded_retention_v1 || "$candidate" == causal_joint_conditioned_subband_v1 || "$candidate" == causal_wide_conditioned_subband_v1 || "$candidate" == causal_monaural_spectral_v1 || "$candidate" == causal_decoder_monaural_spectral_v1 || "$candidate" == causal_recurrent_adapt_monaural_v1 || "$candidate" == causal_global_spectral_unet_v1 || "$candidate" == causal_global_spectral_beamformer_v1 ]] || { printf 'Resume is verified for the full-channel,folded-retention,joint-conditioned,wide-conditioned,monaural-spectral,global-spectral and global-beam candidates only.\n' >&2; exit 2; }
    run="$(basename -- "$CAUSAL_RESUME_RUN")"
    [[ "$run" == "${run_prefix}_"* && -d "./logs_${candidate}/$run" ]] || { printf 'Invalid existing causal run.\n' >&2; exit 2; }
fi
checkpoints="./checkpoints_${candidate}_${run}"
best="./bestmodels_${candidate}_${run}"
logs="./logs_${candidate}/${run}"
if [[ -n "${CAUSAL_RESUME_RUN:-}" ]]; then
    resume_settings=$("$train_py" - "$logs" "$checkpoints" "$best" "$candidate" "$train_dir" "$val_dir" <<'PY_RESUME'
import hashlib,json,math,os,sys,time,torch
from pathlib import Path
from audit_light_final import require_protocol
from light_candidate_runtime import read_candidate_checkpoint
torch.set_num_threads(2)
root,checkpoints,best=map(lambda value:Path(value).resolve(),sys.argv[1:4])
state,_=read_candidate_checkpoint(checkpoints/'checkpoint_latest.pt',sys.argv[4])
schemas={'causal_fullchannel_dualpath_crn_v1':21,'causal_folded_retention_v1':22,'causal_joint_conditioned_subband_v1':23,'causal_wide_conditioned_subband_v1':25,'causal_monaural_spectral_v1':27,'causal_global_spectral_unet_v1':28,'causal_global_spectral_beamformer_v1':29,'causal_decoder_monaural_spectral_v1':30,'causal_recurrent_adapt_monaural_v1':31}
assert sys.argv[4] in schemas
epoch=int(state['epoch']);assert 1<=epoch<int(os.environ.get('CAUSAL_MAX_EPOCHS','30')) and state['args']['candidate_schema']==schemas[sys.argv[4]]
for key,path in (('log_dir',root),('checkpoint_dir',checkpoints),('best_dir',best),('train_dir',Path(sys.argv[5]).resolve()),('val_dir',Path(sys.argv[6]).resolve())):
    assert Path(state['args'][key]).resolve()==path,key
assert state['args']['parallel_mode']=='ddp' and state['args']['batch_size']*state['args']['grad_accum_steps']*2==8
for name,digest in state['args']['causal_training_implementation_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
timings=[json.loads(line) for line in (root/'epoch_timing.jsonl').read_text().splitlines() if line.strip()]
timing=next(row for row in timings if row['epoch']==epoch)
assert (timing['status'],timing['training_samples'],timing['validation_samples'],timing['full_metric_samples'],timing['optimizer_updates_this_epoch'],timing['global_batch_size'])==('complete',53600,600,600,6700,8)
assert math.isfinite(timing['total_seconds']) and 0<timing['total_seconds']<=3600
baseline=json.loads((root/'pesq_selection/model_epoch_0/summary.json').read_text())
summary=json.loads((root/f'pesq_selection/model_epoch_{epoch}/summary.json').read_text())
require_protocol(baseline);require_protocol(summary)
assert baseline['structure_candidate']==summary['structure_candidate']==sys.argv[4]
for key in ('frontend','frontend_protocol','num_mics','selected_pairs_sha256','metadata_sha256','metric_protocol','candidate_implementation_sha256','candidate_evidence'):
    assert summary[key]==baseline[key],key
assert all(math.isfinite(value) for metrics in summary['mean'].values() for value in metrics.values())
for name,value in baseline['mean']['noisy'].items():assert abs(summary['mean']['noisy'][name]-value)<1e-9
score=summary['mean']['enhanced']['pesq']
assert math.isfinite(score) and score<3.4
if sys.argv[4]=='causal_fullchannel_dualpath_crn_v1':
    assert baseline['mean']['enhanced']['pesq']+.2<=score
ranking=json.loads((root/'pesq_selection/checkpoint_ranking.json').read_text())
assert not ranking['target_reached'] and any(row['epoch']==epoch for row in ranking['results'])
evaluated=checkpoints/f'model_epoch_{epoch}.pt'
assert hashlib.sha256(evaluated.read_bytes()).hexdigest()==summary['checkpoint_sha256']
reference,_=read_candidate_checkpoint(evaluated,sys.argv[4])
assert state['model_state_dict'].keys()==reference['model_state_dict'].keys()
assert all(torch.equal(value,reference['model_state_dict'][name]) and torch.isfinite(value).all() for name,value in state['model_state_dict'].items())
optimizer=state['optimizer_state_dict'];other=reference['optimizer_state_dict']
assert optimizer['param_groups']==other['param_groups'] and optimizer['state'].keys()==other['state'].keys()
assert optimizer['state'] and {int(value['step']) for value in optimizer['state'].values()}=={epoch*6700}
for index,values in optimizer['state'].items():
    assert values.keys()==other['state'][index].keys()
    for name,value in values.items():
        assert torch.isfinite(value).all() and torch.equal(value,other['state'][index][name])
assert state['lr_reducer_state']==reference['lr_reducer_state']
reserve=math.ceil(timing['evaluation_audit_seconds'])+60
assert 0<reserve<1200
(root/'run_status.json').write_text(json.dumps(dict(status='running',resumed_from_completed_epoch=epoch,resumed_unix=time.time()),indent=2))
print(epoch+1,reserve)
PY_RESUME
)
fi
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
    local evaluation_command=("$train_py" -u evaluate_light.py)
    if [[ "$candidate" == causal_decoder_monaural_spectral_v1 || "$candidate" == causal_recurrent_adapt_monaural_v1 ]]; then
        evaluation_command=("$train_py" -u -c 'import torch; torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; import evaluate_light; evaluate_light.main()')
    fi
    timeout --kill-after=15s "${2:-3600}s" "${evaluation_command[@]}" --checkpoint "$checkpoints/model_epoch_${epoch}.pt" \
        --candidate "$candidate" --val-dir "$val_dir" --device cuda \
        --max-samples 0 --save-samples no --match-estimate-level no \
        --estimate-dir "$result" --save-json "$result/summary.json" \
        2>&1 | tee "$logs/evaluation_epoch_${epoch}.log"
}

check_results() {
    "$train_py" - "$logs" "$candidate" "$warm" <<'PY_RESULTS'
import json,math,os,sys
from pathlib import Path
root=Path(sys.argv[1])/'pesq_selection';baseline=json.loads((root/'model_epoch_0/summary.json').read_text())
candidate=sys.argv[2] if len(sys.argv)>2 else 'causal_unet_spatial_v1'
if candidate in ('causal_fullband_spatial_tcn_v1','causal_retention_subband_gru_v1','causal_fullband_conditioned_subband_v1','causal_frozen_conditioned_subband_v1','causal_spatial_conditioned_subband_v1','causal_joint_conditioned_subband_v1','causal_wide_conditioned_subband_v1','causal_monaural_spectral_v1','causal_decoder_monaural_spectral_v1','causal_recurrent_adapt_monaural_v1'):
    import hashlib,torch
    warm=Path(sys.argv[3]).resolve();source=torch.load(warm,map_location='cpu',weights_only=False)
    source_logs=Path(source['args']['log_dir'])
    if candidate=='causal_recurrent_adapt_monaural_v1' and not source_logs.is_absolute():source_logs=warm.parent.parent/source_logs
    old=json.loads((source_logs/'pesq_selection'/f"model_epoch_{source['epoch']}"/'summary.json').read_text())
    assert old['status']=='complete' and old['completed_samples']==old['expected_samples']==600
    assert old['checkpoint_sha256']==hashlib.sha256(warm.read_bytes()).hexdigest()
    expected_source={'causal_retention_subband_gru_v1':'causal_frequency_attention_retention_v1',
        'causal_fullband_conditioned_subband_v1':'causal_retention_subband_gru_v1',
        'causal_frozen_conditioned_subband_v1':'causal_retention_subband_gru_v1',
        'causal_spatial_conditioned_subband_v1':'causal_frozen_conditioned_subband_v1',
        'causal_joint_conditioned_subband_v1':'causal_spatial_conditioned_subband_v1',
        'causal_wide_conditioned_subband_v1':'causal_joint_conditioned_subband_v1',
        'causal_monaural_spectral_v1':'causal_wide_conditioned_subband_v1',
        'causal_decoder_monaural_spectral_v1':'causal_monaural_spectral_v1',
        'causal_recurrent_adapt_monaural_v1':'causal_decoder_monaural_spectral_v1'}.get(candidate,'causal_fullband_tcn_v1')
    if candidate=='causal_frozen_conditioned_subband_v1':
        expected_source=source['args'].get('structure_candidate')
        assert expected_source in ('causal_retention_subband_gru_v1','causal_fullband_conditioned_subband_v1')
    assert old['structure_candidate']==expected_source
    if candidate=='causal_decoder_monaural_spectral_v1':
        reference_path=Path('decoder_source_fp32_reference/summary.json')
        execution=json.loads(reference_path.with_name('execution.json').read_text())
        reference=json.loads(reference_path.read_text())
        assert execution['status']=='complete_original_source_fp32_full600'
        assert execution['cudnn_allow_tf32'] is False and execution['matmul_allow_tf32'] is False
        assert execution['source_checkpoint_sha256']==old['checkpoint_sha256']==reference['checkpoint_sha256']
        assert execution['summary_sha256']==hashlib.sha256(reference_path.read_bytes()).hexdigest()
        assert execution['evaluator_sha256']==hashlib.sha256(Path('evaluate_light.py').read_bytes()).hexdigest()
        assert reference['status']=='complete' and reference['completed_samples']==reference['expected_samples']==600
        assert reference['structure_candidate']==expected_source
        for key in ('frontend','frontend_protocol','num_mics','selected_pairs_sha256','metadata_sha256','metric_protocol'):
            assert reference[key]==old[key],key
        for name,value in old['mean']['noisy'].items():assert abs(reference['mean']['noisy'][name]-value)<1e-10,name
        old=reference
else:
    old=json.loads(Path('logs_cbam_metric_v1/metric_20261030_052113_1573812/pesq_selection/model_epoch_0/summary.json').read_text())
keys=('frontend','frontend_protocol','num_mics','selected_pairs_sha256','metadata_sha256','metric_protocol')
for key in keys:assert baseline[key]==old[key],key
for name,value in old['mean']['noisy'].items():assert abs(baseline['mean']['noisy'][name]-value)<1e-10,name
warm_candidate=candidate in ('causal_fullband_spatial_tcn_v1','causal_retention_subband_gru_v1','causal_fullband_conditioned_subband_v1','causal_frozen_conditioned_subband_v1','causal_spatial_conditioned_subband_v1','causal_joint_conditioned_subband_v1','causal_wide_conditioned_subband_v1','causal_monaural_spectral_v1','causal_decoder_monaural_spectral_v1','causal_recurrent_adapt_monaural_v1')
expected=old['mean']['enhanced'] if warm_candidate else baseline['mean']['noisy']
tolerance=1e-4 if warm_candidate else 1e-3
for name,value in expected.items():assert abs(baseline['mean']['enhanced'][name]-value)<tolerance,name
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
result=dict(status='target_reached_pending_streaming_audit' if best['pesq']>=3.4 else ('complete' if rows[-1]['epoch']==int(os.environ.get('CAUSAL_MAX_EPOCHS','30')) else 'partial'),
    selection_scope='Same full600 comparison samples; not an independent final test',
    model_is_causal=True,results=rows,best_completed_epoch=best['epoch'],best_completed_pesq=best['pesq'],
    target_reached=best['pesq']>=3.4,final_full600_waveform_streaming_audit_required=True)
(root/'checkpoint_ranking.json').write_text(json.dumps(result,indent=2))
print(int(result['target_reached']))
PY_RESULTS
}

if [[ -n "${CAUSAL_RESUME_RUN:-}" ]]; then
    read -r start_epoch evaluation_reserve <<< "$resume_settings"
else
    "$train_py" -u "$entry" "${warm_args[@]}" --train-dir "$train_dir" --val-dir "$val_dir" \
    --checkpoint-dir "$checkpoints" --best-dir "$best" --log-dir "$logs" \
    --num-epochs 0 --parallel-mode none --num-workers 0 --cpu-threads 2 \
        2>&1 | tee "$logs/initialization.log"
    baseline_start=$(date +%s)
    phase=baseline_full600
    evaluate_epoch 0
    evaluation_reserve=$(( $(date +%s) - baseline_start + 60 ))
    (( evaluation_reserve < 1200 )) || { printf 'Full600 evaluation needs too much of the one-hour budget.\n' >&2; exit 1; }
    start_epoch=1
fi
check_results >/dev/null
for ((epoch=start_epoch; epoch<=max_epochs; epoch++)); do
    phase="training_epoch_${epoch}"
    epoch_start=$(date +%s)
    train_budget=$(( 3600 - evaluation_reserve ))
    if [[ "$candidate" == causal_folded_retention_v1 || "$candidate" == causal_joint_conditioned_subband_v1 ]]; then
        # Allocate evaluation from actual elapsed time; retain the final3600s audit.
        train_budget=3590
    fi
    printf 'phase=train epoch=%s\n' "$epoch" >> "$logs/gpu_utilization.log"
    timeout --kill-after=15s "${train_budget}s" "$train_py" -u -m torch.distributed.run --standalone --nproc_per_node=2 \
        "$entry" --train-dir "$train_dir" --val-dir "$val_dir" \
        --checkpoint-dir "$checkpoints" --best-dir "$best" --log-dir "$logs" \
        --num-epochs "$epoch" --parallel-mode ddp --batch-size "$batch_size" --grad-accum-steps "$grad_accum_steps" \
        --segment-seconds 6 --num-workers 4 --strict-memory no --cpu-threads 2 --cpu-thread-scale 1 \
        --learning-rate "$learning_rate" --grad-clip 5 --use-amp no --model-amp no --allow-tf32 no \
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

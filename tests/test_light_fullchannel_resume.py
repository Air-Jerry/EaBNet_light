"""Only evaluated complete boundaries may resume; retain Adam and protocol identity."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch


ROOT=Path(__file__).resolve().parents[1]
CANDIDATE='causal_fullchannel_dualpath_crn_v1'


@pytest.mark.parametrize('fault',[None,'unfinished','wrong_directory','wrong_dataset','wrong_batch',
    'wrong_schema','target_reached','old_target_reached','partial_metrics','gain','cropped',
    'changed_cohort','bad_noisy','no_quality_gain','hour','nonfinite_weights','unevaluated_latest',
    'different_adam','optimizer_steps','scheduler','source_changed','checkpoint_changed'])
def test_resume_requires_same_evaluated_state_and_complete_budget(tmp_path,monkeypatch,fault):
    root=tmp_path/'logs';root.mkdir()
    checkpoints=tmp_path/'checkpoints';checkpoints.mkdir()
    best=tmp_path/'best'
    train=tmp_path/'train';validation=tmp_path/'validation'
    script=tmp_path/'train_entry.py';script.write_text('original implementation')
    args=dict(log_dir=str(root),checkpoint_dir=str(checkpoints),best_dir=str(best),
        train_dir=str(train),val_dir=str(validation),parallel_mode='ddp',batch_size=2,grad_accum_steps=2,
        candidate_schema=21,causal_training_implementation_sha256={str(script):hashlib.sha256(script.read_bytes()).hexdigest()})
    state=dict(epoch=3,args=args,model_state_dict={'weight':torch.ones(1)},
        optimizer_state_dict=dict(state={0:dict(step=torch.tensor(20100),exp_avg=torch.zeros(1),exp_avg_sq=torch.ones(1))},
            param_groups=[dict(lr=.0005,params=[0])]),lr_reducer_state=dict(best=.3,wait=2))
    evaluated=copy.deepcopy(state)
    checked=checkpoints/'model_epoch_3.pt';checked.write_bytes(b'evaluated checkpoint')
    baseline=json.loads((ROOT/'logs_causal_spatial_conditioned_subband_v0/fullchannel_preflight_20261102/zero_full600_summary.json').read_text())
    summary=copy.deepcopy(baseline);summary['mean']['enhanced']['pesq']=2.4
    summary['checkpoint_sha256']=hashlib.sha256(checked.read_bytes()).hexdigest()
    timing=dict(epoch=3,status='complete',training_samples=53600,validation_samples=600,full_metric_samples=600,
        optimizer_updates_this_epoch=6700,global_batch_size=8,total_seconds=3200.,evaluation_audit_seconds=217.4)
    ranking=dict(target_reached=False,results=[dict(epoch=3)])
    if fault=='unfinished':timing['status']='running'
    if fault=='wrong_directory':state['args']['log_dir']=str(tmp_path/'other')
    if fault=='wrong_dataset':state['args']['train_dir']=str(tmp_path/'other_data')
    if fault=='wrong_batch':state['args']['grad_accum_steps']=1
    if fault=='wrong_schema':state['args']['candidate_schema']=20
    if fault=='target_reached':summary['mean']['enhanced']['pesq']=3.4
    if fault=='old_target_reached':ranking['target_reached']=True
    if fault=='partial_metrics':summary['completed_samples']=599
    if fault=='gain':summary['metric_protocol']['target_gain_matching']=True
    if fault=='cropped':summary['frontend_protocol']['evaluation_extent']='crop'
    if fault=='changed_cohort':summary['selected_pairs_sha256']='other samples'
    if fault=='bad_noisy':summary['mean']['noisy']['pesq']+=.01
    if fault=='no_quality_gain':summary['mean']['enhanced']['pesq']=1.1
    if fault=='hour':timing['total_seconds']=3600.01
    if fault=='nonfinite_weights':state['model_state_dict']['weight'][0]=float('nan')
    if fault=='unevaluated_latest':state['model_state_dict']['weight'][0]=2.
    if fault=='different_adam':state['optimizer_state_dict']['state'][0]['exp_avg'][0]=1.
    if fault=='optimizer_steps':
        state['optimizer_state_dict']['state'][0]['step']=torch.tensor(800)
        evaluated['optimizer_state_dict']['state'][0]['step']=torch.tensor(800)
    if fault=='scheduler':state['lr_reducer_state']['wait']=0
    if fault=='source_changed':script.write_text('changed implementation')
    if fault=='checkpoint_changed':checked.write_bytes(b'not evaluated')
    for epoch,report in ((0,baseline),(3,summary)):
        directory=root/f'pesq_selection/model_epoch_{epoch}';directory.mkdir(parents=True)
        (directory/'summary.json').write_text(json.dumps(report))
    (root/'epoch_timing.jsonl').write_text(json.dumps(timing)+'\n')
    (root/'pesq_selection/checkpoint_ranking.json').write_text(json.dumps(ranking))
    status=root/'run_status.json';status.write_text('original complete status')
    def read(path,candidate):
        assert candidate==CANDIDATE
        return (state if Path(path).name=='checkpoint_latest.pt' else evaluated),candidate
    monkeypatch.setattr('light_candidate_runtime.read_candidate_checkpoint',read)
    monkeypatch.setattr(sys,'argv',['-',str(root),str(checkpoints),str(best),CANDIDATE,str(train),str(validation)])
    shell=(ROOT/'run_light_causal.sh').read_text()
    code=shell.split("<<'PY_RESUME'\n",1)[1].split('\nPY_RESUME',1)[0]
    if fault:
        with pytest.raises((AssertionError,ValueError)):
            exec(compile(code,'fullchannel completed-boundary resume','exec'),{})
        assert status.read_text()=='original complete status'
    else:
        exec(compile(code,'fullchannel completed-boundary resume','exec'),{})
        assert json.loads(status.read_text())['resumed_from_completed_epoch']==3
        assert state['optimizer_state_dict']['param_groups'][0]['lr']==.0005
        assert state['lr_reducer_state']['wait']==2
        assert int(state['optimizer_state_dict']['state'][0]['step'])==20100


def test_resume_guard_runs_before_status_sampler_and_cold_initialization():
    shell=(ROOT/'run_light_causal.sh').read_text()
    assert shell.index("<<'PY_RESUME'")<shell.index('sampler=$!')<shell.index('trap finish EXIT')
    block=shell.split('if [[ -n "${CAUSAL_RESUME_RUN:-}" ]]; then\n    read -r start_epoch evaluation_reserve',1)[1]
    assert block.index('else\n')<block.index('--num-epochs 0')<block.index('start_epoch=1\nfi')
    assert 'for ((epoch=start_epoch; epoch<=30; epoch++)); do' in shell

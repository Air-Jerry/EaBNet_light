"""Actual CPU warm inheritance, joint training, saved inference and Adam resume."""
from pathlib import Path
import copy
import hashlib
import json
import sys

import pytest
import torch

import evaluate_light
import train_light
import train_light_causal_joint_conditioned_subband as joint
from light_candidate_runtime import (JOINT_CONDITIONED_CANDIDATE,SPATIAL_CONDITIONED_CANDIDATE,
    build_model_candidate,candidate_metadata,read_candidate_checkpoint)
from test_light_causal_workflow import argv
from test_light_quality_workflow import run_cli


def source_checkpoint(arguments,tmp_path):
    torch.set_num_threads(2);torch.manual_seed(1234)
    settings=train_light.parse_args(arguments,defaults=dict(norm_type='LN'))
    model=build_model_candidate(SPATIAL_CONDITIONED_CANDIDATE)
    with torch.no_grad():
        for module in (model.base.beamformer,model.base.mapping_output,model.output,model.spatial_projection):
            module.weight.normal_(std=.003)
    args=vars(settings);args.update(candidate_metadata(SPATIAL_CONDITIONED_CANDIDATE))
    args['spatial_conditioned_sample_epoch_offset']=9
    path=tmp_path/'source19'/'model_epoch_2.pt';path.parent.mkdir()
    torch.save(dict(epoch=2,args=args,model_state_dict=model.state_dict()),path)
    return path,model


def test_joint_original_loop_warm_transfer_training_resume_and_inference(argv,tmp_path):
    source,before=source_checkpoint(argv,tmp_path)
    directory=Path(argv[argv.index('--checkpoint-dir')+1])
    entry='train_light_causal_joint_conditioned_subband.py'
    run_cli(entry,[*argv,'--warmstart',str(source),'--num-epochs','0'])
    initial,_=read_candidate_checkpoint(directory/'model_epoch_0.pt',JOINT_CONDITIONED_CANDIDATE)
    assert initial['args']['joint_conditioned_sample_epoch_offset']==11
    assert initial['args']['gru_training_precision']=='fp16'
    assert not initial['optimizer_state_dict']['state']
    for name,value in before.state_dict().items():
        torch.testing.assert_close(initial['model_state_dict'][name],value,rtol=0,atol=0)
    after,_=evaluate_light.load_model(directory/'model_epoch_0.pt',torch.device('cpu'),JOINT_CONDITIONED_CANDIDATE)
    before.eval();value=torch.randn(1,17,257,8,2)*.02
    with torch.inference_mode():torch.testing.assert_close(after(value),before(value),rtol=0,atol=0)
    assert all(p.requires_grad for p in after.parameters())
    for epoch in (1,2):
        run_cli(entry,[*argv,'--num-epochs',str(epoch)])
        state,_=read_candidate_checkpoint(directory/'checkpoint_latest.pt',JOINT_CONDITIONED_CANDIDATE)
        assert state['epoch']==epoch
        assert {int(v['step']) for v in state['optimizer_state_dict']['state'].values()}=={epoch*2}
        assert all(torch.isfinite(v).all() for v in state['model_state_dict'].values())
        assert all(torch.isfinite(v).all() for s in state['optimizer_state_dict']['state'].values() for v in s.values())
        assert not torch.equal(state['model_state_dict']['base.mapping_output.weight'],initial['model_state_dict']['base.mapping_output.weight'])
        assert not torch.equal(state['model_state_dict']['base.beamformer.weight'],initial['model_state_dict']['base.beamformer.weight'])
    with pytest.raises(AssertionError,match='gru_training_precision differs'):
        run_cli(entry,[*argv,'--num-epochs','3','--gru-training-precision','fp32'])
    saved,_=read_candidate_checkpoint(directory/'checkpoint_latest.pt',JOINT_CONDITIONED_CANDIDATE)
    assert saved['epoch']==2


@pytest.mark.parametrize('flags,expected',[(['--target-ref-mic','1'],'reference0'),
    (['--use-amp','yes'],'declared internal precision'),(['--skip-non-finite-batches','yes'],'Nonfinite')])
def test_joint_keeps_original_frontend_and_fails_nonfinite(argv,flags,expected):
    with pytest.raises(ValueError,match=expected):joint.parse_args([*argv,*flags])


@pytest.mark.parametrize('fault',[None,'wrong_source','source_changed','sampler_changed','fp32','frozen'])
def test_formal_joint_launcher_binds_source_sampler_and_measured_precision(tmp_path,monkeypatch,fault):
    source=tmp_path/'source.pt';source.write_bytes(b'original source')
    state=dict(args=dict(joint_conditioned_warmstart_checkpoint=str(source),
        joint_conditioned_warmstart_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        joint_conditioned_sample_epoch_offset=12,gru_training_precision='fp16',
        candidate_evidence=dict(frozen_backbone=False)))
    parent=dict(epoch=3,args=dict(spatial_conditioned_sample_epoch_offset=9))
    if fault=='source_changed':source.write_bytes(b'changed')
    if fault=='sampler_changed':parent['epoch']=2
    if fault=='fp32':state['args']['gru_training_precision']='fp32'
    if fault=='frozen':state['args']['candidate_evidence']['frozen_backbone']=True
    def read(path,candidate):
        assert candidate==SPATIAL_CONDITIONED_CANDIDATE and fault!='wrong_source'
        return parent,candidate
    shell=(Path(__file__).resolve().parents[1]/'run_light_causal.sh').read_text()
    marker="    if sys.argv[3]=='causal_joint_conditioned_subband_v1':"
    code=marker+shell.split(marker,1)[1].split("    projection=json.loads",1)[0]
    import textwrap
    monkeypatch.setattr(sys,'argv',['-','preflight','validation',JOINT_CONDITIONED_CANDIDATE])
    namespace=dict(Path=Path,sys=sys,hashlib=hashlib,state=state,read_candidate_checkpoint=read)
    if fault:
        with pytest.raises(AssertionError):exec(compile(textwrap.dedent(code),'joint source binding','exec'),namespace)
    else:exec(compile(textwrap.dedent(code),'joint source binding','exec'),namespace)


@pytest.mark.parametrize('fault',[None,'partial','gain','noncausal'])
def test_registered_joint_keeps_all_full600_protocol_guards(fault):
    from audit_light_final import require_protocol
    root=Path(__file__).resolve().parents[1]
    report=json.loads((root/'logs_causal_spatial_conditioned_subband_v0/joint_quality_256updates_20261102/joint_full600_summary.json').read_text())
    report=copy.deepcopy(report)
    if fault=='partial':report['completed_samples']=599
    if fault=='gain':report['metric_protocol']['target_gain_matching']=True
    if fault=='noncausal':report['candidate_evidence']['future_frames']=1
    if fault:
        with pytest.raises(ValueError):require_protocol(report)
    else:require_protocol(report)

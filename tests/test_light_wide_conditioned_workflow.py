"""Capacity comparison uses real warm weights, original loop and exact Adam resume."""
from pathlib import Path

import pytest
import torch

import evaluate_light
import train_light
import train_light_causal_wide_conditioned_subband as capacity
from light_candidate_runtime import (JOINT_CONDITIONED_CANDIDATE,JOINT_FIXED_CONDITIONED_CANDIDATE,
    WIDE_CONDITIONED_CANDIDATE,build_model_candidate,candidate_metadata,read_candidate_checkpoint)
from test_light_causal_workflow import argv
from test_light_quality_workflow import run_cli


def source_checkpoint(arguments,tmp_path):
    torch.set_num_threads(2);torch.manual_seed(1234)
    settings=train_light.parse_args(arguments,defaults=dict(norm_type='LN'))
    model=build_model_candidate(JOINT_CONDITIONED_CANDIDATE)
    with torch.no_grad():
        for module in (model.base.beamformer,model.base.mapping_output,model.output,
                       model.fullband_projection,model.spatial_projection):
            module.weight.normal_(std=.003)
    args=vars(settings);args.update(candidate_metadata(JOINT_CONDITIONED_CANDIDATE))
    args['joint_conditioned_sample_epoch_offset']=11
    path=tmp_path/'source23'/'model_epoch_2.pt';path.parent.mkdir()
    torch.save(dict(epoch=2,args=args,model_state_dict=model.state_dict()),path)
    return path,model.eval()


@pytest.mark.parametrize('width,candidate,schema',[(32,JOINT_FIXED_CONDITIONED_CANDIDATE,24),(128,WIDE_CONDITIONED_CANDIDATE,25)])
def test_capacity_original_loop_warm_inference_training_and_resume(argv,tmp_path,width,candidate,schema):
    source,before=source_checkpoint(argv,tmp_path)
    directory=Path(argv[argv.index('--checkpoint-dir')+1]);entry='train_light_causal_wide_conditioned_subband.py'
    run_cli(entry,[*argv,'--warmstart',str(source),'--head-width',str(width),'--num-epochs','0'])
    initial,_=read_candidate_checkpoint(directory/'model_epoch_0.pt',candidate)
    assert initial['args']['candidate_schema']==schema and initial['args']['capacity_sample_epoch_offset']==13
    assert initial['args']['capacity_head_width']==width and not initial['optimizer_state_dict']['state']
    model,_=evaluate_light.load_model(directory/'model_epoch_0.pt',torch.device('cpu'),candidate)
    value=torch.randn(1,17,257,8,2)*.02
    with torch.inference_mode():torch.testing.assert_close(model(value),before(value),rtol=1e-4,atol=2e-6)
    evidence=initial['args']['candidate_evidence']
    assert evidence['author_confirmed'] is False and evidence['frozen_backbone_is_causal'] is True
    assert evidence['warm_start_candidate']==JOINT_CONDITIONED_CANDIDATE
    assert evidence['model_config']['parameters']==sum(p.numel() for p in model.parameters())
    assert evidence['model_config']['trainable_parameters']==sum(p.numel() for p in model.parameters() if p.requires_grad)
    for epoch in (1,2):
        run_cli(entry,[*argv,'--num-epochs',str(epoch)])
        state,_=read_candidate_checkpoint(directory/'checkpoint_latest.pt',candidate)
        assert state['epoch']==epoch
        assert {int(item['step']) for item in state['optimizer_state_dict']['state'].values()}=={epoch*2}
        assert all(torch.isfinite(v).all() for v in state['model_state_dict'].values())
        assert all(torch.isfinite(v).all() for item in state['optimizer_state_dict']['state'].values() for v in item.values())
        for name,value in state['model_state_dict'].items():
            if name.startswith('base.'):torch.testing.assert_close(value,initial['model_state_dict'][name],rtol=0,atol=0)
        assert not torch.equal(state['model_state_dict']['output.weight'],initial['model_state_dict']['output.weight'])
    with pytest.raises(AssertionError,match='Head width differs'):
        run_cli(entry,[*argv,'--num-epochs','3','--head-width',str(32 if width==128 else 128)])
    unchanged,_=read_candidate_checkpoint(directory/'checkpoint_latest.pt',candidate)
    assert unchanged['epoch']==2
    with pytest.raises(ValueError,match='Candidate mismatch'):
        read_candidate_checkpoint(directory/'checkpoint_latest.pt',JOINT_CONDITIONED_CANDIDATE)


@pytest.mark.parametrize('flags,expected',[(['--target-ref-mic','1'],'reference0'),
    (['--use-amp','yes'],'declared internal precision'),(['--skip-non-finite-batches','yes'],'Nonfinite')])
def test_capacity_rejects_changed_frontend_and_nonfinite_skipping(argv,flags,expected):
    with pytest.raises(ValueError,match=expected):capacity.parse_args([*argv,*flags])


@pytest.mark.parametrize('candidate',[JOINT_FIXED_CONDITIONED_CANDIDATE,WIDE_CONDITIONED_CANDIDATE])
@pytest.mark.parametrize('fault',[None,'partial','gain','future'])
def test_capacity_full600_protocol_guards(candidate,fault):
    import json
    from audit_light_final import require_protocol
    root=Path(__file__).resolve().parents[1]
    report=json.loads((root/'logs_causal_spatial_conditioned_subband_v0/joint_conditioned_preflight_20261102/epoch1_full600_summary.json').read_text())
    report.update(candidate_metadata(candidate))
    if fault=='partial':report['completed_samples']=599
    if fault=='gain':report['metric_protocol']['target_gain_matching']=True
    if fault=='future':report['candidate_evidence']['future_frames']=1
    if fault:
        with pytest.raises(ValueError):require_protocol(report)
    else:require_protocol(report)


@pytest.mark.parametrize('fault',[None,'wrong_source','source_changed','sampler_changed','width','precision','noncausal','changed_base'])
def test_wide_formal_preflight_binds_real_source_and_frozen_causal_weights(tmp_path,monkeypatch,fault):
    import hashlib
    import sys
    import textwrap
    source=tmp_path/'source.pt';source.write_bytes(b'completed source')
    state=dict(args=dict(capacity_warmstart_checkpoint=str(source),
        capacity_warmstart_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        capacity_sample_epoch_offset=13,capacity_head_width=128,gru_training_precision='fp16',
        candidate_evidence=dict(frozen_backbone_is_causal=True)))
    check=dict(frozen_causal_weights_unchanged=True)
    parent=dict(epoch=2,args=dict(joint_conditioned_sample_epoch_offset=11))
    if fault=='source_changed':source.write_bytes(b'changed')
    if fault=='sampler_changed':parent['epoch']=1
    if fault=='width':state['args']['capacity_head_width']=32
    if fault=='precision':state['args']['gru_training_precision']='fp32'
    if fault=='noncausal':state['args']['candidate_evidence']['frozen_backbone_is_causal']=False
    if fault=='changed_base':check['frozen_causal_weights_unchanged']=False
    def read(path,candidate):
        assert candidate==JOINT_CONDITIONED_CANDIDATE and fault!='wrong_source'
        return parent,candidate
    root=Path(__file__).resolve().parents[1];shell=(root/'run_light_causal.sh').read_text()
    marker="    if sys.argv[3]=='causal_wide_conditioned_subband_v1':"
    code=marker+shell.split(marker,1)[1].split('    projection=json.loads',1)[0]
    monkeypatch.setattr(sys,'argv',['-','preflight','validation',WIDE_CONDITIONED_CANDIDATE])
    namespace=dict(Path=Path,sys=sys,hashlib=hashlib,state=state,check=check,read_candidate_checkpoint=read)
    if fault:
        with pytest.raises(AssertionError):exec(compile(textwrap.dedent(code),'wide preflight source binding','exec'),namespace)
    else:exec(compile(textwrap.dedent(code),'wide preflight source binding','exec'),namespace)

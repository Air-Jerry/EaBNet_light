"""Real original-loop warm start, finite learning and saved Adam recovery."""
from pathlib import Path

import pytest
import torch

import evaluate_light
import train_light
import train_light_causal_monaural as monaural
from light_candidate_runtime import (MONAURAL_SPECTRAL_CANDIDATE,WIDE_CONDITIONED_CANDIDATE,
    build_model_candidate,candidate_metadata,read_candidate_checkpoint)
from test_light_causal_workflow import argv
from test_light_quality_workflow import run_cli


def source_checkpoint(arguments,tmp_path):
    torch.set_num_threads(2);torch.manual_seed(1234)
    settings=train_light.parse_args(arguments,defaults=dict(norm_type='LN'))
    model=build_model_candidate(WIDE_CONDITIONED_CANDIDATE)
    with torch.no_grad():model.output.weight.normal_(std=.003)
    saved=vars(settings);saved.update(candidate_metadata(WIDE_CONDITIONED_CANDIDATE))
    saved['capacity_sample_epoch_offset']=13
    path=tmp_path/'source25'/'model_epoch_2.pt';path.parent.mkdir()
    torch.save(dict(epoch=2,args=saved,model_state_dict=model.state_dict()),path)
    return path,model.eval()


def test_monaural_original_loop_warm_inference_learning_and_adam_resume(argv,tmp_path):
    source,before=source_checkpoint(argv,tmp_path)
    directory=Path(argv[argv.index('--checkpoint-dir')+1]);entry='train_light_causal_monaural.py'
    run_cli(entry,[*argv,'--warmstart',str(source),'--num-epochs','0'])
    initial,_=read_candidate_checkpoint(directory/'model_epoch_0.pt',MONAURAL_SPECTRAL_CANDIDATE)
    assert initial['args']['candidate_schema']==27 and initial['args']['monaural_sample_epoch_offset']==15
    assert not initial['optimizer_state_dict']['state']
    model,_=evaluate_light.load_model(directory/'model_epoch_0.pt',torch.device('cpu'),MONAURAL_SPECTRAL_CANDIDATE)
    value=torch.randn(1,17,257,8,2)*.02
    with torch.inference_mode():torch.testing.assert_close(model(value),before(value),rtol=0,atol=0)
    evidence=initial['args']['candidate_evidence']
    assert evidence['author_confirmed'] is False and evidence['frozen_backbone_is_causal'] is True
    assert evidence['warm_start_candidate']==WIDE_CONDITIONED_CANDIDATE
    assert evidence['model_config']['parameters']==sum(p.numel() for p in model.parameters())==5686130
    assert evidence['model_config']['trainable_parameters']==sum(p.numel() for p in model.parameters() if p.requires_grad)==3106432
    for epoch in (1,2):
        run_cli(entry,[*argv,'--num-epochs',str(epoch)])
        state,_=read_candidate_checkpoint(directory/'checkpoint_latest.pt',MONAURAL_SPECTRAL_CANDIDATE)
        assert state['epoch']==epoch
        assert {int(item['step']) for item in state['optimizer_state_dict']['state'].values()}=={epoch*2}
        assert all(torch.isfinite(v).all() for v in state['model_state_dict'].values())
        assert all(torch.isfinite(v).all() for item in state['optimizer_state_dict']['state'].values() for v in item.values())
        for name,item in state['model_state_dict'].items():
            if name.startswith('base.base.'):torch.testing.assert_close(item,initial['model_state_dict'][name],rtol=0,atol=0)
        assert not torch.equal(state['model_state_dict']['output.weight'],initial['model_state_dict']['output.weight'])
        assert not torch.equal(state['model_state_dict']['base.output.weight'],initial['model_state_dict']['base.output.weight'])
    changed=torch.load(source,map_location='cpu',weights_only=False)
    changed['model_state_dict']['output.bias'].add_(.01);torch.save(changed,source)
    with pytest.raises(ValueError,match='warmstart_sha256 differs'):
        monaural.parse_args([*argv,'--num-epochs','3'])
    assert read_candidate_checkpoint(directory/'checkpoint_latest.pt',MONAURAL_SPECTRAL_CANDIDATE)[0]['epoch']==2


@pytest.mark.parametrize('flags,expected',[(['--target-ref-mic','1'],'reference0'),
    (['--model-amp','yes'],'FP32'),(['--skip-non-finite-batches','yes'],'Nonfinite')])
def test_monaural_rejects_changed_frontend_precision_and_nonfinite_skipping(argv,flags,expected):
    with pytest.raises(ValueError,match=expected):monaural.parse_args([*argv,*flags])


@pytest.mark.parametrize('fault',[None,'partial','gain','future'])
def test_monaural_preserves_full600_protocol_guards(fault):
    import json
    from audit_light_final import require_protocol
    root=Path(__file__).resolve().parents[1]
    report=json.loads((root/'logs_causal_spatial_conditioned_subband_v0/wide_conditioned_preflight_20261102/zero_full600_summary.json').read_text())
    report.update(candidate_metadata(MONAURAL_SPECTRAL_CANDIDATE))
    if fault=='partial':report['completed_samples']=599
    if fault=='gain':report['metric_protocol']['target_gain_matching']=True
    if fault=='future':report['candidate_evidence']['future_frames']=1
    if fault:
        with pytest.raises(ValueError):require_protocol(report)
    else:require_protocol(report)


@pytest.mark.parametrize('fault',[None,'wrong_source','source_changed','source_epoch','sampler_changed',
    'noncausal','source_head_frozen','prefix','frozen_root_changed','check_prefix','unchanged_head'])
def test_monaural_formal_preflight_binds_complete_source_and_nested_frozen_root(tmp_path,monkeypatch,fault):
    import hashlib
    import sys
    import textwrap
    source=tmp_path/'source.pt';source.write_bytes(b'completed source')
    state=dict(args=dict(monaural_warmstart_checkpoint=str(source),
        monaural_warmstart_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        monaural_warmstart_epoch=2,monaural_sample_epoch_offset=15,
        candidate_evidence=dict(frozen_backbone_is_causal=True,source_head_trainable=True,frozen_parameter_prefix='base.base.')))
    check=dict(frozen_causal_weights_unchanged=True,frozen_causal_parameter_prefix='base.base.',changed_source_head_state_tensors=50)
    parent=dict(epoch=2,args=dict(capacity_sample_epoch_offset=13))
    if fault=='source_changed':source.write_bytes(b'changed')
    if fault=='source_epoch':parent['epoch']=1
    if fault=='sampler_changed':parent['args']['capacity_sample_epoch_offset']=12
    if fault=='noncausal':state['args']['candidate_evidence']['frozen_backbone_is_causal']=False
    if fault=='source_head_frozen':state['args']['candidate_evidence']['source_head_trainable']=False
    if fault=='prefix':state['args']['candidate_evidence']['frozen_parameter_prefix']='base.'
    if fault=='frozen_root_changed':check['frozen_causal_weights_unchanged']=False
    if fault=='check_prefix':check['frozen_causal_parameter_prefix']='base.'
    if fault=='unchanged_head':check['changed_source_head_state_tensors']=0
    def read(path,candidate):
        assert candidate==WIDE_CONDITIONED_CANDIDATE and fault!='wrong_source'
        return parent,candidate
    root=Path(__file__).resolve().parents[1];shell=(root/'run_light_causal.sh').read_text()
    marker="    if sys.argv[3]=='causal_monaural_spectral_v1':"
    code=marker+shell.split(marker,1)[1].split('    projection=json.loads',1)[0]
    monkeypatch.setattr(sys,'argv',['-','preflight','validation',MONAURAL_SPECTRAL_CANDIDATE])
    namespace=dict(Path=Path,sys=sys,hashlib=hashlib,state=state,check=check,read_candidate_checkpoint=read)
    if fault:
        with pytest.raises(AssertionError):exec(compile(textwrap.dedent(code),'monaural preflight source binding','exec'),namespace)
    else:exec(compile(textwrap.dedent(code),'monaural preflight source binding','exec'),namespace)


@pytest.mark.parametrize('candidate,prefix',[(MONAURAL_SPECTRAL_CANDIDATE,'base.base.'),(WIDE_CONDITIONED_CANDIDATE,'base.')])
@pytest.mark.parametrize('changed_root',[False,True])
def test_cuda_checker_keeps_the_correct_frozen_root_namespace(candidate,prefix,changed_root):
    import textwrap
    root=Path(__file__).resolve().parents[1]
    script=(root/'check_light_causal_cuda.py').read_text()
    start=script.index('        monaural = ');end=script.index('        report.update(status="passed"',start)
    names=[prefix+'weight','base.output.weight' if candidate==MONAURAL_SPECTRAL_CANDIDATE else 'output.weight','output.bias']
    zero={'model_state_dict':{name:torch.zeros(1) for name in names}}
    weights={name:torch.ones(1) for name in names};weights[names[0]]=torch.tensor([float(changed_root)])
    report={};namespace=dict(state={'args':{'structure_candidate':candidate}},weights=weights,zero=zero,torch=torch,report=report)
    if changed_root:
        with pytest.raises(AssertionError):exec(compile(textwrap.dedent(script[start:end]),'nested root invariant','exec'),namespace)
    else:
        exec(compile(textwrap.dedent(script[start:end]),'nested root invariant','exec'),namespace)
        assert report['frozen_causal_weights_unchanged'] and report['frozen_causal_parameter_prefix']==prefix

"""Real original-loop warm start, finite learning and saved Adam recovery."""
from pathlib import Path

import pytest
import torch

import evaluate_light
import train_light
import train_light_causal_covariance as covariance
from light_candidate_runtime import (COVARIANCE_CANDIDATE,WIDE_CONDITIONED_CANDIDATE,
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


def test_covariance_original_loop_warm_inference_learning_and_adam_resume(argv,tmp_path):
    source,before=source_checkpoint(argv,tmp_path)
    directory=Path(argv[argv.index('--checkpoint-dir')+1]);entry='train_light_causal_covariance.py'
    run_cli(entry,[*argv,'--warmstart',str(source),'--num-epochs','0'])
    initial,_=read_candidate_checkpoint(directory/'model_epoch_0.pt',COVARIANCE_CANDIDATE)
    assert initial['args']['candidate_schema']==26 and initial['args']['covariance_sample_epoch_offset']==15
    assert not initial['optimizer_state_dict']['state']
    model,_=evaluate_light.load_model(directory/'model_epoch_0.pt',torch.device('cpu'),COVARIANCE_CANDIDATE)
    value=torch.randn(1,17,257,8,2)*.02
    with torch.inference_mode():torch.testing.assert_close(model(value),before(value),rtol=0,atol=0)
    evidence=initial['args']['candidate_evidence']
    assert evidence['author_confirmed'] is False and evidence['frozen_backbone_is_causal'] is True
    assert evidence['warm_start_candidate']==WIDE_CONDITIONED_CANDIDATE
    assert evidence['model_config']['parameters']==sum(p.numel() for p in model.parameters())==5175434
    assert evidence['model_config']['trainable_parameters']==sum(p.numel() for p in model.parameters() if p.requires_grad)==8850
    for epoch in (1,2):
        run_cli(entry,[*argv,'--num-epochs',str(epoch)])
        state,_=read_candidate_checkpoint(directory/'checkpoint_latest.pt',COVARIANCE_CANDIDATE)
        assert state['epoch']==epoch
        assert {int(item['step']) for item in state['optimizer_state_dict']['state'].values()}=={epoch*2}
        assert all(torch.isfinite(v).all() for v in state['model_state_dict'].values())
        assert all(torch.isfinite(v).all() for item in state['optimizer_state_dict']['state'].values() for v in item.values())
        for name,item in state['model_state_dict'].items():
            if name.startswith('base.'):torch.testing.assert_close(item,initial['model_state_dict'][name],rtol=0,atol=0)
        assert not torch.equal(state['model_state_dict']['output.weight'],initial['model_state_dict']['output.weight'])
    changed=torch.load(source,map_location='cpu',weights_only=False)
    changed['model_state_dict']['output.bias'].add_(.01);torch.save(changed,source)
    with pytest.raises(ValueError,match='warmstart_sha256 differs'):
        covariance.parse_args([*argv,'--num-epochs','3'])
    assert read_candidate_checkpoint(directory/'checkpoint_latest.pt',COVARIANCE_CANDIDATE)[0]['epoch']==2


@pytest.mark.parametrize('flags,expected',[(['--target-ref-mic','1'],'reference0'),
    (['--model-amp','yes'],'FP32'),(['--skip-non-finite-batches','yes'],'Nonfinite')])
def test_covariance_rejects_changed_frontend_precision_and_nonfinite_skipping(argv,flags,expected):
    with pytest.raises(ValueError,match=expected):covariance.parse_args([*argv,*flags])


@pytest.mark.parametrize('fault',[None,'partial','gain','future'])
def test_covariance_preserves_full600_protocol_guards(fault):
    import json
    from audit_light_final import require_protocol
    root=Path(__file__).resolve().parents[1]
    report=json.loads((root/'logs_causal_spatial_conditioned_subband_v0/wide_conditioned_preflight_20261102/zero_full600_summary.json').read_text())
    report.update(candidate_metadata(COVARIANCE_CANDIDATE))
    if fault=='partial':report['completed_samples']=599
    if fault=='gain':report['metric_protocol']['target_gain_matching']=True
    if fault=='future':report['candidate_evidence']['future_frames']=1
    if fault:
        with pytest.raises(ValueError):require_protocol(report)
    else:require_protocol(report)

"""Exercise the new model through original initialization, actual learning and Adam resume."""
from pathlib import Path

import pytest
import torch

import evaluate_light
import train_light_causal_global_spectral as global_spectral
from light_candidate_runtime import (GLOBAL_SPECTRAL_CANDIDATE, all_model_candidates,
    candidate_metadata, read_candidate_checkpoint)
from test_light_causal_workflow import argv
from test_light_quality_workflow import run_cli


def test_global_spectral_original_loop_cold_start_learning_and_resume(argv):
    torch.set_num_threads(2)
    entry='train_light_causal_global_spectral.py'
    directory=Path(argv[argv.index('--checkpoint-dir')+1])
    run_cli(entry,[*argv,'--num-epochs','0'])
    initial,_=read_candidate_checkpoint(directory/'model_epoch_0.pt',GLOBAL_SPECTRAL_CANDIDATE)
    assert initial['args']['candidate_schema']==28 and not initial['optimizer_state_dict']['state']
    assert GLOBAL_SPECTRAL_CANDIDATE in all_model_candidates()
    evidence=candidate_metadata(GLOBAL_SPECTRAL_CANDIDATE)['candidate_evidence']
    assert evidence['author_confirmed'] is False and evidence['frozen_backbone'] is False
    network,_=evaluate_light.load_model(directory/'model_epoch_0.pt',torch.device('cpu'),GLOBAL_SPECTRAL_CANDIDATE)
    assert sum(p.numel() for p in network.parameters())==4945266
    assert all(p.requires_grad and p.dtype==torch.float32 for p in network.parameters())
    value=torch.randn(1,9,257,8,2)*.02
    with torch.inference_mode():assert torch.equal(network(value),value[...,0,:].permute(0,3,1,2))
    for epoch in (1,2):
        run_cli(entry,[*argv,'--num-epochs',str(epoch)])
        state,_=read_candidate_checkpoint(directory/'checkpoint_latest.pt',GLOBAL_SPECTRAL_CANDIDATE)
        assert state['epoch']==epoch
        assert {int(item['step']) for item in state['optimizer_state_dict']['state'].values()}=={epoch*2}
        assert all(torch.isfinite(value).all() for value in state['model_state_dict'].values())
        assert all(torch.isfinite(v).all() for item in state['optimizer_state_dict']['state'].values() for v in item.values())
        for name in ('output.weight','encoders.0.0.weight','recurrent.weight_ih_l0','recurrent.weight_ih_l2'):
            assert not torch.equal(state['model_state_dict'][name],initial['model_state_dict'][name])
    assert state['args']['causal_training_objective']=='spectral_mrstft_waveform_pmsqe_v1'
    assert state['args']['pmsqe_weight']==.1
    assert all('warmstart' not in key for key in state['args'])


@pytest.mark.parametrize('flags,match',[
    (['--target-ref-mic','1'],'reference0'),(['--model-amp','yes'],'FP32'),
    (['--parallel-mode','dp'],'ddp'),(['--skip-non-finite-batches','yes'],'Nonfinite')])
def test_global_spectral_rejects_changed_protocol(argv,flags,match):
    with pytest.raises(ValueError,match=match):global_spectral.parse_args([*argv,*flags])


@pytest.mark.parametrize('fault',[None,'partial','gain','future','noncausal','frontend','channels'])
def test_global_spectral_full_comparison_audit_eligibility(fault):
    import json
    from audit_light_final import require_protocol
    root=Path(__file__).resolve().parents[1]
    report=json.loads((root/'logs_causal_spatial_conditioned_subband_v0/monaural_preflight_20261103/initial_full600_summary.json').read_text())
    report.update(candidate_metadata(GLOBAL_SPECTRAL_CANDIDATE))
    if fault=='partial':report['completed_samples']=599
    if fault=='gain':report['metric_protocol']['target_gain_matching']=True
    if fault=='future':report['candidate_evidence']['future_frames']=1
    if fault=='noncausal':report['candidate_evidence']['model_causal']=False
    if fault=='frontend':report['frontend']['hop_length']=320
    if fault=='channels':report['num_mics']=9
    if fault:
        with pytest.raises(ValueError):require_protocol(report)
    else:require_protocol(report)

"""Original pipeline training, saved inference, Adam resume and protocol guards."""
from pathlib import Path
import copy
import json
import sys

import pytest
import torch

import evaluate_light
import train_light_causal as causal
import train_light_causal_gated_multichannel as gated
from light_candidate_runtime import GATED_MULTICHANNEL_CANDIDATE, build_model_candidate, read_candidate_checkpoint
from test_light_causal_workflow import argv
from test_light_quality_workflow import run_cli


def test_actual_train_save_evaluate_and_resume(argv):
    torch.set_num_threads(2)
    parser,objective,epoch=causal.parse_args,causal.QualityObjective,causal.trainer.run_epoch
    entry='train_light_causal_gated_multichannel.py'
    directory=Path(argv[argv.index('--checkpoint-dir')+1])
    run_cli(entry,[*argv,'--num-epochs','1'])
    assert causal.parse_args is parser and causal.QualityObjective is objective and causal.trainer.run_epoch is epoch
    first,_=read_candidate_checkpoint(directory/'checkpoint_latest.pt',GATED_MULTICHANNEL_CANDIDATE)
    assert first['epoch']==1 and first['args']['candidate_schema']==20
    assert first['args']['pmsqe_weight']==.1 and first['args']['candidate_evidence']['author_confirmed'] is False
    assert first['args']['candidate_evidence']['external_pretrained_weights'] is False
    assert {int(value['step']) for value in first['optimizer_state_dict']['state'].values()}=={2}
    assert torch.count_nonzero(first['model_state_dict']['output.weight'])>0
    model,_=evaluate_light.load_model(directory/'model_epoch_1.pt',torch.device('cpu'),GATED_MULTICHANNEL_CANDIDATE)
    assert sum(p.numel() for p in model.parameters())==405266 and all(p.requires_grad for p in model.parameters())
    value=torch.randn(1,17,257,8,2)*.02
    with torch.inference_mode():
        expected=model(value);changed=value.clone();changed[:,7:]*=13
        torch.testing.assert_close(expected[:,:,:7],model(changed)[:,:,:7],rtol=0,atol=0)
    run_cli(entry,[*argv,'--num-epochs','2'])
    second,_=read_candidate_checkpoint(directory/'checkpoint_latest.pt',GATED_MULTICHANNEL_CANDIDATE)
    assert second['epoch']==2
    assert {int(value['step']) for value in second['optimizer_state_dict']['state'].values()}=={4}
    assert all(torch.isfinite(value).all() for value in second['model_state_dict'].values())
    assert all(torch.isfinite(value).all() for state in second['optimizer_state_dict']['state'].values()
               for value in state.values() if torch.is_tensor(value))


@pytest.mark.parametrize('setting,value',[
    ('--target-ref-mic','1'),('--num-mics','9'),('--sample-rate','8000'),
    ('--hop-length','256'),('--win-length','512'),('--channels','32'),
    ('--allow-tf32','yes'),('--use-amp','yes'),('--model-amp','yes'),
    ('--skip-non-finite-batches','yes'),('--stop-on-non-finite','no'),
])
def test_protocol_changes_are_rejected(argv,setting,value):
    with pytest.raises(ValueError):gated.parse_args([*argv,setting,value])


def test_registered_fixed_width_is_enforced():
    with pytest.raises(ValueError,match='Registered gated'):
        build_model_candidate(GATED_MULTICHANNEL_CANDIDATE,c=32)
    with pytest.raises(ValueError,match='Registered gated'):
        build_model_candidate(GATED_MULTICHANNEL_CANDIDATE,recurrent_layers=2)


@pytest.mark.parametrize('fault',[None,'partial','gain','external_gain','future_frames','wrong_initial'])
def test_launcher_requires_complete_unmodified600_and_causal_evidence(tmp_path,monkeypatch,fault):
    source=dict(status='complete',completed_samples=600,expected_samples=600,
        frontend=dict(n_fft=512),frontend_protocol=dict(normalization='none'),num_mics=8,
        selected_pairs_sha256='fixed600',metadata_sha256='fixedmetadata',
        metric_protocol=dict(target_gain_matching=False,external_normalization_clipping_or_alignment=False),
        mean=dict(enhanced=dict(pesq=2.5),noisy=dict(pesq=1.08)))
    old=tmp_path/'logs_cbam_metric_v1/metric_20261030_052113_1573812/pesq_selection/model_epoch_0/summary.json'
    old.parent.mkdir(parents=True);old.write_text(json.dumps(source))
    run=tmp_path/'new_run'
    for epoch in (0,1):
        row=copy.deepcopy(source)
        row.update(structure_candidate=GATED_MULTICHANNEL_CANDIDATE,
            candidate_evidence=dict(model_causal=True,future_frames=0),
            candidate_implementation_sha256=dict(model='gated'),checkpoint=f'epoch{epoch}.pt',
            checkpoint_sha256=f'epoch{epoch}')
        row['mean']['enhanced']['pesq']=1.08 if epoch==0 else 3.45
        if epoch==0 and fault=='wrong_initial':row['mean']['enhanced']['pesq']=2.
        if epoch==1:
            if fault=='partial':row['completed_samples']=599
            if fault=='gain':row['metric_protocol']['target_gain_matching']=True
            if fault=='external_gain':row['metric_protocol']['external_normalization_clipping_or_alignment']=True
            if fault=='future_frames':row['candidate_evidence']['future_frames']=1
        path=run/f'pesq_selection/model_epoch_{epoch}/summary.json'
        path.parent.mkdir(parents=True);path.write_text(json.dumps(row))
    shell=(Path(__file__).resolve().parents[1]/'run_light_causal.sh').read_text()
    code=shell.split("<<'PY_RESULTS'\n",1)[1].split('\nPY_RESULTS',1)[0]
    monkeypatch.chdir(tmp_path);monkeypatch.setattr(sys,'argv',['-',str(run),GATED_MULTICHANNEL_CANDIDATE,''])
    if fault:
        with pytest.raises(AssertionError):exec(compile(code,'gated launcher results gate','exec'),{})
    else:
        exec(compile(code,'gated launcher results gate','exec'),{})
        ranking=json.loads((run/'pesq_selection/checkpoint_ranking.json').read_text())
        assert ranking['target_reached'] and ranking['final_full600_waveform_streaming_audit_required']

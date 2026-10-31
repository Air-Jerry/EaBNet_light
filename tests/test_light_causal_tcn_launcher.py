"""Full600/identity rejection and complete-epoch recovery for the TCN launcher."""

import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('fault', [None, 'partial', 'gain', 'baseline_changed', 'wrong_source'])
def test_tcn_launcher_requires_full600_and_exact_warm_source(tmp_path, monkeypatch, fault):
    source = tmp_path / 'source.pt'; source_logs = tmp_path / 'source_logs'
    torch.save(dict(epoch=17, args={'log_dir':str(source_logs)}), source)
    report = dict(status='complete', completed_samples=600, expected_samples=600,
        frontend={'n_fft':512}, frontend_protocol={'normalization':'none'}, num_mics=8,
        selected_pairs_sha256='same', metadata_sha256='same', metric_protocol={'target_gain_matching':False},
        structure_candidate='wrong' if fault=='wrong_source' else 'causal_recurrent_unet65_v1',
        candidate_evidence={'model_causal':True,'future_frames':0},
        candidate_implementation_sha256={'model':'source'}, checkpoint=str(source),
        checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        mean={'enhanced':{'pesq':2.50},'noisy':{'pesq':1.08}})
    path = source_logs / 'pesq_selection/model_epoch_17/summary.json'
    path.parent.mkdir(parents=True); path.write_text(json.dumps(report))
    for epoch in (0, 1):
        value = copy.deepcopy(report); value['structure_candidate'] = 'causal_recurrent_tcn_v1'
        value['candidate_implementation_sha256'] = {'model':'tcn'}
        if epoch==0 and fault=='baseline_changed': value['mean']['enhanced']['pesq'] += .01
        if epoch==1:
            value['mean']['enhanced']['pesq'] = 3.45
            if fault=='partial': value['completed_samples'] = 599
            if fault=='gain': value['metric_protocol']['target_gain_matching'] = True
        path = tmp_path / f'run/pesq_selection/model_epoch_{epoch}/summary.json'
        path.parent.mkdir(parents=True); path.write_text(json.dumps(value))
    script = (ROOT / 'run_light_causal_tcn.sh').read_text()
    code = script.split("<<'PY_RESULTS'\n",1)[1].split('\nPY_RESULTS',1)[0]
    monkeypatch.setattr(sys, 'argv', ['-',str(tmp_path/'run'),str(source)])
    if fault:
        with pytest.raises(AssertionError): exec(compile(code,'TCN full600 gate','exec'),{})
    else:
        exec(compile(code,'TCN full600 gate','exec'),{})
        rank = json.loads((tmp_path/'run/pesq_selection/checkpoint_ranking.json').read_text())
        assert rank['target_reached'] and rank['model_is_causal']


@pytest.mark.parametrize('rate,allowed', [('16.9',True),('16.0',False)])
@pytest.mark.parametrize('fault', [None, 'partial_metrics', 'wrong_checkpoint', 'nonfinite_time'])
def test_tcn_hour_projection_retains_full_data_and_measured_metrics(tmp_path, rate, allowed, fault):
    text = (ROOT / 'run_light_causal_tcn.sh').read_text()
    marker = 'rate=float(rates[-1]);predicted_training_seconds=53600/rate'
    code = marker + text.split(marker,1)[1].split("\nassert check['batch_size_per_gpu']",1)[0]
    directory=tmp_path/'cuda_final/checkpoints';directory.mkdir(parents=True)
    zero=directory/'model_epoch_0.pt';zero.write_bytes(b'actual zero checkpoint')
    sha=hashlib.sha256(zero.read_bytes()).hexdigest()
    baseline=dict(status='complete',completed_samples=599 if fault=='partial_metrics' else 600,
        expected_samples=600,checkpoint_sha256=sha)
    projection=dict(status='measured',training_samples=53600,validation_samples=600,metric_samples=600,
        cuda_checkpoint_sha256='wrong' if fault=='wrong_checkpoint' else 'trained64',
        evaluation_checkpoint_sha256=sha,training_samples_per_second=float(rate),
        validation_samples_per_second=40.0,full600_evaluation_seconds=float('nan') if fault=='nonfinite_time' else 220.0)
    (tmp_path/'timing_projection.json').write_text(json.dumps(projection))
    result=tmp_path/'zero_mapping_full600';result.mkdir();(result/'summary.json').write_text(json.dumps(baseline))
    (tmp_path/'cuda_final/training.log').write_text('[PERF][val] samples/s(local/global)=20.0/40.0\n'*2)
    values=dict(rates=[rate],root=tmp_path,json=json,hashlib=hashlib,re=__import__('re'),math=math,
                check={'checkpoint_sha256':'trained64'})
    if allowed and fault is None:
        exec(compile(code,'complete hour budget','exec'),values)
        assert values['predicted_training_seconds'] + 15 + 220 + 60 <= 3600
    else:
        with pytest.raises(AssertionError): exec(compile(code,'complete hour budget','exec'),values)


@pytest.mark.parametrize('fault', [None, 'unfinished', 'wrong_directory', 'target_reached'])
def test_tcn_resume_rejects_incomplete_boundaries(tmp_path, monkeypatch, fault):
    root=tmp_path/'run'; root.mkdir()
    checkpoints=tmp_path/'checkpoints'; best=tmp_path/'best'; warm=tmp_path/'source.pt'
    saved=dict(log_dir=str(root),checkpoint_dir=str(checkpoints),best_dir=str(best),tcn_warmstart_checkpoint=str(warm))
    if fault=='wrong_directory': saved['log_dir']=str(tmp_path/'different')
    state=dict(epoch=3,args=saved,model_state_dict={'weight':torch.ones(1)},
               optimizer_state_dict={'state':{0:{'step':torch.tensor(20100)}}})
    monkeypatch.setattr('light_candidate_runtime.read_candidate_checkpoint',lambda *unused:(state,'causal_recurrent_tcn_v1'))
    timing=dict(epoch=2 if fault=='unfinished' else 3,status='complete',evaluation_audit_seconds=202.3)
    (root/'epoch_timing.jsonl').write_text(json.dumps(timing)+'\n')
    (root/'pesq_selection').mkdir()
    (root/'pesq_selection/checkpoint_ranking.json').write_text(json.dumps(dict(results=[{'epoch':3}],target_reached=fault=='target_reached')))
    script=(ROOT/'run_light_causal_tcn.sh').read_text()
    code=script.split("<<'PY_RESUME'\n",1)[1].split('\nPY_RESUME',1)[0]
    monkeypatch.setattr(sys,'argv',['-',str(root),str(checkpoints),str(best),str(warm)])
    if fault:
        with pytest.raises(AssertionError): exec(compile(code,'TCN resume boundary','exec'),{})
        assert not (root/'run_status.json').exists()
    else:
        exec(compile(code,'TCN resume boundary','exec'),{})
        assert json.loads((root/'run_status.json').read_text())['resumed_from_completed_epoch']==3

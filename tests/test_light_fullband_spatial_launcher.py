"""The new warm baseline must equal its completed source, with all600 unchanged."""

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch


@pytest.mark.parametrize('fault',[None,'raw_initial','partial','gain','source_hash'])
def test_spatial_launcher_binds_warm_source_and_complete_scores(tmp_path,monkeypatch,fault):
    old_logs=tmp_path/'source_logs'
    source=tmp_path/'source.pt'
    torch.save(dict(epoch=1,args=dict(log_dir=str(old_logs))),source)
    report=dict(status='complete',completed_samples=600,expected_samples=600,
        structure_candidate='causal_fullband_tcn_v1',candidate_evidence=dict(model_causal=True,future_frames=0),
        candidate_implementation_sha256=dict(model='fixed'),frontend=dict(n_fft=512),
        frontend_protocol=dict(normalization='none'),num_mics=8,selected_pairs_sha256='same',metadata_sha256='same',
        metric_protocol=dict(target_gain_matching=False),checkpoint='source.pt',
        checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        mean=dict(enhanced=dict(pesq=2.1),noisy=dict(pesq=1.08)))
    old=old_logs/'pesq_selection/model_epoch_1/summary.json';old.parent.mkdir(parents=True)
    original=copy.deepcopy(report)
    if fault=='source_hash':original['checkpoint_sha256']='wrong'
    old.write_text(json.dumps(original))
    run=tmp_path/'run'
    for epoch in (0,1):
        current=copy.deepcopy(report);current['structure_candidate']='causal_fullband_spatial_tcn_v1'
        current['checkpoint']=f'model_epoch_{epoch}.pt';current['checkpoint_sha256']=f'current{epoch}'
        if epoch==0 and fault=='raw_initial':current['mean']['enhanced']['pesq']=1.08
        if epoch==1:
            current['mean']['enhanced']['pesq']=3.45
            if fault=='partial':current['completed_samples']=599
            if fault=='gain':current['metric_protocol']['target_gain_matching']=True
        path=run/f'pesq_selection/model_epoch_{epoch}/summary.json'
        path.parent.mkdir(parents=True);path.write_text(json.dumps(current))
    shell=(Path(__file__).resolve().parents[1]/'run_light_causal.sh').read_text()
    code=shell.split("<<'PY_RESULTS'\n",1)[1].split('\nPY_RESULTS',1)[0]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys,'argv',['-',str(run),'causal_fullband_spatial_tcn_v1',str(source)])
    if fault:
        with pytest.raises(AssertionError):exec(compile(code,'spatial full600 gate','exec'),{})
    else:
        exec(compile(code,'spatial full600 gate','exec'),{})
        ranked=json.loads((run/'pesq_selection/checkpoint_ranking.json').read_text())
        assert ranked['target_reached'] and ranked['final_full600_waveform_streaming_audit_required']

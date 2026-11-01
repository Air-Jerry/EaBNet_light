"""The new warm baseline must equal its completed source, with all600 unchanged."""

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch


@pytest.mark.parametrize('fault',[None,'raw_initial','partial','gain','source_hash'])
@pytest.mark.parametrize('candidate,source_id',[
    ('causal_fullband_spatial_tcn_v1','causal_fullband_tcn_v1'),
    ('causal_retention_subband_gru_v1','causal_frequency_attention_retention_v1'),
    ('causal_fullband_conditioned_subband_v1','causal_retention_subband_gru_v1'),
    ('causal_frozen_conditioned_subband_v1','causal_retention_subband_gru_v1'),
    ('causal_frozen_conditioned_subband_v1','causal_fullband_conditioned_subband_v1')])
def test_spatial_launcher_binds_warm_source_and_complete_scores(tmp_path,monkeypatch,fault,candidate,source_id):
    old_logs=tmp_path/'source_logs'
    source=tmp_path/'source.pt'
    torch.save(dict(epoch=1,args=dict(log_dir=str(old_logs),structure_candidate=source_id)),source)
    report=dict(status='complete',completed_samples=600,expected_samples=600,
        structure_candidate=source_id,candidate_evidence=dict(model_causal=True,future_frames=0),
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
        current=copy.deepcopy(report);current['structure_candidate']=candidate
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
    monkeypatch.setattr(sys,'argv',['-',str(run),candidate,str(source)])
    if fault:
        with pytest.raises(AssertionError):exec(compile(code,'spatial full600 gate','exec'),{})
    else:
        exec(compile(code,'spatial full600 gate','exec'),{})
        ranked=json.loads((run/'pesq_selection/checkpoint_ranking.json').read_text())
        assert ranked['target_reached'] and ranked['final_full600_waveform_streaming_audit_required']


@pytest.mark.parametrize('rate,allowed',[(18.,True),(15.,False)])
@pytest.mark.parametrize('fault',[None,'partial','wrong_checkpoint','nonfinite_time'])
def test_subband_hour_gate_uses_complete_measured_budget(tmp_path,monkeypatch,rate,allowed,fault):
    import re
    import textwrap
    directory=tmp_path/'cuda_final/checkpoints';directory.mkdir(parents=True)
    zero=directory/'model_epoch_0.pt';zero.write_bytes(b'zero prototype checkpoint')
    sha=hashlib.sha256(zero.read_bytes()).hexdigest()
    baseline=dict(status='complete',completed_samples=599 if fault=='partial' else 600,
        expected_samples=600,checkpoint_sha256=sha)
    p=tmp_path/'zero_mapping_full600';p.mkdir();(p/'summary.json').write_text(json.dumps(baseline))
    predicted=53600/rate+600/30+200+60
    projection=dict(status='measured',training_samples=53600,validation_samples=600,metric_samples=600,
        cuda_checkpoint_sha256='wrong' if fault=='wrong_checkpoint' else 'trained64',
        evaluation_checkpoint_sha256=sha,full600_evaluation_seconds=float('nan') if fault=='nonfinite_time' else 200,
        projected_complete_epoch_seconds=predicted)
    (tmp_path/'timing_projection.json').write_text(json.dumps(projection))
    (tmp_path/'cuda_final/training.log').write_text('[PERF][val] samples/s(local/global)=15/30\n')
    text=(Path(__file__).resolve().parents[1]/'run_light_causal.sh').read_text()
    marker="projection=json.loads((root/'timing_projection.json').read_text())"
    code=textwrap.dedent('    '+marker+text.split(marker,1)[1].split('\nassert hashlib.sha256((Path(sys.argv[2])',1)[0])
    namespace=dict(root=tmp_path,check=dict(checkpoint_sha256='trained64'),speeds=[str(rate)],
                   json=json,Path=Path,hashlib=hashlib,re=re)
    if fault or not allowed:
        with pytest.raises(AssertionError):exec(compile(code,'subband hour gate','exec'),namespace)
    else:
        exec(compile(code,'subband hour gate','exec'),namespace)

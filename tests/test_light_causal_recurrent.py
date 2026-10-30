"""Full-frequency recurrence: causal gradients, caches and actual train/resume/DDP."""

import os
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

import evaluate_light
from light_candidate_runtime import read_candidate_checkpoint
from light_causal_recurrent_model import CausalRecurrentSpatial, RECURRENT_CANDIDATE
from light_causal_stream import CausalWaveformStream
from light_quality_loss import QualityObjective
from train_light import build_stft_batch
from test_light_causal_workflow import argv
from test_light_quality_workflow import ROOT, run_cli
from test_light_quality_ddp import run_two_windows_gloo_ranks


@pytest.fixture
def model():
    torch.set_num_threads(2)
    torch.manual_seed(53)
    model = CausalRecurrentSpatial(hidden=32, ffn=64, layers=2, context=5)
    model.decoder.weight.data.normal_(std=.01)
    return model


@pytest.mark.parametrize('training', [False, True])
def test_full_frequency_recurrent_future_frames_have_no_effect(model, training):
    model.train(training)
    value = (torch.randn(1, 15, 257, 8, 2) * .02).requires_grad_()
    changed = value.detach().clone(); changed[:, 9:] *= 13
    output = model(value)
    torch.testing.assert_close(output[:, :, :9], model(changed)[:, :, :9], rtol=0, atol=0)
    output[:, :, :9].square().mean().backward()
    assert torch.count_nonzero(value.grad[:, 9:]) == 0
    assert torch.count_nonzero(value.grad[:, :9]) > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_full_frequency_recurrent_chunking_and_fixed_state(model):
    model.eval(); value = torch.randn(1, 23, 257, 8, 2) * .02
    with torch.inference_mode():
        expected = model(value)
        state = None; parts = []
        for start, end in ((0, 1), (1, 7), (7, 9), (9, 23)):
            part, state = model.forward_stream(value[:, start:end], state)
            parts.append(part)
            assert all(s['hidden'].shape == (1, 257, 32) for s in state['layers'])
            assert not any('k' in s or 'v' in s for s in state['layers'])
        torch.testing.assert_close(torch.cat(parts, 2), expected, rtol=1e-4, atol=3e-6)


def test_full_frequency_recurrent_waveform_stream(model):
    model.eval(); mixture = torch.randn(1, 1481, 8) * .02
    with torch.inference_mode():
        features, _ = build_stft_batch(mixture, mixture[:, :, 0], 512, 160, 320, .5, torch.device('cpu'))
        expected = QualityObjective().reconstruct(model(features), mixture.shape[1])
        stream = CausalWaveformStream(model); stream.warmup()
        actual = torch.cat((stream.push(mixture[:, :311]), stream.push(mixture[:, 311:]), stream.flush()), 1)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-5)


def test_full_frequency_recurrent_initialization_actual_training_and_resume(argv):
    directory = Path(argv[argv.index('--checkpoint-dir') + 1])
    run_cli('train_light_causal_recurrent.py', [*argv, '--num-epochs', '0'])
    initial, _ = read_candidate_checkpoint(directory / 'model_epoch_0.pt', RECURRENT_CANDIDATE)
    model, _ = evaluate_light.load_model(directory / 'model_epoch_0.pt', torch.device('cpu'), RECURRENT_CANDIDATE)
    with torch.inference_mode():
        value = torch.randn(1, 7, 257, 8, 2) * .02
        torch.testing.assert_close(model(value), value[..., 0, :].permute(0, 3, 1, 2), rtol=0, atol=0)
    assert not initial['optimizer_state_dict']['state']
    for epoch in (1, 2):
        run_cli('train_light_causal_recurrent.py', [*argv, '--num-epochs', str(epoch)])
        state, _ = read_candidate_checkpoint(directory / 'checkpoint_latest.pt', RECURRENT_CANDIDATE)
        assert state['epoch'] == epoch
        assert {int(v['step']) for v in state['optimizer_state_dict']['state'].values()} == {2 * epoch}
        assert all(torch.isfinite(v).all() for v in state['model_state_dict'].values())
    assert any(not torch.equal(v, initial['model_state_dict'][k]) for k, v in state['model_state_dict'].items() if 'mhsa' in k)


def test_full_frequency_recurrent_actual_two_rank_update(argv, tmp_path):
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONIOENCODING='utf-8')
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        environment.pop(key, None)
    args = [*argv, '--num-epochs', '1', '--parallel-mode', 'ddp']
    if os.name == 'nt':
        result = run_two_windows_gloo_ranks(tmp_path, args, environment, entry_module='train_light_causal_recurrent')
    else:
        result = subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
            'train_light_causal_recurrent.py', *args], cwd=ROOT, env=environment, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
    state, _ = read_candidate_checkpoint(Path(argv[argv.index('--checkpoint-dir') + 1]) / 'checkpoint_latest.pt', RECURRENT_CANDIDATE)
    assert {int(v['step']) for v in state['optimizer_state_dict']['state'].values()} == {1}


@pytest.mark.parametrize('fault', [None, 'partial', 'gain', 'noncausal'])
def test_recurrent_launcher_selects_only_full600_causal_same_protocol(tmp_path, monkeypatch, fault):
    report = dict(status='complete', expected_samples=600, completed_samples=600,
        structure_candidate=RECURRENT_CANDIDATE, candidate_evidence=dict(model_causal=True, future_frames=0),
        candidate_implementation_sha256={'model': 'same'}, frontend={'n_fft': 512}, frontend_protocol={'power': .5},
        num_mics=8, selected_pairs_sha256='same', metadata_sha256='same', metric_protocol={'target_gain_matching': False},
        mean={'enhanced': {'pesq': 1.08}, 'noisy': {'pesq': 1.08}}, checkpoint='model.pt', checkpoint_sha256='same')
    old = tmp_path / 'logs_cbam_metric_v1/metric_20261030_052113_1573812/pesq_selection/model_epoch_0/summary.json'
    old.parent.mkdir(parents=True); old.write_text(json.dumps(report))
    for epoch in (0, 1):
        value = copy.deepcopy(report)
        if epoch == 1:
            value['mean']['enhanced']['pesq'] = 3.45
            if fault == 'partial': value['completed_samples'] = 599
            if fault == 'gain': value['metric_protocol']['target_gain_matching'] = True
            if fault == 'noncausal': value['candidate_evidence']['model_causal'] = False
        path = tmp_path / f'run/pesq_selection/model_epoch_{epoch}/summary.json'
        path.parent.mkdir(parents=True); path.write_text(json.dumps(value))
    shell = (ROOT / 'run_light_causal.sh').read_text()
    code = shell.split("<<'PY_RESULTS'\n", 1)[1].split('\nPY_RESULTS', 1)[0]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['-', str(tmp_path / 'run'), RECURRENT_CANDIDATE])
    if fault is None:
        exec(compile(code, 'cold causal launcher selection', 'exec'), {})
        ranking = json.loads((tmp_path / 'run/pesq_selection/checkpoint_ranking.json').read_text())
        assert ranking['target_reached'] and ranking['best_completed_epoch'] == 1
    else:
        with pytest.raises(AssertionError): exec(compile(code, 'cold causal launcher selection', 'exec'), {})

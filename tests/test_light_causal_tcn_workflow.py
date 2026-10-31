"""Warm weights, fresh/full Adam state, saved inference and real two-rank updates."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

import evaluate_light
from light_candidate_runtime import TCN_CANDIDATE, RECURRENT65_CANDIDATE, read_candidate_checkpoint
from train_light_causal_tcn import aligned_sampler_type
from test_light_causal_workflow import argv
from test_light_quality_workflow import ROOT, run_cli
from test_light_quality_ddp import run_two_windows_gloo_ranks


def tcn_case(argv, tmp_path):
    run_cli('train_light_causal_recurrent65.py', [*argv, '--num-epochs', '1'])
    source = Path(argv[argv.index('--checkpoint-dir') + 1]) / 'model_epoch_1.pt'
    arguments = list(argv)
    for flag, part in (('--checkpoint-dir','checkpoints'), ('--best-dir','best'), ('--log-dir','logs')):
        arguments[arguments.index(flag) + 1] = str(tmp_path / 'tcn' / part)
    run_cli('train_light_causal_tcn.py', [*arguments, '--warmstart', str(source), '--num-epochs', '0'])
    return arguments, source


def test_tcn_sampler_continues_source_data_epoch():
    base = torch.utils.data.distributed.DistributedSampler
    source = base(list(range(23)), num_replicas=2, rank=0, shuffle=True)
    changed = aligned_sampler_type(base, 17)(list(range(23)), num_replicas=2, rank=0, shuffle=True)
    source.set_epoch(18); changed.set_epoch(1)
    assert list(source) == list(changed)


def test_tcn_actual_warmstart_train_resume_and_saved_inference(argv, tmp_path):
    arguments, source = tcn_case(argv, tmp_path)
    directory = Path(arguments[arguments.index('--checkpoint-dir') + 1])
    initial, _ = read_candidate_checkpoint(directory / 'model_epoch_0.pt', TCN_CANDIDATE)
    original, _ = read_candidate_checkpoint(source, RECURRENT65_CANDIDATE)
    assert not initial['optimizer_state_dict']['state']
    for key, value in original['model_state_dict'].items():
        torch.testing.assert_close(initial['model_state_dict'][key], value, rtol=0, atol=0)
    a, _ = evaluate_light.load_model(source, torch.device('cpu'), RECURRENT65_CANDIDATE)
    b, _ = evaluate_light.load_model(directory / 'model_epoch_0.pt', torch.device('cpu'), TCN_CANDIDATE)
    assert all(p.requires_grad for p in b.parameters())
    with torch.inference_mode():
        x = torch.randn(1, 7, 257, 8, 2) * .01
        torch.testing.assert_close(a(x), b(x), rtol=0, atol=0)
    for epoch in (1, 2):
        run_cli('train_light_causal_tcn.py', [*arguments, '--num-epochs', str(epoch)])
        state, _ = read_candidate_checkpoint(directory / 'checkpoint_latest.pt', TCN_CANDIDATE)
        assert state['epoch'] == epoch
        assert {int(v['step']) for v in state['optimizer_state_dict']['state'].values()} == {2 * epoch}
        assert all(torch.isfinite(v).all() for v in state['model_state_dict'].values())
        assert torch.count_nonzero(state['model_state_dict']['mapping_output.weight']) > 0
    assert state['args']['tcn_sample_epoch_offset'] == original['epoch'] == 1
    assert state['args']['candidate_schema'] == 9
    assert state['args']['candidate_evidence']['frozen_backbone'] is False


def test_tcn_actual_two_rank_update(argv, tmp_path):
    arguments, _ = tcn_case(argv, tmp_path)
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONIOENCODING='utf-8')
    for key in ('RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT'): environment.pop(key, None)
    arguments = [*arguments, '--num-epochs', '1', '--parallel-mode', 'ddp']
    if os.name == 'nt':
        result = run_two_windows_gloo_ranks(tmp_path, arguments, environment, entry_module='train_light_causal_tcn')
    else:
        result = subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
            'train_light_causal_tcn.py', *arguments], cwd=ROOT, env=environment, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
    state, _ = read_candidate_checkpoint(Path(arguments[arguments.index('--checkpoint-dir') + 1]) / 'checkpoint_latest.pt', TCN_CANDIDATE)
    assert {int(v['step']) for v in state['optimizer_state_dict']['state'].values()} == {1}

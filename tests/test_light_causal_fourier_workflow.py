"""Saved Fourier warm weights, cumulative sampler alignment and finite resumed updates."""

from pathlib import Path

import torch

import evaluate_light
from light_candidate_runtime import FOURIER_CANDIDATE, TCN_CANDIDATE, read_candidate_checkpoint
from train_light_causal_fourier import aligned_sampler_type
from test_light_causal_workflow import argv
from test_light_causal_tcn_workflow import tcn_case
from test_light_quality_workflow import run_cli


def test_fourier_sampler_continues_both_previous_training_stages():
    base = torch.utils.data.distributed.DistributedSampler
    old = base(list(range(31)), num_replicas=2, rank=0, shuffle=True)
    new = aligned_sampler_type(base, 17 + 3)(list(range(31)), num_replicas=2, rank=0, shuffle=True)
    old.set_epoch(21); new.set_epoch(1)
    assert list(old) == list(new)


def test_actual_fourier_warmstart_resume_and_saved_inference(argv, tmp_path):
    arguments, _ = tcn_case(argv, tmp_path)
    run_cli('train_light_causal_tcn.py', [*arguments, '--num-epochs', '1'])
    source = Path(arguments[arguments.index('--checkpoint-dir') + 1]) / 'model_epoch_1.pt'
    for flag, part in (('--checkpoint-dir', 'checkpoints'), ('--best-dir', 'best'), ('--log-dir', 'logs')):
        arguments[arguments.index(flag) + 1] = str(tmp_path / 'fourier' / part)
    run_cli('train_light_causal_fourier.py', [*arguments, '--warmstart', str(source), '--num-epochs', '0'])
    directory = Path(arguments[arguments.index('--checkpoint-dir') + 1])
    initial, _ = read_candidate_checkpoint(directory / 'model_epoch_0.pt', FOURIER_CANDIDATE)
    original, _ = read_candidate_checkpoint(source, TCN_CANDIDATE)
    assert initial['args']['fourier_sample_epoch_offset'] == original['args']['tcn_sample_epoch_offset'] + original['epoch'] == 2
    assert not initial['optimizer_state_dict']['state']
    assert initial['args']['candidate_schema'] == 10
    assert initial['args']['candidate_evidence']['author_confirmed'] is False
    assert initial['args']['candidate_evidence']['future_frames'] == 0
    for name, value in original['model_state_dict'].items():
        torch.testing.assert_close(initial['model_state_dict'][name], value, rtol=0, atol=0)
    first, _ = evaluate_light.load_model(source, torch.device('cpu'), TCN_CANDIDATE)
    second, _ = evaluate_light.load_model(directory / 'model_epoch_0.pt', torch.device('cpu'), FOURIER_CANDIDATE)
    assert all(p.requires_grad for p in second.parameters())
    with torch.inference_mode():
        x = torch.randn(1, 17, 257, 8, 2) * .03
        torch.testing.assert_close(first(x), second(x), rtol=0, atol=0)
    for epoch in (1, 2):
        run_cli('train_light_causal_fourier.py', [*arguments, '--num-epochs', str(epoch)])
        state, _ = read_candidate_checkpoint(directory / 'checkpoint_latest.pt', FOURIER_CANDIDATE)
        assert state['epoch'] == epoch
        assert {int(item['step']) for item in state['optimizer_state_dict']['state'].values()} == {2 * epoch}
        assert all(torch.isfinite(value).all() for value in state['model_state_dict'].values())
        assert torch.count_nonzero(state['model_state_dict']['mapping_input.fourier_out.weight']) > 0

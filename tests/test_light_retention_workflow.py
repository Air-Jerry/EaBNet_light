"""Warm-start an actual trained attention checkpoint; save, resume and infer retention."""

from pathlib import Path

import pytest
import torch

import evaluate_light
from light_candidate_runtime import FREQUENCY_ATTENTION_CANDIDATE, RETENTION_CANDIDATE, read_candidate_checkpoint
from test_light_causal_workflow import argv
from test_light_causal_tcn_workflow import tcn_case
from test_light_quality_workflow import run_cli


def test_retention_actual_attention_warmstart_resume_and_saved_inference(argv, tmp_path):
    arguments, _ = tcn_case(argv, tmp_path)
    run_cli('train_light_causal_tcn.py', [*arguments, '--num-epochs', '1'])
    source = Path(arguments[arguments.index('--checkpoint-dir') + 1]) / 'model_epoch_1.pt'
    for flag, part in (('--checkpoint-dir', 'checkpoints'), ('--best-dir', 'best'), ('--log-dir', 'logs')):
        arguments[arguments.index(flag) + 1] = str(tmp_path / 'attention' / part)
    run_cli('train_light_causal_compiled_attention.py', [*arguments, '--warmstart', str(source), '--num-epochs', '1'])
    source = Path(arguments[arguments.index('--checkpoint-dir') + 1]) / 'model_epoch_1.pt'
    original, _ = read_candidate_checkpoint(source, FREQUENCY_ATTENTION_CANDIDATE)
    for flag, part in (('--checkpoint-dir', 'checkpoints'), ('--best-dir', 'best'), ('--log-dir', 'logs')):
        arguments[arguments.index(flag) + 1] = str(tmp_path / 'retention' / part)
    entry = 'train_light_causal_retention.py'
    run_cli(entry, [*arguments, '--warmstart', str(source), '--num-epochs', '0'])
    directory = Path(arguments[arguments.index('--checkpoint-dir') + 1])
    initial, _ = read_candidate_checkpoint(directory / 'model_epoch_0.pt', RETENTION_CANDIDATE)
    assert initial['args']['attention_sample_epoch_offset'] == original['args']['attention_sample_epoch_offset'] + 1
    assert initial['args']['candidate_schema'] == 13
    assert not initial['optimizer_state_dict']['state']
    assert initial['args']['pmsqe_weight'] == .1
    assert entry in initial['args']['causal_training_implementation_sha256']
    assert 'experiments/compiled_training_norm.py' in initial['args']['causal_training_implementation_sha256']
    assert 'experiments/compiled_training_retention.py' in initial['args']['causal_training_implementation_sha256']
    for key, value in original['model_state_dict'].items():
        torch.testing.assert_close(initial['model_state_dict'][key], value, rtol=0, atol=0)
    before, _ = evaluate_light.load_model(source, torch.device('cpu'), FREQUENCY_ATTENTION_CANDIDATE)
    after, _ = evaluate_light.load_model(directory / 'model_epoch_0.pt', torch.device('cpu'), RETENTION_CANDIDATE)
    assert all(parameter.requires_grad for parameter in after.parameters())
    with torch.inference_mode():
        value = torch.randn(1, 17, 257, 8, 2) * .03
        torch.testing.assert_close(before(value), after(value), rtol=0, atol=0)
    for epoch in (1, 2):
        run_cli(entry, [*arguments, '--num-epochs', str(epoch)])
        state, _ = read_candidate_checkpoint(directory / 'checkpoint_latest.pt', RETENTION_CANDIDATE)
        assert {int(item['step']) for item in state['optimizer_state_dict']['state'].values()} == {epoch * 2}
        assert torch.count_nonzero(state['model_state_dict']['retention.output.weight']) > 0
        assert not torch.equal(state['model_state_dict']['retention.qkv.weight'], initial['model_state_dict']['retention.qkv.weight'])
        assert all(torch.isfinite(item).all() for item in state['model_state_dict'].values())
        assert all(torch.isfinite(value).all() for item in state['optimizer_state_dict']['state'].values()
                   for value in item.values() if torch.is_tensor(value))
    bad = [*arguments, '--warmstart', str(source.parent / 'model_epoch_0.pt'), '--num-epochs', '3']
    with pytest.raises(AssertionError, match='warm-start differs'):
        run_cli(entry, bad)

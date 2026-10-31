"""Real warm-start, saved inference and resumed updates from either full-data stage."""

from pathlib import Path

import pytest
import torch

import evaluate_light
from light_candidate_runtime import FREQUENCY_ATTENTION_CANDIDATE, POOLED_FREQUENCY_ATTENTION_CANDIDATE, TCN_CANDIDATE, read_candidate_checkpoint
from test_light_causal_workflow import argv
from test_light_causal_tcn_workflow import tcn_case
from test_light_quality_workflow import run_cli


@pytest.mark.parametrize('refined', [False, True])
@pytest.mark.parametrize('entry,candidate,schema', [('train_light_causal_frequency_attention.py', FREQUENCY_ATTENTION_CANDIDATE, 11), ('train_light_causal_pooled_attention.py', POOLED_FREQUENCY_ATTENTION_CANDIDATE, 12), ('train_light_causal_compiled_attention.py', FREQUENCY_ATTENTION_CANDIDATE, 11)])
def test_attention_warmstart_resume_and_saved_inference(argv, tmp_path, refined, entry, candidate, schema):
    arguments, _ = tcn_case(argv, tmp_path)
    run_cli('train_light_causal_tcn.py', [*arguments, '--num-epochs', '1'])
    source = Path(arguments[arguments.index('--checkpoint-dir') + 1]) / 'model_epoch_1.pt'
    if refined:
        for flag, part in (('--checkpoint-dir', 'checkpoints'), ('--best-dir', 'best'), ('--log-dir', 'logs')):
            arguments[arguments.index(flag) + 1] = str(tmp_path / 'pmsqe' / part)
        run_cli('train_light_causal_pmsqe.py', [*arguments, '--warmstart', str(source), '--num-epochs', '1'])
        source = Path(arguments[arguments.index('--checkpoint-dir') + 1]) / 'model_epoch_1.pt'
    for flag, part in (('--checkpoint-dir', 'checkpoints'), ('--best-dir', 'best'), ('--log-dir', 'logs')):
        arguments[arguments.index(flag) + 1] = str(tmp_path / 'attention' / part)
    run_cli(entry, [*arguments, '--warmstart', str(source), '--num-epochs', '0'])
    directory = Path(arguments[arguments.index('--checkpoint-dir') + 1])
    initial, _ = read_candidate_checkpoint(directory / 'model_epoch_0.pt', candidate)
    original, _ = read_candidate_checkpoint(source, TCN_CANDIDATE)
    assert not initial['optimizer_state_dict']['state']
    assert initial['args']['attention_sample_epoch_offset'] == (3 if refined else 2)
    assert initial['args']['causal_training_objective'] == 'spectral_mrstft_waveform_pmsqe_v1'
    assert initial['args']['pmsqe_weight'] == .1
    assert initial['args']['candidate_schema'] == schema
    assert initial['args']['candidate_evidence']['author_confirmed'] is False
    assert initial['args']['candidate_evidence']['future_frames'] == 0
    assert 'train_light_causal_pmsqe.py' in initial['args']['causal_training_implementation_sha256']
    assert entry in initial['args']['causal_training_implementation_sha256']
    if entry == 'train_light_causal_compiled_attention.py':
        assert 'experiments/compiled_training_norm.py' in initial['args']['causal_training_implementation_sha256']
    for name, value in original['model_state_dict'].items():
        torch.testing.assert_close(initial['model_state_dict'][name], value, rtol=0, atol=0)
    before, _ = evaluate_light.load_model(source, torch.device('cpu'), TCN_CANDIDATE)
    after, _ = evaluate_light.load_model(directory / 'model_epoch_0.pt', torch.device('cpu'), candidate)
    assert all(parameter.requires_grad for parameter in after.parameters())
    with torch.inference_mode():
        value = torch.randn(1, 17, 257, 8, 2) * .03
        torch.testing.assert_close(before(value), after(value), rtol=0, atol=0)
    for epoch in (1, 2):
        run_cli(entry, [*arguments, '--num-epochs', str(epoch)])
        state, _ = read_candidate_checkpoint(directory / 'checkpoint_latest.pt', candidate)
        assert state['epoch'] == epoch
        assert {int(item['step']) for item in state['optimizer_state_dict']['state'].values()} == {2 * epoch}
        assert all(torch.isfinite(value).all() for value in state['model_state_dict'].values())
        assert torch.count_nonzero(state['model_state_dict']['mapping_input.frequency_output.weight']) > 0
        assert all(torch.isfinite(value).all() for item in state['optimizer_state_dict']['state'].values()
                   for value in item.values() if torch.is_tensor(value))
    with pytest.raises(AssertionError, match='reference0'):
        run_cli(entry, [*arguments, '--target-ref-mic', '1', '--num-epochs', '3'])

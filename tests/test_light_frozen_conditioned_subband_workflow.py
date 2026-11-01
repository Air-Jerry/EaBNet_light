"""Actual source16 warm start, zero-condition inference and resumed full-model Adam."""

from pathlib import Path

import pytest
import torch

import evaluate_light
from light_candidate_runtime import (SUBBAND_GRU_CANDIDATE, CONDITIONED_SUBBAND_CANDIDATE, FROZEN_CONDITIONED_CANDIDATE,
                                     read_candidate_checkpoint)
from test_light_causal_workflow import argv
from test_light_causal_tcn_workflow import tcn_case
from test_light_quality_workflow import run_cli


@pytest.mark.parametrize('conditioned_source',[False,True])
def test_frozen_conditioned_subband_actual_warmstart_and_resume(argv, tmp_path, conditioned_source):
    arguments, _ = tcn_case(argv, tmp_path)
    arguments += ['--grad-accum-steps', '1']
    run_cli('train_light_causal_tcn.py', [*arguments, '--num-epochs', '1'])
    source = Path(arguments[arguments.index('--checkpoint-dir')+1])/'model_epoch_1.pt'
    chain = [('attention', 'train_light_causal_compiled_attention.py'),
                        ('retention', 'train_light_causal_retention.py'),
                        ('subband', 'train_light_causal_subband.py')]
    if conditioned_source:chain.append(('source_conditioned','train_light_causal_conditioned_subband.py'))
    for name, entry in chain:
        for flag, part in (('--checkpoint-dir', 'checkpoints'), ('--best-dir', 'best'), ('--log-dir', 'logs')):
            arguments[arguments.index(flag)+1] = str(tmp_path/name/part)
        run_cli(entry, [*arguments, '--warmstart', str(source), '--num-epochs', '1'])
        source = Path(arguments[arguments.index('--checkpoint-dir')+1])/'model_epoch_1.pt'
    source_id = CONDITIONED_SUBBAND_CANDIDATE if conditioned_source else SUBBAND_GRU_CANDIDATE
    original, _ = read_candidate_checkpoint(source, source_id)
    for flag, part in (('--checkpoint-dir', 'checkpoints'), ('--best-dir', 'best'), ('--log-dir', 'logs')):
        arguments[arguments.index(flag)+1] = str(tmp_path/'conditioned'/part)
    entry = 'train_light_causal_frozen_conditioned_subband.py'
    run_cli(entry, [*arguments, '--warmstart', str(source), '--num-epochs', '0'])
    directory = Path(arguments[arguments.index('--checkpoint-dir')+1])
    initial, _ = read_candidate_checkpoint(directory/'model_epoch_0.pt', FROZEN_CONDITIONED_CANDIDATE)
    assert initial['args']['candidate_schema'] == 18
    offset = 'conditioned_subband_sample_epoch_offset' if conditioned_source else 'subband_sample_epoch_offset'
    assert initial['args']['frozen_subband_sample_epoch_offset'] == original['args'][offset]+1
    assert initial['args']['frozen_subband_warmstart_candidate'] == source_id
    assert initial['args']['pmsqe_weight'] == .1 and not initial['optimizer_state_dict']['state']
    for key, value in original['model_state_dict'].items():
        torch.testing.assert_close(initial['model_state_dict'][key], value, rtol=0, atol=0)
    before, _ = evaluate_light.load_model(source, torch.device('cpu'), source_id)
    after, _ = evaluate_light.load_model(directory/'model_epoch_0.pt', torch.device('cpu'), FROZEN_CONDITIONED_CANDIDATE)
    assert all(not p.requires_grad for p in after.base.parameters())
    assert all(p.requires_grad for name,p in after.named_parameters() if not name.startswith('base.'))
    with torch.inference_mode():
        value = torch.randn(1, 17, 257, 8, 2)*.03
        torch.testing.assert_close(before(value), after(value), rtol=0, atol=0)
    for epoch in (1, 2):
        run_cli(entry, [*arguments, '--num-epochs', str(epoch)])
        state, _ = read_candidate_checkpoint(directory/'checkpoint_latest.pt', FROZEN_CONDITIONED_CANDIDATE)
        assert {int(item['step']) for item in state['optimizer_state_dict']['state'].values()} == {epoch*2}
        for key,value in state['model_state_dict'].items():
            if key.startswith('base.'):
                torch.testing.assert_close(value, original['model_state_dict'][key], rtol=0, atol=0)
        assert torch.count_nonzero(state['model_state_dict']['fullband_mix.weight']) > 0
        assert not torch.equal(state['model_state_dict']['fullband_rnn.weight_hh_l0'], initial['model_state_dict']['fullband_rnn.weight_hh_l0'])
        assert all(torch.isfinite(value).all() for value in state['model_state_dict'].values())
        assert all(torch.isfinite(value).all() for item in state['optimizer_state_dict']['state'].values()
                   for value in item.values() if torch.is_tensor(value))
    with pytest.raises(AssertionError, match='warm-start differs'):
        run_cli(entry, [*arguments, '--warmstart', str(source.parent/'model_epoch_0.pt'), '--num-epochs', '3'])
    latest = directory/'checkpoint_latest.pt'
    changed = torch.load(latest, map_location='cpu', weights_only=False)
    key = next(key for key, value in changed['model_state_dict'].items()
               if key.startswith('base.') and value.is_floating_point())
    changed['model_state_dict'][key] = changed['model_state_dict'][key]+.01
    torch.save(changed, latest)
    with pytest.raises(AssertionError, match='Fixed causal base changed'):
        run_cli(entry, [*arguments, '--num-epochs', '3'])

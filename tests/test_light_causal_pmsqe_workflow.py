"""Both perceptual-training modes keep the source output and resume real Adam updates."""

from pathlib import Path

import pytest
import torch

import evaluate_light
from light_candidate_runtime import FOURIER_CANDIDATE, TCN_CANDIDATE, read_candidate_checkpoint
from test_light_causal_workflow import argv
from test_light_causal_tcn_workflow import tcn_case
from test_light_quality_workflow import run_cli


@pytest.mark.parametrize('fourier,candidate', [('no', TCN_CANDIDATE), ('yes', FOURIER_CANDIDATE)])
def test_actual_pmsqe_warmstart_resume_and_inference(argv, tmp_path, fourier, candidate):
    arguments, _ = tcn_case(argv, tmp_path)
    run_cli('train_light_causal_tcn.py', [*arguments, '--num-epochs', '1'])
    source = Path(arguments[arguments.index('--checkpoint-dir') + 1]) / 'model_epoch_1.pt'
    for flag, part in (('--checkpoint-dir', 'checkpoints'), ('--best-dir', 'best'), ('--log-dir', 'logs')):
        arguments[arguments.index(flag) + 1] = str(tmp_path / 'pmsqe' / part)
    arguments.extend(['--frequency-fourier', fourier])
    run_cli('train_light_causal_pmsqe.py', [*arguments, '--warmstart', str(source), '--num-epochs', '0'])
    directory = Path(arguments[arguments.index('--checkpoint-dir') + 1])
    initial, _ = read_candidate_checkpoint(directory / 'model_epoch_0.pt', candidate)
    original, _ = read_candidate_checkpoint(source, TCN_CANDIDATE)
    assert not initial['optimizer_state_dict']['state']
    assert initial['args']['perceptual_sample_epoch_offset'] == 2
    assert initial['args']['perceptual_frequency_fourier'] is (fourier == 'yes')
    assert initial['args']['causal_training_objective'] == 'spectral_mrstft_waveform_pmsqe_v1'
    assert initial['args']['pmsqe_weight'] == .1
    assert 'third_party/asteroid_pmsqe/bark_matrix_16k.mat' in initial['args']['causal_training_implementation_sha256']
    for name, value in original['model_state_dict'].items():
        torch.testing.assert_close(initial['model_state_dict'][name], value, rtol=0, atol=0)
    before, _ = evaluate_light.load_model(source, torch.device('cpu'), TCN_CANDIDATE)
    after, _ = evaluate_light.load_model(directory / 'model_epoch_0.pt', torch.device('cpu'), candidate)
    with torch.inference_mode():
        value = torch.randn(1, 17, 257, 8, 2) * .03
        torch.testing.assert_close(before(value), after(value), rtol=0, atol=0)
    for epoch in (1, 2):
        run_cli('train_light_causal_pmsqe.py', [*arguments, '--num-epochs', str(epoch)])
        state, _ = read_candidate_checkpoint(directory / 'checkpoint_latest.pt', candidate)
        assert state['epoch'] == epoch
        assert {int(item['step']) for item in state['optimizer_state_dict']['state'].values()} == {2 * epoch}
        assert all(torch.isfinite(value).all() for value in state['model_state_dict'].values())
    if fourier == 'no':
        physical_arguments = arguments[:-2]
        with pytest.raises(AssertionError, match='causal_training_objective'):
            run_cli('train_light_causal_tcn.py', [*physical_arguments, '--num-epochs', '3'])

"""Training invariants and a real two-stage, epoch-boundary resume check."""

import csv
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from torch.amp import GradScaler

import train_cts
import train_light


ROOT = Path(__file__).resolve().parents[1]


def test_paper_defaults_and_rejected_architecture_options():
    args = train_cts.parse_args([])
    assert (args.n_fft, args.win_length, args.hop_length, args.power, args.num_mics) == (320, 320, 160, 1.0, 1)
    assert (args.batch_size, args.grad_accum_steps, args.segment_seconds) == (8, 1, 8)
    with pytest.raises(ValueError, match='num-mics'):
        train_cts.parse_args(['--num-mics', '8'])
    with pytest.raises(SystemExit):
        train_cts.parse_args(['--dfsmn-layers', '4'])


def test_rng_round_trip():
    generator = torch.Generator().manual_seed(12)
    train_cts.seed_everything(12, deterministic=False)
    state = train_cts.capture_rng(generator)
    expected = (random.random(), np.random.rand(), torch.rand(3), torch.rand(2, generator=generator))
    train_cts.restore_rng(state, generator)
    actual = (random.random(), np.random.rand(), torch.rand(3), torch.rand(2, generator=generator))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
    assert torch.equal(actual[3], expected[3])


def test_stage_transition_rejects_unrelated_best_checkpoint(tmp_path):
    args = train_cts.parse_args([])
    manifest = {'digest': 'data-and-code-identity'}
    state = train_cts.initial_stage_state()
    state['best_val_loss'] = 0.2
    checkpoint = {'model_name': 'CTSNet', 'stage': 'me', 'val_loss': 0.1,
                  'args': vars(args), 'reproducibility_manifest': manifest}
    path = tmp_path / 'best_me.pt'
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match='validation loss does not match'):
        train_cts.load_verified_best(path, state, args, manifest)
    checkpoint['val_loss'] = 0.2
    checkpoint['stage'] = 'joint'
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match='stage does not match'):
        train_cts.load_verified_best(path, state, args, manifest)
    checkpoint['stage'] = 'me'
    checkpoint['reproducibility_manifest'] = {'digest': 'different-dataset'}
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match='manifest changed'):
        train_cts.load_verified_best(path, state, args, manifest)


class ScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))

    def forward(self, values, frame_lengths=None):
        return self.weight * values


def test_tail_accumulation_matches_valid_frame_weighted_updates():
    # Three microbatches with distinct valid-frame weights form one full window
    # and one short window. Compare against independently computed SGD updates.
    batches = []
    values = [(2.0, 160), (3.0, 480), (4.0, 800)]
    for index, (value, length) in enumerate(values):
        batches.append({'mixture': torch.tensor([[value]]), 'target': torch.tensor([[0.0]]),
                        'lengths': torch.tensor([length]), 'sample_ids': torch.tensor([index])})
    args = train_cts.parse_args(['--batch-size', '1', '--grad-accum-steps', '2',
                                 '--log-perf', 'no', '--mem-log-every-batches', '0',
                                 '--malloc-trim-every-batches', '0', '--gc-every-batches', '0'])
    model = ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    def features(mixture, target, **_kwargs):
        return mixture, target
    def loss(estimate, target, _frames):
        return ((estimate - target) ** 2).mean()
    train_light.run_epoch(model, batches, optimizer, GradScaler('cuda', enabled=False),
                          torch.device('cpu'), [], args, True,
                          loss_fn=loss, feature_builder=features,
                          correct_accumulation=True, loss_weighting='frames',
                          fail_nonfinite_grad=True, forward_with_lengths=True)
    expected = 0.25
    for window in (values[:2], values[2:]):
        numerator = sum((length // 160 + 1) * 2 * expected * value ** 2 for value, length in window)
        denominator = sum(length // 160 + 1 for _value, length in window)
        expected -= 0.01 * numerator / denominator
    assert model.weight.item() == pytest.approx(expected, abs=1e-7)


def test_finite_loss_with_nonfinite_gradient_cannot_silently_step():
    class BrokenGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.clone()

        @staticmethod
        def backward(ctx, gradient):
            return torch.full_like(gradient, float('inf'))

    args = train_cts.parse_args(['--log-perf', 'no'])
    model = ScalarModel()
    batch = {'mixture': torch.ones(1, 1), 'target': torch.zeros(1, 1),
             'lengths': torch.tensor([320]), 'sample_ids': torch.tensor([0])}
    def features(mixture, target, **_kwargs):
        return mixture, target
    def loss(estimate, _target, _frames):
        return BrokenGradient.apply(estimate).mean()
    with pytest.raises(FloatingPointError, match='Non-finite gradient'):
        train_light.run_epoch(model, [batch], torch.optim.SGD(model.parameters(), lr=1),
                              GradScaler('cuda', enabled=False), torch.device('cpu'), [], args, True,
                              loss_fn=loss, feature_builder=features,
                              correct_accumulation=True, loss_weighting='frames', fail_nonfinite_grad=True)
    assert model.weight.item() == 0.25


def _make_audio_dataset(root):
    for split in ('train', 'val'):
        directory = root / split
        directory.mkdir(parents=True)
        rows = []
        for index, length in enumerate((801, 1121)):
            t = np.arange(length) / 16000
            target = (0.1 * np.sin(2 * np.pi * (230 + 60 * index + (20 if split == 'val' else 0)) * t)).astype(np.float32)
            noise = np.random.default_rng(index + (10 if split == 'val' else 0)).normal(0, 0.015, length).astype(np.float32)
            # Existing metadata interface also accepts multichannel mixtures;
            # CTS selects channel zero through the unchanged Dataset contract.
            mixture = np.stack([target + noise, target - noise], axis=-1)
            mix_path, target_path = directory / f'mix{index}.wav', directory / f'target{index}.wav'
            sf.write(mix_path, mixture, 16000, subtype='FLOAT')
            sf.write(target_path, target, 16000, subtype='FLOAT')
            rows.append({'sample_id': index, 'mixture_path': mix_path.name, 'target_path': target_path.name})
        with (directory / 'metadata.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _train_command(data, output, epochs, resume):
    return [sys.executable, str(ROOT / 'train_cts.py'),
            '--train-dir', str(data / 'train'), '--val-dir', str(data / 'val'),
            '--checkpoint-dir', str(output / 'checkpoints'), '--best-dir', str(output / 'best'),
            '--log-dir', str(output / 'logs'), '--num-epochs', str(epochs),
            '--pretrain-max-epochs', '1', '--allow-unconverged-pretrain', 'yes',
            '--device', 'cpu', '--batch-size', '2', '--num-workers', '0',
            '--cpu-threads', '2', '--resume', resume, '--log-perf', 'no',
            '--mem-log-every-batches', '0', '--malloc-trim-every-batches', '0',
            '--gc-every-batches', '0']


def _run(command):
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + '\n' + result.stderr


def test_real_two_stage_variable_length_training_and_exact_resume(tmp_path):
    data = tmp_path / 'data'
    _make_audio_dataset(data)
    resumed, direct = tmp_path / 'resumed', tmp_path / 'direct'
    _run(_train_command(data, resumed, 2, 'no'))
    _run(_train_command(data, resumed, 3, 'yes'))
    _run(_train_command(data, direct, 3, 'no'))
    resumed_ckpt = torch.load(resumed / 'checkpoints' / 'checkpoint_latest.pt', weights_only=False)
    direct_ckpt = torch.load(direct / 'checkpoints' / 'checkpoint_latest.pt', weights_only=False)
    assert resumed_ckpt['stage'] == 'joint'
    assert resumed_ckpt['epoch'] == 3
    assert resumed_ckpt['stage_epoch'] == 2
    assert resumed_ckpt['me_converged'] is False
    assert [group['lr'] for group in resumed_ckpt['optimizer_state_dict']['param_groups']] == [1e-4, 1e-3]
    assert resumed_ckpt['train_loss'] == direct_ckpt['train_loss']
    assert resumed_ckpt['val_loss'] == direct_ckpt['val_loss']
    for key, value in direct_ckpt['model_state_dict'].items():
        assert torch.equal(resumed_ckpt['model_state_dict'][key], value), key
    assert (resumed / 'best' / 'best_me.pt').exists()
    assert (resumed / 'best' / 'best_model.pt').exists()
    summary = json.loads((resumed / 'checkpoints' / 'training_summary.json').read_text())
    assert summary['status'] == 'finished'
    assert any('before ME convergence' in item for item in summary['protocol_deviations'])
    # An already early-stopped joint run must stay stopped on resume even if the
    # user extends the total budget. This guards a previously identified bug.
    resumed_ckpt['lr_reducer_state']['stale_stop_epochs'] = 5
    torch.save(resumed_ckpt, resumed / 'checkpoints' / 'checkpoint_latest.pt')
    _run(_train_command(data, resumed, 4, 'yes'))
    stopped = torch.load(resumed / 'checkpoints' / 'checkpoint_latest.pt', weights_only=False)
    assert stopped['epoch'] == 3
    stopped_summary = json.loads((resumed / 'checkpoints' / 'training_summary.json').read_text())
    assert stopped_summary['epoch'] == 3


def test_unconverged_me_does_not_silently_start_joint(tmp_path):
    data = tmp_path / 'data'
    _make_audio_dataset(data)
    output = tmp_path / 'run'
    command = _train_command(data, output, 2, 'no')
    command[command.index('--allow-unconverged-pretrain') + 1] = 'no'
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    assert result.returncode != 0
    assert 'without the validation stopping criterion' in result.stderr
    checkpoint = torch.load(output / 'checkpoints' / 'checkpoint_latest.pt', weights_only=False)
    assert checkpoint['stage'] == 'me'
    assert not (output / 'best' / 'best_model.pt').exists()


@pytest.mark.parametrize('corruption', ['length', 'overlap', 'duplicate_id', 'target_channel'])
def test_data_audit_refuses_silent_pairing_and_split_errors(tmp_path, corruption):
    _make_audio_dataset(tmp_path)
    argv = ['--train-dir', str(tmp_path / 'train'), '--val-dir', str(tmp_path / 'val')]
    if corruption == 'length':
        sf.write(tmp_path / 'train' / 'target0.wav', np.ones(802, dtype=np.float32), 16000, subtype='FLOAT')
    elif corruption == 'overlap':
        (tmp_path / 'val' / 'target0.wav').write_bytes((tmp_path / 'train' / 'target0.wav').read_bytes())
    elif corruption == 'duplicate_id':
        metadata = tmp_path / 'train' / 'metadata.csv'
        metadata.write_text(metadata.read_text().replace('1,mix1', '0,mix1'))
    else:
        argv += ['--target-ref-mic', '1']
    args = train_cts.parse_args(argv)
    datasets, _loaders, _sampler = train_cts.build_loaders(args, torch.Generator())
    with pytest.raises(ValueError):
        train_cts.make_manifest(args, datasets, 1)

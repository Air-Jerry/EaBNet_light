"""Warm-start causal TCN/Fourier training with the tested training-only PMSQE loss."""

import argparse
from pathlib import Path
import sys

import torch

import train_light_causal as causal
from experiments.causal_pmsqe_objective import PMSQEObjective
from light_candidate_runtime import (TCN_CANDIDATE,
    candidate_metadata, read_candidate_checkpoint, select_model_factory)


ROOT = Path(__file__).resolve().parent
WARM_FIELDS = ('perceptual_warmstart_checkpoint', 'perceptual_warmstart_sha256',
    'perceptual_warmstart_epoch', 'perceptual_sample_epoch_offset', 'perceptual_frequency_fourier', 'pmsqe_weight')


class FrontendPMSQE(PMSQEObjective):
    def __init__(self, n_fft, hop_length, win_length, power):
        if (n_fft, hop_length, win_length, power) != (512, 160, 320, .5):
            raise ValueError('Keep the existing PMSQE frontend')
        super().__init__(weight=.1)


def parse_args(argv=None):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('--warmstart')
    selector.add_argument('--frequency-fourier', choices=('yes', 'no'), default='no')
    selected, remaining = selector.parse_known_args(sys.argv[1:] if argv is None else argv)
    candidate = TCN_CANDIDATE
    if selected.frequency_fourier == 'yes':
        from light_candidate_runtime import FOURIER_CANDIDATE
        candidate = FOURIER_CANDIDATE
    args = causal.trainer.parse_args(remaining, defaults=dict(
        checkpoint_dir='./checkpoints_causal_pmsqe_v1', best_dir='./bestmodels_causal_pmsqe_v1',
        log_dir='./logs_causal_pmsqe_v1', num_epochs=6, learning_rate=.0003,
        batch_size=4, grad_accum_steps=1, segment_seconds=6, save_every=1,
        train_lr_patience=3, grad_clip=5., norm_type='LN', use_amp='no',
        model_amp='no', allow_tf32='no', stop_on_non_finite='yes', skip_non_finite_batches='no'))
    if (args.sample_rate, args.n_fft, args.hop_length, args.win_length, args.power,
            args.num_mics, args.target_ref_mic) != (16000, 512, 160, 320, .5, 8, 0):
        raise ValueError('Keep the existing16kHz, eight-microphone, reference0 frontend')
    if args.is_causal != 'yes' or args.norm_type != 'LN' or args.parallel_mode == 'dp':
        raise ValueError('Use causal LayerNorm with none or ddp')
    if any(value != 'no' for value in (args.use_amp, args.model_amp, args.allow_tf32)):
        raise ValueError('Keep internal BF16 activations and FP32 frontend/loss/inference')
    args.causal_training_objective = 'spectral_mrstft_waveform_pmsqe_v1'
    args.causal_training_precision = 'CUDA bfloat16 activations; FP32 parameters, frontend, loss and inference; CPU FP32'
    files = ('train_light_causal_pmsqe.py', 'train_light_causal.py', 'train_light.py',
        'light_quality_loss.py', 'experiments/causal_pmsqe_objective.py',
        'third_party/asteroid_pmsqe/pmsqe.py', 'third_party/asteroid_pmsqe/bark_matrix_16k.mat',
        'third_party/asteroid_pmsqe/PROVENANCE.json')
    args.causal_training_implementation_sha256 = {name:causal.quality.sha256_file(ROOT / name) for name in files}
    for name, value in candidate_metadata(candidate).items():
        setattr(args, name, value)
    latest = Path(args.checkpoint_dir).expanduser() / 'checkpoint_latest.pt'
    existing = None
    if latest.exists():
        if args.resume != 'yes':
            raise FileExistsError('Use --resume yes or new outputs')
        existing, _ = read_candidate_checkpoint(latest, candidate, args)
        source_name = existing['args'].get('perceptual_warmstart_checkpoint')
        if not source_name:
            raise ValueError('Resume checkpoint lacks PMSQE warm-start provenance')
        if selected.warmstart and Path(selected.warmstart).expanduser().resolve() != Path(source_name):
            raise ValueError('PMSQE warm-start differs from saved run')
    else:
        if not selected.warmstart:
            raise ValueError('A new PMSQE run requires a completed full-data causal TCN --warmstart')
        source_name = selected.warmstart
    source_path = Path(source_name).expanduser().resolve()
    source, _ = read_candidate_checkpoint(source_path, TCN_CANDIDATE)
    offset = source['args'].get('tcn_sample_epoch_offset')
    if int(source['epoch']) < 1 or isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError('Source TCN must contain completed training and a valid cumulative sampler offset')
    for name in causal.quality.SOURCE_FIELDS:
        if getattr(args, name) != source['args'][name]:
            raise ValueError(f'PMSQE frontend/model setting differs from source: {name}')
    args.perceptual_warmstart_checkpoint = str(source_path)
    args.perceptual_warmstart_sha256 = causal.quality.sha256_file(source_path)
    args.perceptual_warmstart_epoch = int(source['epoch'])
    args.perceptual_sample_epoch_offset = offset + int(source['epoch'])
    args.perceptual_frequency_fourier = selected.frequency_fourier == 'yes'
    args.pmsqe_weight = .1
    if existing is not None:
        for name in (*WARM_FIELDS, 'causal_training_objective', 'causal_training_precision',
                     'causal_training_implementation_sha256'):
            if existing['args'].get(name) != getattr(args, name):
                raise ValueError(f'PMSQE resume {name} differs')
    causal.quality._validate_output_paths(args, source_path)
    return args, source


def main(argv=None):
    args, source = parse_args(argv)
    parser, initializer, objective = causal.parse_args, causal.initialize_checkpoint, causal.QualityObjective
    sampler = causal.trainer.DistributedSampler

    def initialize(settings, unused=None):
        directory = Path(settings.checkpoint_dir)
        if (directory / 'checkpoint_latest.pt').exists():
            return
        for output in (directory, Path(settings.best_dir)):
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(f'New PMSQE run requires empty outputs: {output}')
        torch.manual_seed(settings.seed)
        with select_model_factory(causal.trainer, settings.structure_candidate):
            model = causal.trainer.create_model(settings, torch.device('cpu'))
        target, warm = model.state_dict(), source['model_state_dict']
        extra = set(target) - set(warm)
        if settings.perceptual_frequency_fourier:
            assert len(extra) == 6 and all(name.startswith('mapping_input.fourier_') for name in extra), extra
        else:
            assert not extra, extra
        assert not set(warm) - set(target)
        for name, value in warm.items():
            assert target[name].shape == value.shape and torch.isfinite(value).all(), name
            target[name] = value
        model.load_state_dict(target, strict=True)
        assert all(parameter.requires_grad for parameter in model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
        state = dict(epoch=0, best_val_loss=float('inf'), model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(), scaler_state_dict={}, lr_reducer_state={}, args=vars(settings).copy())
        for name in ('model_epoch_0.pt', 'checkpoint_latest.pt'):
            causal.trainer.save_checkpoint(directory / name, state)
        print('Causal PMSQE warm-start, exact source output, all parameters trainable, fresh Adam', flush=True)

    class AlignedSampler(sampler):
        def set_epoch(self, epoch):
            super().set_epoch(epoch + args.perceptual_sample_epoch_offset if self.shuffle else epoch)

    causal.parse_args = lambda *unused, **kwargs: args
    causal.initialize_checkpoint, causal.QualityObjective = initialize, FrontendPMSQE
    causal.trainer.DistributedSampler = AlignedSampler
    try:
        return causal.main([], candidate_id=args.structure_candidate, entry_file=__file__)
    finally:
        causal.parse_args, causal.initialize_checkpoint, causal.QualityObjective = parser, initializer, objective
        causal.trainer.DistributedSampler = sampler


if __name__ == '__main__':
    raise SystemExit(main())

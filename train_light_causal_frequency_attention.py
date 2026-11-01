"""Warm-start current-frame full-frequency attention through the unchanged causal trainer."""

import argparse
from pathlib import Path
import sys

import torch

import train_light_causal as causal
from train_light_causal_pmsqe import FrontendPMSQE
from light_candidate_runtime import (TCN_CANDIDATE, FREQUENCY_ATTENTION_CANDIDATE, RETENTION_CANDIDATE,
    candidate_metadata, read_candidate_checkpoint, select_model_factory)


ROOT = Path(__file__).resolve().parent
WARM_FIELDS = ('attention_warmstart_checkpoint', 'attention_warmstart_sha256',
    'attention_warmstart_epoch', 'attention_sample_epoch_offset', 'pmsqe_weight')


def parse_args(argv=None, candidate_id=FREQUENCY_ATTENTION_CANDIDATE, entry_file=None, training_files=()):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('--warmstart')
    selected, remaining = selector.parse_known_args(sys.argv[1:] if argv is None else argv)
    candidate = candidate_id
    args = causal.trainer.parse_args(remaining, defaults=dict(
        checkpoint_dir=f'./checkpoints_{candidate}', best_dir=f'./bestmodels_{candidate}',
        log_dir=f'./logs_{candidate}', num_epochs=6, learning_rate=.0003,
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
    if candidate == RETENTION_CANDIDATE:
        args.causal_training_precision = 'CUDA bfloat16 activations except FP32 retention; FP32 parameters, frontend, loss and inference; CPU FP32'
    files = ('train_light_causal_frequency_attention.py','train_light_causal_pmsqe.py', 'train_light_causal.py', 'train_light.py',
        'light_quality_loss.py', 'experiments/causal_pmsqe_objective.py',
        'third_party/asteroid_pmsqe/pmsqe.py', 'third_party/asteroid_pmsqe/bark_matrix_16k.mat',
        'third_party/asteroid_pmsqe/PROVENANCE.json', *training_files)
    args.causal_training_implementation_sha256 = {name:causal.quality.sha256_file(ROOT / name) for name in files}
    if entry_file is not None:
        args.causal_training_implementation_sha256[Path(entry_file).name] = causal.quality.sha256_file(entry_file)
    for name, value in candidate_metadata(candidate).items():
        setattr(args, name, value)
    latest = Path(args.checkpoint_dir).expanduser() / 'checkpoint_latest.pt'
    existing = None
    if latest.exists():
        if args.resume != 'yes':
            raise FileExistsError('Use --resume yes or new outputs')
        existing, _ = read_candidate_checkpoint(latest, candidate, args)
        source_name = existing['args'].get('attention_warmstart_checkpoint')
        if not source_name:
            raise ValueError('Resume checkpoint lacks Frequency-attention warm-start provenance')
        if selected.warmstart and Path(selected.warmstart).expanduser().resolve() != Path(source_name):
            raise ValueError('Frequency-attention warm-start differs from saved run')
    else:
        if not selected.warmstart:
            raise ValueError('A new run requires a completed full-data compatible causal --warmstart')
        source_name = selected.warmstart
    source_path = Path(source_name).expanduser().resolve()
    warm_candidate = FREQUENCY_ATTENTION_CANDIDATE if candidate == RETENTION_CANDIDATE else TCN_CANDIDATE
    source, _ = read_candidate_checkpoint(source_path, warm_candidate)
    offset = source['args'].get('attention_sample_epoch_offset',
        source['args'].get('perceptual_sample_epoch_offset', source['args'].get('tcn_sample_epoch_offset')))
    if int(source['epoch']) < 1 or isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError('Source checkpoint must contain completed training and a valid cumulative sampler offset')
    for name in causal.quality.SOURCE_FIELDS:
        if getattr(args, name) != source['args'][name]:
            raise ValueError(f'PMSQE frontend/model setting differs from source: {name}')
    args.attention_warmstart_checkpoint = str(source_path)
    args.attention_warmstart_sha256 = causal.quality.sha256_file(source_path)
    args.attention_warmstart_epoch = int(source['epoch'])
    args.attention_sample_epoch_offset = offset + int(source['epoch'])
    args.pmsqe_weight = .1
    if existing is not None:
        for name in (*WARM_FIELDS, 'causal_training_objective', 'causal_training_precision',
                     'causal_training_implementation_sha256'):
            if existing['args'].get(name) != getattr(args, name):
                raise ValueError(f'Frequency-attention resume {name} differs')
    causal.quality._validate_output_paths(args, source_path)
    return args, source


def main(argv=None, candidate_id=FREQUENCY_ATTENTION_CANDIDATE, entry_file=None, training_files=()):
    args, source = parse_args(argv, candidate_id=candidate_id, entry_file=entry_file, training_files=training_files)
    parser, initializer, objective = causal.parse_args, causal.initialize_checkpoint, causal.QualityObjective
    sampler = causal.trainer.DistributedSampler

    def initialize(settings, unused=None):
        directory = Path(settings.checkpoint_dir)
        if (directory / 'checkpoint_latest.pt').exists():
            return
        for output in (directory, Path(settings.best_dir)):
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(f'New Frequency-attention run requires empty outputs: {output}')
        torch.manual_seed(settings.seed)
        with select_model_factory(causal.trainer, settings.structure_candidate):
            model = causal.trainer.create_model(settings, torch.device('cpu'))
        target, warm = model.state_dict(), source['model_state_dict']
        extra = set(target) - set(warm)
        prefix, count = ('retention.', 9) if settings.structure_candidate == RETENTION_CANDIDATE else ('mapping_input.frequency_', 7)
        assert len(extra) == count and all(name.startswith(prefix) for name in extra), extra
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
        print('Causal Frequency-attention warm-start, exact source output, all parameters trainable, fresh Adam', flush=True)

    class AlignedSampler(sampler):
        def set_epoch(self, epoch):
            super().set_epoch(epoch + args.attention_sample_epoch_offset if self.shuffle else epoch)

    causal.parse_args = lambda *unused, **kwargs: args
    causal.initialize_checkpoint, causal.QualityObjective = initialize, FrontendPMSQE
    causal.trainer.DistributedSampler = AlignedSampler
    try:
        return causal.main([], candidate_id=args.structure_candidate, entry_file=entry_file or __file__)
    finally:
        causal.parse_args, causal.initialize_checkpoint, causal.QualityObjective = parser, initializer, objective
        causal.trainer.DistributedSampler = sampler


if __name__ == '__main__':
    raise SystemExit(main())

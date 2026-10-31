"""Warm-start the causal TCN model through the unchanged dataset and trainer."""

import argparse
from pathlib import Path
import sys

import torch

import train_light_causal as causal
from light_candidate_runtime import (RECURRENT65_CANDIDATE, TCN_CANDIDATE,
                                     read_candidate_checkpoint, select_model_factory)


WARM_FIELDS = ('tcn_warmstart_checkpoint', 'tcn_warmstart_sha256',
               'tcn_warmstart_epoch', 'tcn_sample_epoch_offset')


def aligned_sampler_type(base, offset):
    class AlignedSampler(base):
        def set_epoch(self, epoch):
            super().set_epoch(epoch + offset if self.shuffle else epoch)
    return AlignedSampler


def parse_args(argv=None):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('--warmstart')
    selected, remaining = selector.parse_known_args(sys.argv[1:] if argv is None else argv)
    args = causal.parse_args(remaining, candidate_id=TCN_CANDIDATE,
        defaults={'batch_size':4, 'grad_accum_steps':1, 'learning_rate':.0003, 'num_epochs':6}, entry_file=__file__)
    latest = Path(args.checkpoint_dir).expanduser() / 'checkpoint_latest.pt'
    existing = None
    if latest.exists():
        existing, _ = read_candidate_checkpoint(latest, TCN_CANDIDATE, args)
        saved = existing['args']
        if not all(key in saved for key in WARM_FIELDS):
            raise ValueError('TCN checkpoint lacks warm-start provenance')
        source_name = saved['tcn_warmstart_checkpoint']
        if selected.warmstart and Path(selected.warmstart).expanduser().resolve() != Path(source_name):
            raise ValueError('TCN warm-start differs from saved run')
    else:
        if not selected.warmstart:
            raise ValueError('A new TCN run requires --warmstart from a completed recurrent65 checkpoint')
        source_name = selected.warmstart
    path = Path(source_name).expanduser().resolve()
    source, _ = read_candidate_checkpoint(path, RECURRENT65_CANDIDATE)
    if int(source['epoch']) < 1:
        raise ValueError('Warm-start must contain completed real training')
    for name in causal.quality.SOURCE_FIELDS:
        if getattr(args, name) != source['args'][name]:
            raise ValueError(f'TCN frontend/model setting differs from source: {name}')
    args.tcn_warmstart_checkpoint = str(path)
    args.tcn_warmstart_sha256 = causal.quality.sha256_file(path)
    args.tcn_warmstart_epoch = int(source['epoch'])
    args.tcn_sample_epoch_offset = int(source['epoch'])
    if existing is not None:
        for name in WARM_FIELDS:
            if existing['args'][name] != getattr(args, name):
                raise ValueError(f'TCN resume {name} differs')
    for output in (args.checkpoint_dir, args.best_dir, args.log_dir):
        directory = Path(output).expanduser().resolve()
        if directory == path.parent or directory in path.parents or path.parent in directory.parents:
            raise ValueError('TCN outputs must be separate from source checkpoint')
    return args, source


def main(argv=None):
    args, source = parse_args(argv)
    parser, initializer = causal.parse_args, causal.initialize_checkpoint
    sampler = causal.trainer.DistributedSampler

    def initialize(settings, unused=None):
        directory = Path(settings.checkpoint_dir)
        if (directory / 'checkpoint_latest.pt').exists(): return
        for output in (directory, Path(settings.best_dir)):
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(f'New TCN model requires empty output: {output}')
        torch.manual_seed(settings.seed)
        with select_model_factory(causal.trainer, TCN_CANDIDATE):
            model = causal.trainer.create_model(settings, torch.device('cpu'))
        target, warm = model.state_dict(), source['model_state_dict']
        extra = set(target) - set(warm)
        assert extra and all(name.startswith('mapping_') for name in extra), extra
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
        print(f'Causal TCN warm-start epoch {settings.tcn_warmstart_epoch}; zero new mapping; fresh Adam', flush=True)

    causal.parse_args = lambda *unused, **keywords: args
    causal.initialize_checkpoint = initialize
    causal.trainer.DistributedSampler = aligned_sampler_type(sampler, args.tcn_sample_epoch_offset)
    try:
        return causal.main([], candidate_id=TCN_CANDIDATE, entry_file=__file__)
    finally:
        causal.parse_args, causal.initialize_checkpoint = parser, initializer
        causal.trainer.DistributedSampler = sampler


if __name__ == '__main__':
    raise SystemExit(main())

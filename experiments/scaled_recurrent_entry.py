"""Resume original recurrent training with explicit, checkpointed backward scaling."""
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

import torch

FACTOR = 1. / 1024


def same_tensors(first, second):
    if isinstance(first, torch.Tensor):
        return isinstance(second, torch.Tensor) and torch.equal(first.cpu(), second.cpu())
    if isinstance(first, dict):
        return isinstance(second, dict) and first.keys() == second.keys() and all(
            same_tensors(value, second[name]) for name, value in first.items())
    if isinstance(first, (list, tuple)):
        return isinstance(second, type(first)) and len(first) == len(second) and all(
            same_tensors(a, b) for a, b in zip(first, second))
    return first == second


def declare_policy(args, latest, source, source_path, implementation):
    if latest.get('diagnostic_only') or latest['args'].get('diagnostic_only'):
        raise ValueError('Diagnostic weights cannot initialize formal training')
    if source.get('diagnostic_only') or source['args'].get('diagnostic_only') or source['epoch'] < 1:
        raise ValueError('Use a completed formal source for the numerical transition')
    source_path = Path(source_path).resolve()
    policy = dict(backward_loss_factor=FACTOR,
                  backward_scaling_source_checkpoint=str(source_path),
                  backward_scaling_source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                  backward_scaling_initial_epoch=source['epoch'],
                  numerical_training_implementation_sha256=implementation)
    if 'backward_loss_factor' in latest['args']:
        if any(latest['args'].get(key) != value for key, value in policy.items()):
            raise ValueError('Saved numerical policy or implementation differs')
    elif (latest['epoch'] != source['epoch'] or not same_tensors(latest['model_state_dict'], source['model_state_dict'])
          or not same_tensors(latest['optimizer_state_dict'], source['optimizer_state_dict'])):
        raise ValueError('The initial transition must preserve the exact formal model and Adam')
    for key, value in policy.items():
        setattr(args, key, value)
    args.backward_scaling_unscale_before_weight_normalization_and_clipping = True


def main(argv=None):
    root = Path.cwd().resolve()
    sys.path.insert(0, str(root))
    import train_light_causal_recurrent_adapt as entry
    import train_light as trainer
    from light_candidate_runtime import read_candidate_checkpoint, RECURRENT_ADAPT_CANDIDATE
    if Path(entry.__file__).resolve().parent != root:
        raise ValueError('Run from the intended original training tree')
    helper = Path(__file__).with_name('scaled_backward.py')
    spec = importlib.util.spec_from_file_location('declared_scaled_backward', helper)
    scaling = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scaling)
    implementation = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        Path(__file__), helper, Path(__file__).with_name('scaled_recurrent_python.py'))}
    original = entry.parse_args

    def parse(arguments=None):
        args, donor = original(arguments)
        latest, _ = read_candidate_checkpoint(Path(args.checkpoint_dir)/'checkpoint_latest.pt', RECURRENT_ADAPT_CANDIDATE)
        source_path = Path(os.environ['CAUSAL_BACKWARD_SOURCE'])
        source, _ = read_candidate_checkpoint(source_path, RECURRENT_ADAPT_CANDIDATE)
        declare_policy(args, latest, source, source_path, implementation)
        return args, donor

    entry.parse_args = parse
    try:
        with scaling.backward_loss_scaling(trainer, FACTOR):
            return entry.main(argv)
    finally:
        entry.parse_args = original


if __name__ == '__main__':
    raise SystemExit(main())


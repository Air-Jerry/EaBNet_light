#!/usr/bin/env python3
"""Record the new PMSQE weight while retaining the original scaled training entry."""
import hashlib
import os
import runpy
import sys
from pathlib import Path


WEIGHT = .3
SOURCE_SHA = '0208a4f1fc059582f90ea4582beafa6a9609b86931158079c082f0535fecffb7'


def worker(argv):
    root = Path.cwd()
    sys.path.insert(0, str(root))
    import train_light_causal_recurrent_adapt as entry
    from light_candidate_runtime import RECURRENT_ADAPT_CANDIDATE, read_candidate_checkpoint

    scaled_file = Path(os.environ['CAUSAL_SCALED_ENTRY'])
    scaled = runpy.run_path(str(scaled_file))
    original = entry.parse_args
    trainer = entry.fullband.causal.trainer
    original_epoch = trainer.run_epoch
    objective_class = entry.fullband.FrontendPMSQE
    policy = dict(effective_pmsqe_weight=WEIGHT, validation_pmsqe_weight=.1, pmsqe_transition_source_epoch=40,
                  pmsqe_transition_source_sha256=SOURCE_SHA,
                  pmsqe_transition_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def run_epoch(*args, **kwargs):
        objective = kwargs.get('loss_fn')
        if not isinstance(objective, objective_class):
            raise ValueError('Expected the original PMSQE loss at the epoch entry')
        objective.weight = WEIGHT if kwargs['training'] else .1
        return original_epoch(*args, **kwargs)

    def parse(arguments=None):
        args, donor = original(arguments)
        ckpts = Path(args.checkpoint_dir)
        latest, _ = read_candidate_checkpoint(ckpts/'checkpoint_latest.pt', RECURRENT_ADAPT_CANDIDATE)
        if latest['epoch'] == 40:
            if hashlib.sha256((ckpts/'model_epoch_40.pt').read_bytes()).hexdigest() != SOURCE_SHA:
                raise ValueError('PMSQE transition must start from the scored formal epoch 40')
        elif latest['epoch'] < 40 or any(latest['args'].get(k) != v for k, v in policy.items()):
            raise ValueError('Saved PMSQE transition policy differs')
        for key, value in policy.items():
            setattr(args, key, value)
        print(f'Effective PMSQE weight={WEIGHT}; verified formal resume epoch={latest["epoch"]}', flush=True)
        return args, donor

    entry.parse_args = parse
    trainer.run_epoch = run_epoch
    try:
        return scaled['main'](argv)
    finally:
        entry.parse_args = original
        trainer.run_epoch = original_epoch


def main(argv):
    if argv[:1] == ['--formal-pmsqe-worker']:
        return worker(argv[1:])
    arguments = list(argv)
    if '-m' in arguments and arguments[arguments.index('-m') + 1] == 'torch.distributed.run':
        index = arguments.index('train_light_causal_recurrent_adapt.py')
        arguments[index:index+1] = [str(Path(__file__).resolve()), '--formal-pmsqe-worker']
    interpreter = os.environ['CAUSAL_REAL_PYTHON']
    os.execv(interpreter, [interpreter, *arguments])


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))

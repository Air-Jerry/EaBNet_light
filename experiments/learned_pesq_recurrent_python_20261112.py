#!/usr/bin/env python3
"""Add frozen PESQ supervision to the original scaled training entry."""
import hashlib
import os
import runpy
import sys
from pathlib import Path


WEIGHT = .6
SOURCE_SHA = '2234118d90c8f31c93168a154e648fefca31f1b103e59dd5ca49e1471ca3ed7f'


def worker(argv):
    root = Path.cwd()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    import train_light_causal_recurrent_adapt as entry
    from experiments.bark_pesq_regressor import BarkPESQRegressor, bark_score_features
    from light_candidate_runtime import RECURRENT_ADAPT_CANDIDATE, read_candidate_checkpoint

    scaled_file = Path(os.environ['CAUSAL_SCALED_ENTRY'])
    scaled = runpy.run_path(str(scaled_file))
    original = entry.parse_args
    trainer = entry.fullband.causal.trainer
    original_epoch = trainer.run_epoch
    objective_class = entry.fullband.FrontendPMSQE
    regressor_path = Path(__file__).with_name('score_regressor_delta.pt')
    regressor_sha = hashlib.sha256(regressor_path.read_bytes()).hexdigest()
    if regressor_sha != 'e2875e0764516b7cfb82530414918d907bb30fa44928fa10d3322a38c5bdaa9e':
        raise ValueError('Unexpected frozen score regressor')
    saved = torch.load(regressor_path, map_location='cpu', weights_only=True)
    if not saved['training_only']:
        raise ValueError('Expected the training-only score regressor')
    implementation = Path(__file__).resolve().parent/'experiments/bark_pesq_regressor.py'
    policy = dict(learned_pesq_weight=.15, learned_pesq_regressor_sha256=regressor_sha,
                  learned_pesq_regressor_implementation_sha256=hashlib.sha256(implementation.read_bytes()).hexdigest(),
                  learned_pesq_slope=float(saved['slope']), effective_pmsqe_weight=WEIGHT,
                  validation_pmsqe_weight=.1, pmsqe_transition_source_epoch=46,
                  pmsqe_transition_source_sha256=SOURCE_SHA,
                  pmsqe_transition_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    class LearnedPMSQE(objective_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.regressor = BarkPESQRegressor().eval().requires_grad_(False)
            self.regressor.load_state_dict(saved['model_state_dict'])
            self.use_learned_score = False

        def __call__(self, estimate, target_stft, frame_list):
            result = super().__call__(estimate, target_stft, frame_list)
            if not self.use_learned_score:
                return result
            if any(int(length) != 601 for length in frame_list):
                raise ValueError('PESQ regression uses the six-second training frontend')
            if next(self.regressor.parameters()).device != estimate.device:
                self.regressor.to(estimate.device)
            with torch.autocast(estimate.device.type, enabled=False):
                score = self.regressor(bark_score_features(estimate, target_stft, self.perceptual.bark_matrix))
                return result + .15 * float(saved['slope']) * (4.5 - score).mean()

    def run_epoch(*args, **kwargs):
        objective = kwargs.get('loss_fn')
        if not isinstance(objective, LearnedPMSQE):
            raise ValueError('Expected the original PMSQE loss at the epoch entry')
        objective.weight = WEIGHT if kwargs['training'] else .1
        objective.use_learned_score = kwargs['training']
        return original_epoch(*args, **kwargs)

    def parse(arguments=None):
        args, donor = original(arguments)
        ckpts = Path(args.checkpoint_dir)
        latest, _ = read_candidate_checkpoint(ckpts/'checkpoint_latest.pt', RECURRENT_ADAPT_CANDIDATE)
        if latest['epoch'] == 46:
            if hashlib.sha256((ckpts/'model_epoch_46.pt').read_bytes()).hexdigest() != SOURCE_SHA:
                raise ValueError('PMSQE transition must start from the scored formal epoch 46')
        elif latest['epoch'] < 46 or any(latest['args'].get(k) != v for k, v in policy.items()):
            raise ValueError('Saved PMSQE transition policy differs')
        for key, value in policy.items():
            setattr(args, key, value)
        print(f'Effective PMSQE weight={WEIGHT}, learned PESQ weight=0.15; verified formal resume epoch={latest["epoch"]}', flush=True)
        return args, donor

    entry.parse_args = parse
    trainer.run_epoch = run_epoch
    entry.fullband.FrontendPMSQE = LearnedPMSQE
    try:
        return scaled['main'](argv)
    finally:
        entry.parse_args = original
        trainer.run_epoch = original_epoch
        entry.fullband.FrontendPMSQE = objective_class


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

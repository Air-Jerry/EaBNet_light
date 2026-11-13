"""Continue a formal recurrent run with a zero-initialized causal phase filter."""
import argparse
import copy
import hashlib
import json
import runpy
from pathlib import Path

import torch

import train_light_causal_recurrent_adapt as entry
from experiments.bark_phase_pesq_regressor import BarkPhasePESQRegressor, bark_phase_score_features
from light_candidate_runtime import (
    PHASE_FILTER_CANDIDATE, RECURRENT_ADAPT_CANDIDATE, build_model_candidate,
    candidate_metadata, read_candidate_checkpoint,
)


ROOT = Path(__file__).resolve().parent
REGRESSOR_SHA256 = 'a2fcafaa5deb6c7666c7f28377f0380b893b0f1f6afc105ae16455fd1c5745dd'
FILTER_LR = 1e-4
trainer = entry.fullband.causal.trainer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def phase_optimizer(parameters, **kwargs):
    parameters = list(parameters)
    return torch.optim.Adam([dict(params=parameters[:-6]),
                             dict(params=parameters[-6:], lr=FILTER_LR)], **kwargs)


def initialize(source_path, checkpoint_dir, best_dir, log_dir):
    source_path = source_path.resolve()
    source, _ = read_candidate_checkpoint(source_path, RECURRENT_ADAPT_CANDIDATE)
    saved = source['args']
    epoch = int(source['epoch'])
    if epoch < 1 or source.get('diagnostic_only') or saved.get('diagnostic_only'):
        raise ValueError('Use a completed formal recurrent checkpoint')
    report_path = Path(saved['log_dir']).expanduser()
    if not report_path.is_absolute():
        report_path = source_path.parent.parent/report_path
    from audit_light_final import require_protocol
    report = json.loads((report_path/f'pesq_selection/model_epoch_{epoch}/summary.json').read_text())
    require_protocol(report)
    source_digest = sha(source_path)
    timings = (json.loads(line) for line in (report_path/'epoch_timing.jsonl').read_text().splitlines() if line)
    timing = next(row for row in timings if row['epoch'] == epoch)
    if report['checkpoint_sha256'] != source_digest or (
        timing['status'], timing['training_samples'], timing['validation_samples'],
        timing['full_metric_samples'], timing['optimizer_updates_this_epoch'], timing['global_batch_size']
    ) != ('complete', 53600, 600, 600, 6700, 8) or not 0 < timing['total_seconds'] <= 3600:
        raise ValueError('Use a scored formal checkpoint with a complete on-budget epoch')
    outputs = dict(checkpoint_dir=checkpoint_dir.resolve(), best_dir=best_dir.resolve(), log_dir=log_dir.resolve())
    settings = argparse.Namespace(**{**saved, **outputs})
    entry.fullband.causal.quality._validate_output_paths(settings, source_path)
    if any(p.exists() for p in outputs.values()):
        raise FileExistsError('Use new output directories for the phase-filter transition')
    torch.manual_seed(saved['seed'])
    model = build_model_candidate(PHASE_FILTER_CANDIDATE)
    model.load_warm(source['model_state_dict'])
    old = source['optimizer_state_dict']
    assert len(old['param_groups']) == 1
    names = [name for name, _ in model.named_parameters()]
    assert sum(name.startswith('phase_filter.') for name in names) == 6
    assert all(name.startswith('phase_filter.') for name in names[-6:])
    assert old['param_groups'][0]['params'] == list(range(len(names)-6))
    assert len(old['state']) == 235 and {int(x['step']) for x in old['state'].values()} == {epoch*6700}
    optimizer = phase_optimizer(model.parameters(), lr=saved['learning_rate'], weight_decay=saved['weight_decay'])
    new_group = optimizer.state_dict()['param_groups'][1]
    new_group.update(copy.deepcopy(old['param_groups'][0]), params=list(range(len(names)-6,len(names))), lr=FILTER_LR)
    state = dict(source, model_state_dict=model.state_dict(),
                 optimizer_state_dict=dict(state=old['state'],param_groups=[old['param_groups'][0],new_group]))
    state['args'] = dict(saved, **candidate_metadata(PHASE_FILTER_CANDIDATE))
    state['args'].update({k:str(v) for k,v in outputs.items()}, phase_filter_source_epoch=epoch,
        phase_filter_source_sha256=source_digest, phase_filter_training_entry_sha256=sha(__file__),
        phase_filter_learning_rate=FILTER_LR)
    for path in outputs.values():
        path.mkdir(parents=True)
    trainer.save_checkpoint(outputs['checkpoint_dir']/'checkpoint_latest.pt',state)


def train(checkpoint_dir, num_epochs, regressor, scaling_helper):
    latest, _ = read_candidate_checkpoint(checkpoint_dir/'checkpoint_latest.pt', PHASE_FILTER_CANDIDATE)
    saved = latest['args']
    if Path(saved['checkpoint_dir']).resolve() != checkpoint_dir.resolve():
        raise ValueError('Checkpoint does not belong to this run directory')
    if latest.get('diagnostic_only') or saved.get('diagnostic_only') or latest['epoch']+1 != num_epochs:
        raise ValueError('Resume the next epoch from the latest formal checkpoint')
    if sha(__file__) != saved['phase_filter_training_entry_sha256'] or sha(regressor) != REGRESSOR_SHA256:
        raise ValueError('Phase-filter training code or score weights differ')
    if sha(ROOT/'experiments/bark_phase_pesq_regressor.py') != saved['learned_pesq_regressor_implementation_sha256']:
        raise ValueError('Training score features differ from the source run')
    helpers = saved['numerical_training_implementation_sha256']
    if helpers.get(str(scaling_helper.resolve())) != sha(scaling_helper):
        raise ValueError('Use the original checkpointed backward scaling helper')
    for name, digest in saved['causal_training_implementation_sha256'].items():
        if name != 'light_candidate_runtime.py' and sha(ROOT/name) != digest:
            raise ValueError(f'Original training implementation differs: {name}')
    settings = argparse.Namespace(**dict(saved, num_epochs=num_epochs, resume='yes', resume_reset_lr='no'))
    score = torch.load(regressor,map_location='cpu',weights_only=True)
    assert score['training_only'] and float(score['slope']) == saved['learned_pesq_slope']
    assert saved['learned_pesq_regressor_sha256'] == REGRESSOR_SHA256
    original_parser, original_adam = entry.parse_args, trainer.Adam
    original_epoch, original_objective = trainer.run_epoch, entry.fullband.FrontendPMSQE
    scaling = runpy.run_path(str(scaling_helper))['backward_loss_scaling']

    class PhaseObjective(original_objective):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.regressor = BarkPhasePESQRegressor().eval().requires_grad_(False)
            self.regressor.load_state_dict(score['model_state_dict'])
            self.training_score = False

        def __call__(self, estimate, target, frame_lengths):
            loss = super().__call__(estimate, target, frame_lengths)
            if not self.training_score:
                return loss
            if any(int(length) != 601 for length in frame_lengths):
                raise ValueError('PESQ regression uses the six-second training frontend')
            if next(self.regressor.parameters()).device != estimate.device:
                self.regressor.to(estimate.device)
            with torch.autocast(estimate.device.type, enabled=False):
                features = bark_phase_score_features(estimate,target,self.perceptual.bark_matrix)
                return loss + saved['learned_pesq_weight']*float(score['slope'])*(4.5-self.regressor(features)).mean()

    def run_epoch(*args, **kwargs):
        objective = kwargs['loss_fn']
        objective.training_score = kwargs['training']
        objective.weight = saved['effective_pmsqe_weight'] if kwargs['training'] else saved['validation_pmsqe_weight']
        return original_epoch(*args, **kwargs)

    entry.parse_args = lambda *unused: (settings, None)
    trainer.Adam = phase_optimizer
    trainer.run_epoch = run_epoch
    entry.fullband.FrontendPMSQE = PhaseObjective
    try:
        with scaling(trainer,saved['backward_loss_factor']):
            return entry.main([])
    finally:
        entry.parse_args, trainer.Adam = original_parser, original_adam
        trainer.run_epoch, entry.fullband.FrontendPMSQE = original_epoch, original_objective


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='mode', required=True)
    prepare = commands.add_parser('initialize')
    for name in ('source-path','checkpoint-dir','best-dir','log-dir'):
        prepare.add_argument(f'--{name}', type=Path, required=True)
    resume = commands.add_parser('train')
    for name in ('checkpoint-dir','regressor','scaling-helper'):
        resume.add_argument(f'--{name}', type=Path, required=True)
    resume.add_argument('--num-epochs', type=int, required=True)
    arguments = vars(parser.parse_args())
    mode = arguments.pop('mode')
    torch.set_num_threads(2)
    raise SystemExit((initialize if mode == 'initialize' else train)(**arguments))

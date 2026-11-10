import copy
from types import SimpleNamespace

import pytest
import torch

from experiments.scaled_recurrent_entry import declare_policy, FACTOR
from experiments.scaled_recurrent_python import training_arguments


def checkpoint():
    return dict(epoch=16, args={}, model_state_dict={'weight': torch.tensor([.25])},
                optimizer_state_dict={'state': {0: {'step': torch.tensor(107200.),
                                                 'exp_avg': torch.tensor([.1])}}})


def test_exact_transition_and_saved_policy_resume_preserve_source(tmp_path):
    source = checkpoint()
    before = copy.deepcopy(source)
    path = tmp_path/'formal.pt'
    torch.save(source, path)
    args = SimpleNamespace()
    declare_policy(args, copy.deepcopy(source), source, path, {'helper.py': 'exact'})
    assert args.backward_loss_factor == FACTOR
    assert source['args'] == before['args'] == {}
    assert torch.equal(source['model_state_dict']['weight'], before['model_state_dict']['weight'])
    later = copy.deepcopy(source)
    later['epoch'] = 17
    later['args'] = vars(args).copy()
    declare_policy(SimpleNamespace(), later, source, path, {'helper.py': 'exact'})
    with pytest.raises(ValueError, match='implementation differs'):
        declare_policy(SimpleNamespace(), later, source, path, {'helper.py': 'changed'})


@pytest.mark.parametrize('fault', ['model', 'adam', 'epoch', 'diagnostic_latest', 'diagnostic_source'])
def test_wrong_initial_transition_or_diagnostic_source_rejected(tmp_path, fault):
    source = checkpoint()
    latest = copy.deepcopy(source)
    if fault == 'model': latest['model_state_dict']['weight'].add_(.1)
    if fault == 'adam': latest['optimizer_state_dict']['state'][0]['exp_avg'].add_(.1)
    if fault == 'epoch': latest['epoch'] = 15
    if fault == 'diagnostic_latest': latest['diagnostic_only'] = True
    if fault == 'diagnostic_source': source['args']['diagnostic_only'] = True
    path = tmp_path/'formal.pt'
    torch.save(source, path)
    with pytest.raises(ValueError):
        declare_policy(SimpleNamespace(), latest, source, path, {'helper.py': 'exact'})


def test_proxy_preserves_evaluation_and_all_original_training_arguments():
    evaluate = ['-u', 'evaluate_light.py', '--checkpoint', 'train_light_causal_recurrent_adapt.py']
    assert training_arguments(evaluate, '/overlay/entry.py') == evaluate
    train = ['-u', '-m', 'torch.distributed.run', '--nproc_per_node=2',
             'train_light_causal_recurrent_adapt.py', '--batch-size', '2', '--resume', 'yes']
    expected = train.copy()
    expected[4] = '/overlay/entry.py'
    assert training_arguments(train, '/overlay/entry.py') == expected
    assert train[4] == 'train_light_causal_recurrent_adapt.py'
    with pytest.raises(ValueError):
        training_arguments(['-m', 'torch.distributed.run', 'other_training.py'], '/overlay/entry.py')

"""Keep the formal source Adam while only appended attention parameters start empty."""
import copy

import torch


def restore_joint_attention_adam(optimizer, model, original, source):
    if source.get('diagnostic_only') or source.get('args', {}).get('diagnostic_only') or source['epoch'] < 1:
        raise ValueError('Use a completed formal source')
    old, new = list(original.named_parameters()), list(model.named_parameters())
    assert len(new) - len(old) in (0, 4)
    assert all(name == old_name and p.shape == old_p.shape and p.requires_grad == old_p.requires_grad
               for (name, p), (old_name, old_p) in zip(new[:len(old)], old))
    assert all(name.startswith('temporal_memory.') and p.requires_grad for name, p in new[len(old):])
    assert all(torch.equal(p.detach().cpu(), source['model_state_dict'][name])
               for name, p in new[:len(old)])
    stored = source['optimizer_state_dict']
    assert len(stored['param_groups']) == len(optimizer.param_groups) == 1
    assert stored['param_groups'][0]['params'] == list(range(len(old)))
    assert set(stored['state']) == {index for index, (_, p) in enumerate(old) if p.requires_grad}
    assert {int(v['step']) for v in stored['state'].values()} == {6700 * source['epoch']}
    assert all(torch.isfinite(v).all() for item in stored['state'].values() for v in item.values())
    group = copy.deepcopy(stored['param_groups'][0])
    group['params'] = list(range(len(new)))
    optimizer.load_state_dict(dict(state=copy.deepcopy(stored['state']), param_groups=[group]))
    actual = optimizer.state_dict()
    assert all(torch.equal(v.cpu(), actual['state'][index][key].cpu())
               for index, item in stored['state'].items() for key, v in item.items())
    assert all(not optimizer.state.get(p) for _, p in new[len(old):])
    return dict(existing_adam_moments_exactly_preserved=True, source_adam_steps=6700 * source['epoch'],
                existing_state_parameters=len(stored['state']), new_attention_parameter_tensors=len(new) - len(old))

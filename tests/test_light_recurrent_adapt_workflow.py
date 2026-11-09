"""Original-loop recurrent adaptation preserves formal-source, frozen-state and resume contracts."""
from pathlib import Path

import pytest
import torch

import evaluate_light
import train_light
import train_light_causal_recurrent_adapt as recurrent
from experiments.causal_recurrent_adapt_monaural import recurrent_root_editable
from light_candidate_runtime import (DECODER_MONAURAL_CANDIDATE, RECURRENT_ADAPT_CANDIDATE,
    build_model_candidate, candidate_metadata, read_candidate_checkpoint)
from test_light_causal_workflow import argv
from test_light_quality_workflow import run_cli


def source_checkpoint(arguments, tmp_path, diagnostic=False):
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    settings = train_light.parse_args(arguments, defaults=dict(norm_type='LN'))
    source = build_model_candidate(DECODER_MONAURAL_CANDIDATE)
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            if name.startswith(('base.base.beamformer.', 'base.base.mapping_output.',
                                'base.base.retention.output.', 'base.output.', 'output.')):
                parameter.normal_(std=.003)
    values = vars(settings)
    values.update(candidate_metadata(DECODER_MONAURAL_CANDIDATE))
    values['decoder_monaural_sample_epoch_offset'] = 74
    path = tmp_path/'source30'/'model_epoch_2.pt'
    path.parent.mkdir()
    torch.save(dict(epoch=2, args=values, model_state_dict=source.state_dict(), diagnostic_only=diagnostic), path)
    return path, source.eval()


def test_recurrent_original_loop_initialization_learning_resume_and_frozen_shared_matrix(argv, tmp_path):
    previous = candidate_metadata(DECODER_MONAURAL_CANDIDATE)
    source_path, source = source_checkpoint(argv, tmp_path)
    directory = Path(argv[argv.index('--checkpoint-dir')+1])
    entry = 'train_light_causal_recurrent_adapt.py'
    run_cli(entry, [*argv, '--warmstart', str(source_path), '--num-epochs', '0'])
    initial, _ = read_candidate_checkpoint(directory/'model_epoch_0.pt', RECURRENT_ADAPT_CANDIDATE)
    assert initial['args']['candidate_schema'] == 31
    assert initial['args']['recurrent_adapt_sample_epoch_offset'] == 76
    assert initial['args']['cudnn_training_benchmark'] is True
    assert initial['args']['candidate_evidence']['author_confirmed'] is False
    assert initial['args']['candidate_evidence']['model_config']['trainable_parameters'] == 5101386
    assert not initial['optimizer_state_dict']['state']
    assert all(torch.equal(value, initial['model_state_dict'][name]) for name, value in source.state_dict().items())
    restored, _ = evaluate_light.load_model(directory/'model_epoch_0.pt', torch.device('cpu'), RECURRENT_ADAPT_CANDIDATE)
    value = torch.randn(1, 17, 257, 8, 2)*.02
    with torch.inference_mode():
        assert torch.equal(restored(value), source(value))
    assert sum(p.numel() for p in restored.parameters() if p.requires_grad) == 5101386
    assert all(p.requires_grad == recurrent_root_editable(name) for name, p in restored.named_parameters()
               if name.startswith('base.base.'))
    frozen = {name for name in initial['model_state_dict']
              if name.startswith('base.base.') and not recurrent_root_editable(name)}
    shared = {name for name in frozen if name.startswith('base.base.layers.3.full.')}
    assert shared and 'base.base.stem.conv.weight' in frozen
    for epoch in (1, 2):
        run_cli(entry, [*argv, '--num-epochs', str(epoch)])
        state, _ = read_candidate_checkpoint(directory/'checkpoint_latest.pt', RECURRENT_ADAPT_CANDIDATE)
        assert state['epoch'] == epoch
        assert {int(item['step']) for item in state['optimizer_state_dict']['state'].values()} == {epoch*2}
        assert all(torch.isfinite(value).all() for value in state['model_state_dict'].values())
        assert all(torch.isfinite(value).all() for item in state['optimizer_state_dict']['state'].values()
                   for value in item.values())
        assert all(torch.equal(state['model_state_dict'][name], initial['model_state_dict'][name]) for name in frozen)
    assert previous == candidate_metadata(DECODER_MONAURAL_CANDIDATE)
    from check_light_causal_cuda import verify_recurrent_adaptation
    report = verify_recurrent_adaptation(state['model_state_dict'], initial['model_state_dict'])
    assert report['changed_causal_recurrent_state_tensors'] > 0
    for name in ('base.base.stem.conv.weight', sorted(shared)[0]):
        broken = {key: value.clone() for key, value in state['model_state_dict'].items()}
        broken[name].add_(.01)
        with pytest.raises(AssertionError):
            verify_recurrent_adaptation(broken, initial['model_state_dict'])
    with pytest.raises(AssertionError):
        verify_recurrent_adaptation(initial['model_state_dict'], initial['model_state_dict'])
    source_state = torch.load(source_path, map_location='cpu', weights_only=False)
    source_state['model_state_dict']['output.bias'].add_(.01)
    torch.save(source_state, source_path)
    with pytest.raises(ValueError, match='warmstart_sha256 differs'):
        recurrent.parse_args([*argv, '--num-epochs', '3'])
    assert read_candidate_checkpoint(directory/'checkpoint_latest.pt', RECURRENT_ADAPT_CANDIDATE)[0]['epoch'] == 2


def test_diagnostic_source_cannot_initialize_formal_recurrent_run(argv, tmp_path):
    path, _ = source_checkpoint(argv, tmp_path, diagnostic=True)
    with pytest.raises(ValueError, match='completed formal decoder'):
        recurrent.parse_args([*argv, '--warmstart', str(path)])


def test_registered_recurrent_entry_has_full_launcher_objective_and_resume_routes():
    import ast
    root = Path(__file__).resolve().parents[1]
    shell = (root/'run_light_causal.sh').read_text()
    wrapper = (root/'run_light_causal_recurrent_adapt.sh').read_text()
    assert 'export CAUSAL_MODEL_CANDIDATE='+RECURRENT_ADAPT_CANDIDATE in wrapper
    for tag in ('PY_PREFLIGHT', 'PY_RESULTS'):
        body = shell.split("<<'"+tag+"'", 1)[1].split('\n'+tag, 1)[0]
        for node in ast.walk(ast.parse(body)):
            if isinstance(node, ast.Tuple):
                values = {item.value for item in node.elts if isinstance(item, ast.Constant)}
                if DECODER_MONAURAL_CANDIDATE in values:
                    assert RECURRENT_ADAPT_CANDIDATE in values
    for line in shell.splitlines():
        if '[[ "$candidate"' in line and DECODER_MONAURAL_CANDIDATE in line:
            assert RECURRENT_ADAPT_CANDIDATE in line


@pytest.mark.parametrize('flags, expected', [(['--target-ref-mic', '1'], 'reference0'),
    (['--model-amp', 'yes'], 'FP32'), (['--skip-non-finite-batches', 'yes'], 'Nonfinite')])
def test_recurrent_preserves_frontend_precision_and_finite_update_guards(argv, flags, expected):
    with pytest.raises(ValueError, match=expected):
        recurrent.parse_args([*argv, *flags])


@pytest.mark.parametrize('fault', [None, 'diagnostic', 'hash', 'epoch', 'sampler', 'precision',
    'autotune', 'frozen', 'unchanged_recurrence', 'rule'])
def test_formal_launcher_requires_completed_source_and_actual_recurrent_changes(tmp_path, monkeypatch, fault):
    import hashlib
    import sys
    import textwrap
    source = tmp_path/'source.pt'
    source.write_bytes(b'formal completed decoder source')
    state = dict(args=dict(recurrent_adapt_warmstart_checkpoint=str(source),
        recurrent_adapt_warmstart_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        recurrent_adapt_warmstart_epoch=7, recurrent_adapt_sample_epoch_offset=81,
        gru_training_precision='fp16', cudnn_training_benchmark=True,
        candidate_evidence=dict(partially_frozen_backbone=True)))
    parent = dict(epoch=7, args=dict(decoder_monaural_sample_epoch_offset=74), diagnostic_only=False)
    check = dict(frozen_causal_weights_unchanged=True, changed_causal_recurrent_state_tensors=35,
        frozen_causal_parameter_rule='Freeze root stem,encoders,first three recurrent blocks and shared frequency matrix')
    if fault == 'diagnostic': parent['diagnostic_only'] = True
    if fault == 'hash': source.write_bytes(b'changed')
    if fault == 'epoch': parent['epoch'] = 6
    if fault == 'sampler': parent['args']['decoder_monaural_sample_epoch_offset'] = 73
    if fault == 'precision': state['args']['gru_training_precision'] = 'fp32'
    if fault == 'autotune': state['args']['cudnn_training_benchmark'] = False
    if fault == 'frozen': check['frozen_causal_weights_unchanged'] = False
    if fault == 'unchanged_recurrence': check['changed_causal_recurrent_state_tensors'] = 0
    if fault == 'rule': check['frozen_causal_parameter_rule'] = 'base.'
    shell = (Path(__file__).resolve().parents[1]/'run_light_causal.sh').read_text()
    marker = "    elif sys.argv[3]=='causal_recurrent_adapt_monaural_v1':"
    body = marker.replace('elif', 'if', 1)+shell.split(marker, 1)[1].split('    projection=json.loads', 1)[0]
    monkeypatch.setattr(sys, 'argv', ['-', 'preflight', 'validation', RECURRENT_ADAPT_CANDIDATE])
    def read(path, candidate):
        assert candidate == DECODER_MONAURAL_CANDIDATE
        return parent, candidate
    namespace = dict(Path=Path, sys=sys, hashlib=hashlib, state=state, check=check, read_candidate_checkpoint=read)
    if fault:
        with pytest.raises(AssertionError):
            exec(compile(textwrap.dedent(body), 'recurrent source qualification', 'exec'), namespace)
    else:
        exec(compile(textwrap.dedent(body), 'recurrent source qualification', 'exec'), namespace)

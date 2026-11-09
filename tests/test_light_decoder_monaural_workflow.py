"""Real original-loop warm initialization, Adam resume and formal-source rejection."""
from pathlib import Path

import pytest
import torch

import evaluate_light
import train_light
import train_light_causal_decoder_monaural as decoder
from light_candidate_runtime import (DECODER_MONAURAL_CANDIDATE, MONAURAL_SPECTRAL_CANDIDATE,
    build_model_candidate, candidate_metadata, read_candidate_checkpoint)
from test_light_causal_workflow import argv
from test_light_quality_workflow import run_cli


def source_checkpoint(arguments, tmp_path, diagnostic=False):
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    settings = train_light.parse_args(arguments, defaults=dict(norm_type='LN'))
    source = build_model_candidate(MONAURAL_SPECTRAL_CANDIDATE)
    with torch.no_grad():
        for module in (source.base.base.beamformer, source.base.output, source.output):
            module.weight.normal_(std=.003)
    values = vars(settings)
    values.update(candidate_metadata(MONAURAL_SPECTRAL_CANDIDATE))
    values['monaural_sample_epoch_offset'] = 74
    path = tmp_path/'source27'/'model_epoch_2.pt'
    path.parent.mkdir()
    torch.save(dict(epoch=2, args=values, model_state_dict=source.state_dict(), diagnostic_only=diagnostic), path)
    return path, source.eval()


def test_registered_decoder_original_loop_warm_inference_learning_and_resume(argv, tmp_path):
    metadata = candidate_metadata(MONAURAL_SPECTRAL_CANDIDATE)
    source_path, source = source_checkpoint(argv, tmp_path)
    directory = Path(argv[argv.index('--checkpoint-dir')+1])
    entry = 'train_light_causal_decoder_monaural.py'
    run_cli(entry, [*argv, '--warmstart', str(source_path), '--num-epochs', '0'])
    initial, _ = read_candidate_checkpoint(directory/'model_epoch_0.pt', DECODER_MONAURAL_CANDIDATE)
    assert initial['args']['candidate_schema'] == 30
    assert initial['args']['decoder_monaural_sample_epoch_offset'] == 76
    assert initial['args']['cudnn_training_benchmark'] is True
    assert initial['args']['candidate_evidence']['author_confirmed'] is False
    assert initial['args']['candidate_evidence']['model_config']['trainable_parameters'] == 3191840
    assert not initial['optimizer_state_dict']['state']
    assert all(torch.equal(value, initial['model_state_dict'][name]) for name, value in source.state_dict().items())
    restored, _ = evaluate_light.load_model(directory/'model_epoch_0.pt', torch.device('cpu'), DECODER_MONAURAL_CANDIDATE)
    value = torch.randn(1, 17, 257, 8, 2)*.02
    with torch.inference_mode():
        assert torch.equal(restored(value), source(value))
    frozen = {name for name, parameter in restored.named_parameters() if not parameter.requires_grad}
    assert frozen and 'base.base.stem.conv.weight' in frozen
    for epoch in (1, 2):
        run_cli(entry, [*argv, '--num-epochs', str(epoch)])
        state, _ = read_candidate_checkpoint(directory/'checkpoint_latest.pt', DECODER_MONAURAL_CANDIDATE)
        assert state['epoch'] == epoch
        assert {int(item['step']) for item in state['optimizer_state_dict']['state'].values()} == {epoch*2}
        assert all(torch.isfinite(value).all() for value in state['model_state_dict'].values())
        assert all(torch.isfinite(value).all() for item in state['optimizer_state_dict']['state'].values() for value in item.values())
        assert all(torch.equal(state['model_state_dict'][name], initial['model_state_dict'][name]) for name in frozen)
        assert not torch.equal(state['model_state_dict']['base.base.beamformer.weight'], initial['model_state_dict']['base.base.beamformer.weight'])
    assert metadata == candidate_metadata(MONAURAL_SPECTRAL_CANDIDATE)
    from check_light_causal_cuda import verify_decoder_adaptation
    report = verify_decoder_adaptation(state['model_state_dict'], initial['model_state_dict'])
    assert report['changed_causal_decoder_state_tensors'] > 0
    broken = {name: value.clone() for name, value in state['model_state_dict'].items()}
    broken['base.base.stem.conv.weight'].add_(.01)
    with pytest.raises(AssertionError):
        verify_decoder_adaptation(broken, initial['model_state_dict'])
    source_state = torch.load(source_path, map_location='cpu', weights_only=False)
    source_state['model_state_dict']['output.bias'].add_(.01)
    torch.save(source_state, source_path)
    with pytest.raises(ValueError, match='warmstart_sha256 differs'):
        decoder.parse_args([*argv, '--num-epochs', '3'])
    assert read_candidate_checkpoint(directory/'checkpoint_latest.pt', DECODER_MONAURAL_CANDIDATE)[0]['epoch'] == 2


def test_diagnostic_weights_cannot_initialize_formal_decoder(argv, tmp_path):
    path, _ = source_checkpoint(argv, tmp_path, diagnostic=True)
    with pytest.raises(ValueError, match='completed monaural'):
        decoder.parse_args([*argv, '--warmstart', str(path)])


@pytest.mark.parametrize('flags, expected', [(['--target-ref-mic', '1'], 'reference0'),
    (['--model-amp', 'yes'], 'FP32'), (['--skip-non-finite-batches', 'yes'], 'Nonfinite')])
def test_decoder_keeps_frontend_precision_and_finite_update_guards(argv, flags, expected):
    with pytest.raises(ValueError, match=expected):
        decoder.parse_args([*argv, *flags])


@pytest.mark.parametrize('fault', [None, 'diagnostic', 'hash', 'epoch', 'sampler', 'precision', 'autotune', 'frozen', 'unchanged_decoder', 'rule'])
def test_formal_launcher_binds_completed_source_and_actual_decoder_changes(tmp_path, monkeypatch, fault):
    import hashlib
    import sys
    import textwrap
    source = tmp_path/'source.pt'
    source.write_bytes(b'formal completed source')
    state = dict(args=dict(decoder_monaural_warmstart_checkpoint=str(source),
        decoder_monaural_warmstart_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        decoder_monaural_warmstart_epoch=28, decoder_monaural_sample_epoch_offset=74,
        gru_training_precision='fp16', cudnn_training_benchmark=True,
        candidate_evidence=dict(partially_frozen_backbone=True)))
    parent = dict(epoch=28, args=dict(monaural_sample_epoch_offset=46), diagnostic_only=False)
    check = dict(frozen_causal_weights_unchanged=True, changed_causal_decoder_state_tensors=16,
        frozen_causal_parameter_rule='All base.base parameters except decoders,upsamplers,beamformer')
    if fault == 'diagnostic': parent['diagnostic_only'] = True
    if fault == 'hash': source.write_bytes(b'changed')
    if fault == 'epoch': parent['epoch'] = 27
    if fault == 'sampler': parent['args']['monaural_sample_epoch_offset'] = 45
    if fault == 'precision': state['args']['gru_training_precision'] = 'fp32'
    if fault == 'autotune': state['args']['cudnn_training_benchmark'] = False
    if fault == 'frozen': check['frozen_causal_weights_unchanged'] = False
    if fault == 'unchanged_decoder': check['changed_causal_decoder_state_tensors'] = 0
    if fault == 'rule': check['frozen_causal_parameter_rule'] = 'base.'
    shell = (Path(__file__).resolve().parents[1]/'run_light_causal.sh').read_text()
    marker = "    if sys.argv[3]=='causal_decoder_monaural_spectral_v1':"
    body = marker+shell.split(marker, 1)[1].split('    projection=json.loads', 1)[0]
    monkeypatch.setattr(sys, 'argv', ['-', 'preflight', 'validation', DECODER_MONAURAL_CANDIDATE])
    def read(path, candidate):
        assert candidate == MONAURAL_SPECTRAL_CANDIDATE
        return parent, candidate
    namespace = dict(Path=Path, sys=sys, hashlib=hashlib, state=state, check=check, read_candidate_checkpoint=read)
    if fault:
        with pytest.raises(AssertionError):
            exec(compile(textwrap.dedent(body), 'decoder source qualification', 'exec'), namespace)
    else:
        exec(compile(textwrap.dedent(body), 'decoder source qualification', 'exec'), namespace)


@pytest.mark.parametrize('fault', [None, 'tf32', 'source', 'pairs', 'partial', 'evaluator', 'summary'])
def test_same_backend_source_reference_preserves_source_data_and_full600(tmp_path, monkeypatch, fault):
    import copy
    import hashlib
    import json
    import textwrap
    root = Path(__file__).resolve().parents[1]
    shell = (root/'run_light_causal.sh').read_text()
    marker = "    if candidate=='causal_decoder_monaural_spectral_v1':"
    code = marker+shell.split(marker, 1)[1].split('\nelse:', 1)[0]
    monkeypatch.chdir(tmp_path)
    Path('evaluate_light.py').write_bytes(b'protected evaluator')
    folder = Path('decoder_source_fp32_reference')
    folder.mkdir()
    old = dict(status='complete', completed_samples=600, expected_samples=600,
        checkpoint_sha256='formal source hash', structure_candidate=MONAURAL_SPECTRAL_CANDIDATE,
        frontend={'sample_rate': 16000}, frontend_protocol={'normalization': 'none'},
        num_mics=8, selected_pairs_sha256='same600', metadata_sha256='same_metadata',
        metric_protocol={'pesq': 'wb', 'target_gain_matching': False},
        mean={'noisy': {'pesq': 1.08}, 'enhanced': {'si_snr_db': 15.9}})
    reference = copy.deepcopy(old)
    reference['mean']['enhanced']['si_snr_db'] += .0002
    if fault == 'pairs': reference['selected_pairs_sha256'] = 'changed'
    if fault == 'partial': reference['completed_samples'] = 599
    path = folder/'summary.json'
    path.write_text(json.dumps(reference))
    execution = dict(status='complete_original_source_fp32_full600', cudnn_allow_tf32=False,
        matmul_allow_tf32=False, source_checkpoint_sha256=old['checkpoint_sha256'],
        summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        evaluator_sha256=hashlib.sha256(Path('evaluate_light.py').read_bytes()).hexdigest())
    if fault == 'tf32': execution['cudnn_allow_tf32'] = True
    if fault == 'source': execution['source_checkpoint_sha256'] = 'other checkpoint'
    if fault == 'evaluator': execution['evaluator_sha256'] = 'changed evaluator'
    if fault == 'summary': path.write_text(json.dumps(old))
    (folder/'execution.json').write_text(json.dumps(execution))
    namespace = dict(Path=Path, json=json, hashlib=hashlib, candidate=DECODER_MONAURAL_CANDIDATE,
        expected_source=MONAURAL_SPECTRAL_CANDIDATE, old=old)
    if fault:
        with pytest.raises(AssertionError):
            exec(compile(textwrap.dedent(code), 'same backend warm reference', 'exec'), namespace)
    else:
        exec(compile(textwrap.dedent(code), 'same backend warm reference', 'exec'), namespace)
        assert namespace['old'] == reference

"""Exercise actual waveform routing, reset, metrics and strict comparison gates."""

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import evaluate_light as evaluator
import evaluate_light_streaming as streaming
from evaluate_light_streaming import WaveformAdapter, require_full_comparison
from light_causal_global_model import CausalGlobalSpatial


def test_streaming_adapter_preserves_waveform_and_resets_each_utterance():
    torch.set_num_threads(2)
    torch.manual_seed(12)
    model = CausalGlobalSpatial(hidden=32, ffn=64, layers=2, context=5, global_hidden=16).eval()
    model.beamformer.weight.data.normal_(std=.01)
    model.layers[-1].output.weight.data.normal_(std=.01)
    adapter = WaveformAdapter(model, evaluator.build_stft_batch, evaluator.reconstruct_waveform).eval()
    signals = [torch.randn(1, 1327, 8) * .03, torch.randn(1, 1600, 8) * .03]
    outputs = []
    with torch.inference_mode():
        for mixture in (*signals, signals[0]):
            actual = adapter(mixture)
            features = evaluator.build_stft_batch(mixture, 512, 160, 320, .5, torch.device('cpu'))
            expected = evaluator.reconstruct_waveform(model(features), mixture.shape[1], 512, 160, 320, .5, torch.device('cpu'))
            torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-5)
            outputs.append(actual)
    torch.testing.assert_close(outputs[0], outputs[2], rtol=0, atol=0)
    evidence = adapter.evidence()
    assert evidence['completed_utterances'] == 3 and evidence['measured_hops'] == 28
    assert evidence['audio_samples'] == 4254 and evidence['stream_reset_per_utterance']
    assert evidence['warmup_state_discarded'] and evidence['future_stft_frames'] == 0
    assert evidence['buffering_delay_ms'] == 20
    assert evidence['full_vs_stream_waveform_max_error'] < 1e-5
    assert np.isfinite(list(evidence['all_hop_ms'].values())).all()


@pytest.mark.parametrize('length', [257, 319, 320, 321, 479, 480, 481])
def test_eof_partial_hop_preserves_exact_received_length(length):
    torch.set_num_threads(2)
    torch.manual_seed(17)
    model = CausalGlobalSpatial(hidden=32, ffn=64, layers=2, context=5, global_hidden=16).eval()
    model.beamformer.weight.data.normal_(std=.01)
    adapter = WaveformAdapter(model, evaluator.build_stft_batch, evaluator.reconstruct_waveform).eval()
    actual = adapter(torch.randn(1, length, 8) * .02)
    assert actual.shape == (1, length)
    assert adapter.waveform_max_error < 1e-5


@pytest.mark.parametrize('fault', [None, 'partial', 'gain', 'alignment', 'noncausal', 'future'])
def test_complete_causal_comparison_gate(fault):
    report = dict(status='complete', expected_samples=600, completed_samples=600,
        structure_candidate='causal_unet_global_v1', candidate_evidence=dict(model_causal=True, future_frames=0),
        metric_protocol=dict(target_gain_matching=False, external_normalization_clipping_or_alignment=False))
    changed = copy.deepcopy(report)
    if fault == 'partial': changed['completed_samples'] = 599
    if fault == 'gain': changed['metric_protocol']['target_gain_matching'] = True
    if fault == 'alignment': changed['metric_protocol']['external_normalization_clipping_or_alignment'] = True
    if fault == 'noncausal': changed['candidate_evidence']['model_causal'] = False
    if fault == 'future': changed['candidate_evidence']['future_frames'] = 1
    if fault is None:
        require_full_comparison(changed)
    else:
        with pytest.raises(ValueError): require_full_comparison(changed)


def test_adapter_failure_is_recorded_and_original_evaluator_is_restored(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'saved.pt'
    checkpoint.write_bytes(b'unit-test checkpoint identity only')
    reference = tmp_path / 'offline.json'
    baseline = dict(status='complete', expected_samples=600, completed_samples=600,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        structure_candidate='causal_unet_global_v1', candidate_evidence=dict(model_causal=True, future_frames=0),
        metric_protocol=dict(target_gain_matching=False, external_normalization_clipping_or_alignment=False),
        selected_pairs_sha256='same', metadata_sha256='same', device='cpu')
    reference.write_text(json.dumps(baseline))
    output = tmp_path / 'streaming.json'
    args = SimpleNamespace(checkpoint=str(checkpoint), max_samples=0, match_estimate_level='no', device='cpu')
    monkeypatch.setattr(evaluator, 'parse_args', lambda unused: args)
    monkeypatch.setattr(evaluator, 'resolve_evaluation_inputs', lambda unused: ([object()] * 600, {'metadata_sha256': 'same'}, set()))
    monkeypatch.setattr(evaluator, 'pairs_sha256', lambda unused: 'same')
    monkeypatch.setattr(evaluator, '_output_plan', lambda *unused: ([], tmp_path, tmp_path / 'metadata.csv', None, output))
    original = (evaluator.load_model, evaluator.build_stft_batch, evaluator.reconstruct_waveform)

    def fail(unused):
        wave = torch.ones(1, 321, 8)
        assert evaluator.build_stft_batch(wave) is wave
        assert evaluator.reconstruct_waveform(wave) is wave
        output.write_text(json.dumps(dict(status='complete')))
        raise ValueError('Simulated inference failure')

    monkeypatch.setattr(evaluator, 'evaluate', fail)
    with pytest.raises(ValueError, match='Simulated inference failure'):
        streaming.run(['--reference-summary', str(reference)])
    assert (evaluator.load_model, evaluator.build_stft_batch, evaluator.reconstruct_waveform) == original
    result = json.loads(output.read_text())
    assert result['status'] == 'failed' and 'Simulated inference failure' in result['error']


def test_cross_device_reference_fails_before_loading_dataset(tmp_path, monkeypatch):
    reference = tmp_path / 'cuda.json'
    reference.write_text(json.dumps(dict(status='complete', expected_samples=600, completed_samples=600,
        device='cuda', structure_candidate='causal_unet_global_v1',
        candidate_evidence=dict(model_causal=True, future_frames=0),
        metric_protocol=dict(target_gain_matching=False, external_normalization_clipping_or_alignment=False))))
    monkeypatch.setattr(evaluator, 'parse_args', lambda unused: SimpleNamespace(
        device='cpu', max_samples=0, match_estimate_level='no'))
    monkeypatch.setattr(evaluator, 'resolve_evaluation_inputs', lambda unused: pytest.fail('Must reject before reading data'))
    with pytest.raises(ValueError, match='same device'):
        streaming.run(['--reference-summary', str(reference)])


def test_failed_quality_gate_preserves_completed_waveform_diagnostics(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'saved.pt'; checkpoint.write_bytes(b'unit test identity')
    reference = tmp_path / 'cpu.json'; output = tmp_path / 'streaming.json'
    baseline = dict(status='complete', expected_samples=600, completed_samples=600, device='cpu',
        checkpoint_sha256=evaluator.sha256_file(checkpoint), structure_candidate='causal_unet_global_v1',
        candidate_evidence=dict(model_causal=True, future_frames=0),
        metric_protocol=dict(target_gain_matching=False, external_normalization_clipping_or_alignment=False),
        selected_pairs_sha256='same', metadata_sha256='same',
        mean={kind:{key:3.0 for key in evaluator.METRICS} for kind in ('enhanced','noisy')})
    for field in streaming.IDENTITY_FIELDS: baseline.setdefault(field, None)
    reference.write_text(json.dumps(baseline))
    args = SimpleNamespace(checkpoint=str(checkpoint), max_samples=0, match_estimate_level='no', device='cpu')
    monkeypatch.setattr(evaluator, 'parse_args', lambda unused: args)
    monkeypatch.setattr(evaluator, 'resolve_evaluation_inputs', lambda unused: ([object()]*600, {'metadata_sha256':'same'}, set()))
    monkeypatch.setattr(evaluator, 'pairs_sha256', lambda unused: 'same')
    monkeypatch.setattr(evaluator, '_output_plan', lambda *unused: ([],tmp_path,tmp_path/'metadata.csv',None,output))
    monkeypatch.setattr(evaluator, 'load_model', lambda *unused: (None,{}))
    monkeypatch.setattr(evaluator, 'resolve_frontend', lambda *unused: dict(sample_rate=16000,n_fft=512,hop_length=160,win_length=320,power=.5,target_ref_mic=0))

    class SimulatedCompletedAdapter:
        utterances = 600
        def __init__(self, *unused): pass
        def eval(self): return self
        def evidence(self): return dict(completed_utterances=600, full_vs_stream_waveform_max_error=1e-6)

    monkeypatch.setattr(streaming, 'WaveformAdapter', SimulatedCompletedAdapter)
    def evaluate(unused):
        evaluator.load_model(checkpoint,torch.device('cpu'))
        report=copy.deepcopy(baseline); report['mean']['enhanced']['pesq']=2.9
        output.write_text(json.dumps(report)); return report
    monkeypatch.setattr(evaluator, 'evaluate', evaluate)
    with pytest.raises(ValueError, match='PESQ differs'):
        streaming.run(['--reference-summary', str(reference)])
    report=json.loads(output.read_text())
    assert report['status']=='failed'
    assert report['waveform_streaming']['completed_utterances']==600
    assert report['waveform_streaming']['full_vs_stream_waveform_max_error']==1e-6

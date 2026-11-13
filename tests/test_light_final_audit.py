"""Reject partial data, relaxed precision, invalid states and false real-time claims."""

import copy

import pytest
import torch

from audit_light_final import validate_final, validate_training
from evaluate_light_streaming import IDENTITY_FIELDS
from light_candidate_runtime import DOUBLE_PHASE_FILTER_CANDIDATE


def reports():
    report = dict(status='complete', completed_samples=600, expected_samples=600,
        structure_candidate='causal_recurrent_tcn_v1',
        metric_protocol=dict(pesq='pesq wideband', gain_rule='none', target_gain_matching=False, external_normalization_clipping_or_alignment=False),
        candidate_evidence=dict(model_causal=True, future_frames=0),
        mean=dict(enhanced=dict(pesq=3.45, estoi=.9), noisy=dict(pesq=1.08, estoi=.4)))
    for name in IDENTITY_FIELDS:
        report.setdefault(name, 'same')
    report['checkpoint_sha256'] = 'actual'
    report['frontend']=dict(sample_rate=16000,n_fft=512,hop_length=160,win_length=320,power=.5,target_ref_mic=0)
    report['num_mics']=8;report['mixture_ref_mic']=0
    report['frontend_protocol']=dict(window='periodic Hann',center=True,pad_mode='reflect',normalized=False,
        onesided=True,waveform_normalization='none',mixture_channels='first 8 channels, original order',
        evaluation_extent='full common-length utterance; no training crop or batch padding',inference_precision='float32')
    reference, streamed = copy.deepcopy(report), copy.deepcopy(report)
    backend=dict(device='cuda:0', matmul_allow_tf32=False, cudnn_allow_tf32=False)
    reference['inference_backend'] = backend
    streamed['waveform_streaming'] = dict(completed_utterances=600, input_hop_samples=160,
        future_stft_frames=0, buffering_delay_samples=320, stream_reset_per_utterance=True,
        warmup_state_discarded=True, inference_backend=copy.deepcopy(backend),
        all_hops_below_10ms=True, all_hop_ms=dict(mean=3.8, p99=4.2, max=4.8), measured_hops=360000)
    latency=dict(status='measured', checkpoint_sha256='actual', future_perturbation_max_error=0.,
        full_vs_stream_waveform_max_error=3e-7, warmup_state_discarded=True,
        bounded_model_cache_bytes=5822208, all_hops_below_10ms=True,
        all_hop_ms=dict(mean=3.8, p99=4.1, max=4.6), measured_hops=600)
    return report, reference, streamed, latency


@pytest.mark.parametrize('fault', [None, 'partial', 'gain', 'backend', 'tf32', 'identity', 'below_target',
    'quality_drift', 'noisy_nan', 'missed_deadline', 'no_hops', 'future', 'causality_nan', 'nb', 'frontend', 'gain_rule'])
def test_final_reports_require_same600_backend_and_real_time(fault):
    training, reference, streamed, latency = reports()
    if fault=='partial':streamed['completed_samples']=599
    if fault=='gain':reference['metric_protocol']['target_gain_matching']=True
    if fault=='backend':streamed['waveform_streaming']['inference_backend']['device']='cpu'
    if fault=='tf32':
        reference['inference_backend']['cudnn_allow_tf32']=True
        streamed['waveform_streaming']['inference_backend']['cudnn_allow_tf32']=True
    if fault=='identity':reference['metadata_sha256']='different'
    if fault=='below_target':streamed['mean']['enhanced']['pesq']=3.39999
    if fault=='quality_drift':streamed['mean']['enhanced']['pesq']+=.001
    if fault=='noisy_nan':reference['mean']['noisy']['pesq']=float('nan')
    if fault=='missed_deadline':streamed['waveform_streaming']['all_hop_ms']['max']=10.01
    if fault=='no_hops':latency['measured_hops']=0
    if fault=='future':latency['future_perturbation_max_error']=.01
    if fault=='causality_nan':latency['future_perturbation_max_error']=float('nan')
    if fault=='nb':reference['metric_protocol']['pesq']='pesq narrowband'
    if fault=='frontend':reference['frontend']['win_length']=512
    if fault=='gain_rule':reference['metric_protocol']['gain_rule']='target RMS'
    if fault:
        with pytest.raises(ValueError):validate_final(training, reference, streamed, latency)
    else:
        validate_final(training, reference, streamed, latency)


@pytest.mark.parametrize('fault', [None, 'subset', 'initial_model', 'updates', 'hour', 'weights', 'adam'])
def test_training_proof_requires_complete_data_finite_state_and_full_budget(fault):
    training, _, _, _ = reports()
    state=dict(epoch=1, model_state_dict={'weight':torch.ones(1)},
        optimizer_state_dict={'state':{0:dict(step=torch.tensor(6700), exp_avg=torch.zeros(1))}})
    timing=dict(status='complete', epoch=1, training_samples=53600, validation_samples=600,
        full_metric_samples=600, optimizer_updates_this_epoch=6700, global_batch_size=8, total_seconds=3250.)
    if fault=='subset':timing['training_samples']=8
    if fault=='initial_model':state['epoch']=0
    if fault=='updates':state['optimizer_state_dict']['state'][0]['step']=torch.tensor(200)
    if fault=='hour':timing['total_seconds']=3600.01
    if fault=='weights':state['model_state_dict']['weight'][0]=float('inf')
    if fault=='adam':state['optimizer_state_dict']['state'][0]['exp_avg'][0]=float('nan')
    if fault:
        with pytest.raises(ValueError):validate_training(state, training, timing, 'actual')
    else:
        validate_training(state, training, timing, 'actual')


def test_double_phase_filter_audits_both_optimizer_histories():
    training, _, _, _ = reports()
    training['structure_candidate'] = DOUBLE_PHASE_FILTER_CANDIDATE
    base = list(range(235))
    added = list(range(235, 241))
    optimizer = {i: dict(step=torch.tensor(55 * 6700)) for i in base}
    optimizer.update({i: dict(step=torch.tensor(5 * 6700)) for i in added})
    state = dict(epoch=55, args=dict(structure_candidate=DOUBLE_PHASE_FILTER_CANDIDATE,
                                    phase_filter_source_epoch=50),
                 model_state_dict={'weight': torch.ones(1)},
                 optimizer_state_dict=dict(state=optimizer, param_groups=[dict(params=base), dict(params=added)]))
    timing = dict(status='complete', epoch=55, training_samples=53600, validation_samples=600,
                  full_metric_samples=600, optimizer_updates_this_epoch=6700,
                  global_batch_size=8, total_seconds=3340.)
    validate_training(state, training, timing, 'actual')
    optimizer[added[0]]['step'] = torch.tensor(6700)
    with pytest.raises(ValueError, match='Adam updates'):
        validate_training(state, training, timing, 'actual')

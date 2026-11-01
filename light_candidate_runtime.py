"""Process-local model selection; the original training/data code is reused."""

from contextlib import contextmanager
from functools import partial
from pathlib import Path

import torch

from analyze_light_structure import source_hashes
from light_reference_variants import (all_reference_candidates, build_reference_candidate,
                                      reference_candidate_metadata)


IMPLEMENTATION_FILES = ("EaBNet_light.py", "light_structure_variants.py", "light_reference_variants.py")
QUALITY_CANDIDATE = "cbam_refiner_v1"
QUALITY_IMPLEMENTATION_FILES = (*IMPLEMENTATION_FILES, "light_quality_model.py")
SPATIAL_CANDIDATE = "cbam_spatial_refiner_v1"
SPATIAL_IMPLEMENTATION_FILES = (*QUALITY_IMPLEMENTATION_FILES, "light_spatial_model.py",
    "third_party/spatialnet/SpatialNet.py", "third_party/spatialnet/norm.py",
    "third_party/spatialnet/non_linear.py", "third_party/spatialnet/linear_group.py")
CAUSAL_CANDIDATE = "causal_spatial_v1"
CAUSAL_IMPLEMENTATION_FILES = ("light_causal_spatial_model.py", "light_causal_stream.py",
    "third_party/spatialnet/SpatialNet.py", "third_party/spatialnet/norm.py",
    "third_party/spatialnet/non_linear.py", "third_party/spatialnet/linear_group.py")
FAST_CANDIDATE = "causal_unet_spatial_v1"
GLOBAL_CANDIDATE = "causal_unet_global_v1"
RECURRENT_CANDIDATE = "causal_recurrent_spatial_v1"
RECURRENT65_CANDIDATE = "causal_recurrent_unet65_v1"
TCN_CANDIDATE = "causal_recurrent_tcn_v1"
FOURIER_CANDIDATE = "causal_frequency_fourier_v1"
FREQUENCY_ATTENTION_CANDIDATE = "causal_frequency_attention_v1"
POOLED_FREQUENCY_ATTENTION_CANDIDATE = "causal_frequency_attention_kv129_v1"
RETENTION_CANDIDATE = "causal_frequency_attention_retention_v1"
FULLBAND_TCN_CANDIDATE = "causal_fullband_tcn_v1"
FULLBAND_SPATIAL_CANDIDATE = "causal_fullband_spatial_tcn_v1"
SUBBAND_GRU_CANDIDATE = "causal_retention_subband_gru_v1"
CONDITIONED_SUBBAND_CANDIDATE = "causal_fullband_conditioned_subband_v1"
FROZEN_CONDITIONED_CANDIDATE = "causal_frozen_conditioned_subband_v1"
SPATIAL_CONDITIONED_CANDIDATE = "causal_spatial_conditioned_subband_v1"
IDENTITY_FIELDS = ("sample_rate", "n_fft", "hop_length", "win_length", "power", "target_ref_mic",
                   "channels", "num_mics", "embed_dim", "cd1", "dfsmn_layers", "dfsmn_memory_size",
                   "norm_type", "bf_type", "topo_type", "is_causal")


def all_model_candidates():
    return (*all_reference_candidates(), QUALITY_CANDIDATE, SPATIAL_CANDIDATE, CAUSAL_CANDIDATE, FAST_CANDIDATE, GLOBAL_CANDIDATE, RECURRENT_CANDIDATE, RECURRENT65_CANDIDATE, TCN_CANDIDATE, FOURIER_CANDIDATE, FREQUENCY_ATTENTION_CANDIDATE, POOLED_FREQUENCY_ATTENTION_CANDIDATE, RETENTION_CANDIDATE, FULLBAND_TCN_CANDIDATE, FULLBAND_SPATIAL_CANDIDATE, SUBBAND_GRU_CANDIDATE, CONDITIONED_SUBBAND_CANDIDATE, FROZEN_CONDITIONED_CANDIDATE, SPATIAL_CONDITIONED_CANDIDATE)


def build_model_candidate(candidate_id, **model_kwargs):
    if candidate_id == SPATIAL_CONDITIONED_CANDIDATE:
        from experiments.causal_spatial_conditioned_subband import CausalSpatialConditionedSubband
        return CausalSpatialConditionedSubband(build_model_candidate(RETENTION_CANDIDATE, **model_kwargs))
    if candidate_id == FROZEN_CONDITIONED_CANDIDATE:
        from experiments.causal_frozen_conditioned_subband import CausalFrozenBaseConditionedSubband
        return CausalFrozenBaseConditionedSubband(build_model_candidate(RETENTION_CANDIDATE, **model_kwargs))
    if candidate_id == CONDITIONED_SUBBAND_CANDIDATE:
        from experiments.causal_fullband_conditioned_subband import CausalFullbandConditionedSubband
        return CausalFullbandConditionedSubband(build_model_candidate(RETENTION_CANDIDATE, **model_kwargs))
    if candidate_id == SUBBAND_GRU_CANDIDATE:
        from experiments.causal_subband_gru import CausalSubbandGRU
        return CausalSubbandGRU(build_model_candidate(RETENTION_CANDIDATE, **model_kwargs))
    if candidate_id == FULLBAND_SPATIAL_CANDIDATE:
        from experiments.causal_fullband_spatial_tcn import CausalFullbandSpatialTCN
        return CausalFullbandSpatialTCN(**model_kwargs)
    if candidate_id == FULLBAND_TCN_CANDIDATE:
        from experiments.causal_fullband_tcn import CausalFullbandTCN
        return CausalFullbandTCN(**model_kwargs)
    if candidate_id == RETENTION_CANDIDATE:
        from experiments.causal_retention_mapping import CausalAttentionRetentionTCN
        return CausalAttentionRetentionTCN(**model_kwargs)
    if candidate_id == POOLED_FREQUENCY_ATTENTION_CANDIDATE:
        from experiments.causal_pooled_frequency_attention import CausalPooledFrequencyAttentionTCN
        return CausalPooledFrequencyAttentionTCN(**model_kwargs)
    if candidate_id == FREQUENCY_ATTENTION_CANDIDATE:
        from experiments.causal_frequency_attention import CausalFrequencyAttentionTCN
        return CausalFrequencyAttentionTCN(**model_kwargs)
    if candidate_id == FOURIER_CANDIDATE:
        from experiments.causal_frequency_fourier import CausalFrequencyFourierTCN
        return CausalFrequencyFourierTCN(**model_kwargs)
    if candidate_id == TCN_CANDIDATE:
        from experiments.causal_tcn_mapping import CausalTCNMapping
        return CausalTCNMapping(**model_kwargs)
    if candidate_id == RECURRENT65_CANDIDATE:
        from light_causal_recurrent65_model import CausalRecurrentUNet65
        return CausalRecurrentUNet65(**model_kwargs)
    if candidate_id == RECURRENT_CANDIDATE:
        from light_causal_recurrent_model import CausalRecurrentSpatial
        return CausalRecurrentSpatial(**model_kwargs)
    if candidate_id == GLOBAL_CANDIDATE:
        from light_causal_global_model import CausalGlobalSpatial
        return CausalGlobalSpatial(**model_kwargs)
    if candidate_id == FAST_CANDIDATE:
        from light_causal_fast_model import CausalUNetSpatial
        return CausalUNetSpatial(**model_kwargs)
    if candidate_id == CAUSAL_CANDIDATE:
        from light_causal_spatial_model import CausalSpatialEnhancer
        return CausalSpatialEnhancer(**model_kwargs)
    if candidate_id == SPATIAL_CANDIDATE:
        from light_spatial_model import EaBNetSpatialQuality
        return EaBNetSpatialQuality(**model_kwargs)
    if candidate_id == QUALITY_CANDIDATE:
        from light_quality_model import EaBNetQuality
        return EaBNetQuality(**model_kwargs)
    return build_reference_candidate(candidate_id, **model_kwargs)


def candidate_metadata(candidate_id):
    if candidate_id == SPATIAL_CONDITIONED_CANDIDATE:
        metadata = candidate_metadata(FROZEN_CONDITIONED_CANDIDATE)
        files = (*metadata['candidate_implementation_sha256'], 'experiments/causal_spatial_conditioned_subband.py')
        metadata.update(structure_candidate=candidate_id, candidate_schema=19,
                        candidate_implementation_sha256=source_hashes(files))
        evidence = metadata['candidate_evidence']
        evidence.pop('warm_start_candidates')
        evidence.update(candidate_id=candidate_id, warm_start_candidate=FROZEN_CONDITIONED_CANDIDATE,
            purpose='Global causal temporal conditioning from microphone phase, level and speech cues across frequency',
            initialization='Complete trained fixed-base conditioned checkpoint; zero additional spatial condition; fixed causal backbone; fresh head Adam')
        evidence['model_config'].update(parameters=5058680,trainable_parameters=2478982,
            global_spatial_condition=dict(input_features=40,frequency_channels=32,
                frequency_kernels=[5,3,3,3],frequency_stride=2,frequency_bins_after_encoder=17,
                temporal_kernels=1,recurrent_hidden=256,bidirectional=False,condition_channels=16,
                extra_parameters=1688720,extra_cache_bytes_per_stream_fp32=1024,
                normalization='Current-frame channel LayerNorm only'))
        return metadata
    if candidate_id == FROZEN_CONDITIONED_CANDIDATE:
        metadata = candidate_metadata(CONDITIONED_SUBBAND_CANDIDATE)
        files = (*metadata['candidate_implementation_sha256'], 'experiments/causal_frozen_conditioned_subband.py')
        metadata.update(structure_candidate=candidate_id, candidate_schema=18,
                        candidate_implementation_sha256=source_hashes(files))
        evidence = metadata['candidate_evidence']
        evidence.pop('warm_start_candidate')
        evidence.update(candidate_id=candidate_id, frozen_backbone=True, frozen_backbone_is_causal=True,
            warm_start_candidates=[SUBBAND_GRU_CANDIDATE, CONDITIONED_SUBBAND_CANDIDATE],
            purpose='Train the conditioned speech head with stable FP32 causal backbone predictions',
            initialization='Complete causal subband or conditioned checkpoint; preserve existing head; zero absent full-band condition; fixed causal backbone; fresh head Adam')
        evidence['model_config'].update(trainable_parameters=790262, frozen_causal_parameters=2579698,
            frozen_base_mode='FP32 eval during head training and inference')
        return metadata
    if candidate_id == CONDITIONED_SUBBAND_CANDIDATE:
        metadata = candidate_metadata(SUBBAND_GRU_CANDIDATE)
        files = (*metadata['candidate_implementation_sha256'], 'experiments/causal_fullband_conditioned_subband.py')
        metadata.update(structure_candidate=candidate_id, candidate_schema=17,
                        candidate_implementation_sha256=source_hashes(files))
        evidence = metadata['candidate_evidence']
        evidence.update(candidate_id=candidate_id, warm_start_candidate=SUBBAND_GRU_CANDIDATE,
            purpose='Full-band temporal speech context conditions independent frequency-wise spatial GRUs',
            initialization='Complete subband checkpoint; zero full-band condition; all parameters trainable; fresh Adam')
        evidence['model_config']['parameters'] = 3369960
        evidence['model_config']['fullband_condition'] = dict(input_features=514,
            input='Current reference-mixture and causal base-estimate compressed magnitudes',
            recurrent_hidden=128, layers=1, bidirectional=False, condition_channels_per_frequency=16,
            output='32-channel current-frequency condition before the local temporal GRU',
            normalization='Current-frame reference scale and LayerNorm only',
            extra_parameters=779316, extra_cache_bytes_per_stream_fp32=512)
        return metadata
    if candidate_id == SUBBAND_GRU_CANDIDATE:
        metadata = candidate_metadata(RETENTION_CANDIDATE)
        files = (*metadata['candidate_implementation_sha256'], 'experiments/causal_fullband_tcn.py',
                 'experiments/causal_subband_gru.py')
        metadata.update(structure_candidate=candidate_id, candidate_schema=16,
                        candidate_implementation_sha256=source_hashes(files))
        evidence = metadata['candidate_evidence']
        evidence.update(candidate_id=candidate_id, base_candidate=RETENTION_CANDIDATE,
            purpose='Full-resolution frequency-wise gated temporal states with explicit microphone spatial cues',
            initialization='Complete retention checkpoint; zero additional output; all parameters trainable; fresh Adam',
            frozen_backbone=False, reference_audio_used_at_inference=False)
        evidence['model_config']['parameters'] = 2590644
        evidence['model_config']['subband_gru'] = dict(frequencies=257, features=40,
            hidden=32, layers=1, bidirectional=False, parameters_shared_across_frequencies=True,
            independent_state_per_frequency=True, frequency_kernel=3, extra_parameters=10946,
            extra_cache_bytes_per_stream_fp32=32896, output='Complex8-microphone residual plus compressed RI residual',
            normalization='Current-frame channel LayerNorm and current reference-mixture frame scale only')
        return metadata
    if candidate_id == FULLBAND_SPATIAL_CANDIDATE:
        metadata = candidate_metadata(FULLBAND_TCN_CANDIDATE)
        files = (*metadata['candidate_implementation_sha256'],'experiments/causal_fullband_spatial_tcn.py')
        metadata.update(structure_candidate=candidate_id,candidate_schema=15,
            candidate_implementation_sha256=source_hashes(files))
        evidence = metadata['candidate_evidence']
        evidence.update(candidate_id=candidate_id,
            purpose='Joint global full-band context and direct per-frequency causal spatial encoding',
            initialization='Complete full-band checkpoint; zero extra spatial output; fresh Adam')
        evidence['model_config'].update(parameters=45501488,cache_bytes_per_stream_fp32=9227520,
            direct_spatial_path=dict(features_per_frequency=38,channels=64,hidden=96,
                dilations=[1,2,4,8],kernel=[3,3],past_frames=30,
                global_condition='Current global representation conditions output after local temporal layers'))
        return metadata
    if candidate_id == FULLBAND_TCN_CANDIDATE:
        return dict(structure_candidate=candidate_id, candidate_schema=14,
            candidate_implementation_sha256=source_hashes((*CAUSAL_IMPLEMENTATION_FILES,
                'light_causal_fast_model.py', 'experiments/causal_fullband_tcn.py')),
            candidate_evidence=dict(candidate_id=candidate_id, author_confirmed=False,
                exact_reproduction_verified=False, model_causal=True, future_frames=0,
                purpose='Dense full-band causal TCN with explicit current-frame microphone phase and level cues',
                model_config=dict(input_features=9766, frequencies=257, microphones=8, channels=512,
                    temporal_hidden=1024, repeats=3, dilations=[1,2,4,8,16,32,64,128],
                    layers=24, past_frames=1530, parameters=45352542,
                    cache_bytes_per_stream_fp32=6266880),
                initialization='From scratch; zero beamforming residual and mapping; exact reference0 mixture',
                normalization='Current-frame LayerNorm; input scale uses current reference-mixture frame only',
                output='Complex8-microphone residual beamforming and compressed RI mapping',
                frozen_backbone=False, reference_audio_used_at_inference=False,
                waveform_streaming='512/160/320,16kHz,power0.5;320-sample buffering plus compute/I-O'))
    if candidate_id == RETENTION_CANDIDATE:
        metadata = candidate_metadata(FREQUENCY_ATTENTION_CANDIDATE)
        files = (*metadata['candidate_implementation_sha256'], 'experiments/causal_retention_mapping.py')
        metadata.update(structure_candidate=candidate_id, candidate_schema=13,
                        candidate_implementation_sha256=source_hashes(files))
        evidence = metadata['candidate_evidence']
        evidence.update(candidate_id=candidate_id,
            purpose='Full257-frequency causal mapping with current-frame frequency attention and per-frequency matrix memory',
            initialization='Complete full-data frequency-attention checkpoint; zero new retention output; fresh Adam')
        evidence['model_config']['mapping']['retention'] = dict(channels=32, heads=4, head_dim=8,
            chunk_size=64, decay_powers=[4,5,9,10], state='Canonical decayed FP32 key-value sums and counts',
            normalization='Current-frame channel LayerNorm; per-head RMS of causal retained output',
            extra_parameters=5344, extra_cache_bytes_per_stream=263184,
            upstream_source_incorporated=False,
            design_evidence='Own canonical recurrence; no author equivalence or guaranteed WB-PESQ claim')
        return metadata
    if candidate_id == POOLED_FREQUENCY_ATTENTION_CANDIDATE:
        metadata = candidate_metadata(FREQUENCY_ATTENTION_CANDIDATE)
        files = (*metadata['candidate_implementation_sha256'], 'experiments/causal_pooled_frequency_attention.py')
        metadata.update(structure_candidate=candidate_id, candidate_schema=12,
                        candidate_implementation_sha256=source_hashes(files))
        evidence = metadata['candidate_evidence']
        evidence.update(candidate_id=candidate_id,
            purpose='Causal full-frequency TCN with257-query attention and129 current-frame pooled keys/values')
        evidence['model_config']['mapping']['frequency_attention'].update(
            query_frequencies=257, key_value_frequencies=129,
            pooling='Mean of adjacent frequency pairs; Nyquist singleton; no temporal pooling')
        return metadata
    if candidate_id == FREQUENCY_ATTENTION_CANDIDATE:
        metadata = candidate_metadata(TCN_CANDIDATE)
        files = (*metadata['candidate_implementation_sha256'], 'experiments/causal_frequency_attention.py')
        metadata.update(structure_candidate=candidate_id, candidate_schema=11,
                        candidate_implementation_sha256=source_hashes(files))
        evidence = metadata['candidate_evidence']
        evidence.update(candidate_id=candidate_id,
            purpose='Causal TCN with position-aware global frequency attention inside each current frame',
            initialization='Complete causal TCN checkpoint; zero new frequency-attention output; fresh Adam')
        evidence['model_config']['mapping']['frequency_attention'] = dict(
            sequence_axis='257 current-frame frequencies; time is an independent batch axis',
            channels=32, projection=64, heads=4, head_dim=16, dropout=0.,
            learned_frequency_positions=True, extra_parameters=16704, extra_temporal_cache_bytes=0)
        return metadata
    if candidate_id == FOURIER_CANDIDATE:
        metadata = candidate_metadata(TCN_CANDIDATE)
        files = (*metadata['candidate_implementation_sha256'], 'experiments/causal_frequency_fourier.py')
        metadata.update(structure_candidate=candidate_id, candidate_schema=10,
                        candidate_implementation_sha256=source_hashes(files))
        evidence = metadata['candidate_evidence']
        evidence.update(candidate_id=candidate_id,
            purpose='Joint causal TCN with global frequency Fourier mixing within each current frame',
            initialization='Validated complete causal TCN checkpoint; zero new Fourier output; fresh Adam')
        evidence['model_config']['mapping']['frequency_fourier'] = dict(
            transform_axis='frequency only; no time-axis transform', fft_length=512,
            channels=32, nonlinear='64-channel pointwise layers, channel norm and SiLU',
            extra_parameters=8448, extra_temporal_cache_bytes=0)
        return metadata
    if candidate_id == TCN_CANDIDATE:
        metadata = candidate_metadata(RECURRENT65_CANDIDATE)
        files = (*metadata['candidate_implementation_sha256'], 'experiments/causal_tcn_mapping.py')
        metadata.update(structure_candidate=candidate_id, candidate_schema=9,
                        candidate_implementation_sha256=source_hashes(files))
        evidence = metadata['candidate_evidence']
        evidence.update(candidate_id=candidate_id,
            purpose='Jointly trained65-group spatial U-Net and full257-frequency causal TCN spectral mapping',
            initialization='Validated complete recurrent65 checkpoint; zero new mapping output; fresh Adam',
            output='Complex8-microphone residual beamforming plus current-mixture-scaled compressed RI mapping',
            normalization='Current-frame channel LayerNorm; mapping scale from current reference-mixture frame only')
        evidence['model_config'].update(mapping=dict(input=18, channels=32, output=2,
            dilations=[1,2,4,8,16,32], kernel=[3,3], past_frames=126,
            extra_cache_bytes_per_stream=4144896, frozen_parameters=0))
        return metadata
    if candidate_id == RECURRENT65_CANDIDATE:
        metadata=candidate_metadata(FAST_CANDIDATE)
        metadata.update(structure_candidate=candidate_id,candidate_schema=8,
            candidate_implementation_sha256=source_hashes((*CAUSAL_IMPLEMENTATION_FILES,
                'light_causal_fast_model.py','light_causal_global_model.py','light_causal_recurrent_model.py',
                'light_causal_recurrent65_model.py')))
        evidence=metadata['candidate_evidence']
        evidence.update(candidate_id=candidate_id,
            purpose='65-frequency-group recurrent spatial modeling with full-resolution skips and global temporal recurrence',
            model_config=dict(widths=[32,48,96],ffn=192,spatial_layers=4,bottleneck_frequencies=65,
                global_temporal=dict(input=6240,hidden=128,layers=1)),
            attention='Replaced by four unidirectional per-frequency GRU96 blocks plus one global GRU128, FP32',
            initialization='Zero complex beamforming residual; exact current reference mixture',
            recurrent_state='Fixed per-frequency and global hidden states plus left convolution histories')
        return metadata
    if candidate_id == RECURRENT_CANDIDATE:
        metadata=candidate_metadata(CAUSAL_CANDIDATE)
        metadata.update(structure_candidate=candidate_id,candidate_schema=7,
            candidate_implementation_sha256=source_hashes((*CAUSAL_IMPLEMENTATION_FILES,
                'light_causal_fast_model.py','light_causal_recurrent_model.py')))
        evidence=metadata['candidate_evidence']
        evidence.update(candidate_id=candidate_id,
            purpose='Full257-frequency causal spatial encoding with recurrent temporal modeling; all parameters trainable',
            model_config=dict(hidden=96,ffn=192,layers=4,frequencies=257,temporal='One unidirectional GRU96 per block, FP32'),
            attention='Replaced by per-frequency recurrent states; no attention matrix or future frames',
            normalization='Current-frame channels only; reference-mixture scale from current-frame frequencies',
            initialization='Zero physical complex residual; exact current reference mixture',
            recurrent_state='Four fixed GRU96 hidden states per frequency, plus bounded convolution caches')
        return metadata
    if candidate_id == GLOBAL_CANDIDATE:
        metadata=candidate_metadata(FAST_CANDIDATE)
        metadata.update(structure_candidate=candidate_id,candidate_schema=6,
            candidate_implementation_sha256=source_hashes((*CAUSAL_IMPLEMENTATION_FILES,
                'light_causal_fast_model.py','light_causal_global_model.py')))
        evidence=metadata['candidate_evidence']
        evidence.update(candidate_id=candidate_id,
            purpose='Causal spatial encoding plus global-frequency temporal modeling; all parameters trainable',
            initialization='Validated causal fast checkpoint, zero global residual, fresh Adam')
        evidence['model_config'].update(global_temporal=dict(input=2176,hidden=128,layers=1,
            recurrent='Unidirectional GRU, FP32',normalization='Current-frame LayerNorm over17x128 features',
            streaming_state='One128-dimensional hidden vector per utterance'))
        return metadata
    if candidate_id == FAST_CANDIDATE:
        return {
            "structure_candidate": candidate_id, "candidate_schema": 5,
            "candidate_implementation_sha256": source_hashes((*CAUSAL_IMPLEMENTATION_FILES, "light_causal_fast_model.py")),
            "candidate_evidence": {
                "candidate_id": candidate_id, "author_confirmed": False,
                "exact_reproduction_verified": False,
                "purpose": "Multi-resolution causal spatial beamformer; end-to-end, no noncausal source",
                "model_causal": True, "future_frames": 0, "frozen_backbone": False,
                "model_config": {"widths": [32, 48, 64, 96, 128], "ffn": 256,
                    "spatial_layers": 8, "heads": 4, "context": 128, "bottleneck_frequencies": 17},
                "temporal_convolution": "Left padding and fixed input caches; frequency-only transposed convolutions",
                "attention": "Current plus at most127 history frames at17 frequency groups",
                "normalization": "Current-frame channels only; FP32 LayerNorm reductions with activation dtype preserved",
                "output": "Unconstrained complex8-microphone residual beamforming weights; sum(weight*mixture), no conjugation",
                "initialization": "Zero residual weights; current reference mixture",
                "reference_audio_used_at_inference": False,
                "waveform_streaming": "512/160/320,16kHz,power0.5;320-sample buffering plus compute/I-O; discarded warmup",
                "source_revision": "Frequency modules adapted from Audio-WestlakeU/NBSS@cc42fc8ad2e6642c09b8f4169a85b4766dc22b7e",
            },
        }
    if candidate_id == CAUSAL_CANDIDATE:
        return {
            "structure_candidate": candidate_id, "candidate_schema": 4,
            "candidate_implementation_sha256": source_hashes(CAUSAL_IMPLEMENTATION_FILES),
            "candidate_evidence": {
                "candidate_id": candidate_id, "author_confirmed": False,
                "exact_reproduction_verified": False,
                "purpose": "End-to-end causal spatial adaptation; no noncausal source model.",
                "model_causal": True, "future_frames": 0, "frozen_backbone": False,
                "model_config": {"hidden": 96, "ffn": 192, "layers": 8, "heads": 4, "context": 128},
                "temporal_convolution": "Left padding and fixed input caches",
                "attention": "Current and at most 127 past frames; bounded per-layer K/V cache",
                "normalization": "LayerNorm over channels; input scale from current-frame reference-mixture frequencies only",
                "initialization": "Zero complex residual on current reference mixture",
                "reference_audio_used_at_inference": False,
                "waveform_streaming": "512/160/320, 16kHz, power0.5; 320-sample buffering budget plus compute/I-O",
                "source_revision": "Audio-WestlakeU/NBSS@cc42fc8ad2e6642c09b8f4169a85b4766dc22b7e",
            },
        }
    if candidate_id == SPATIAL_CANDIDATE:
        return {
            "structure_candidate": candidate_id,
            "candidate_schema": 3,
            "candidate_implementation_sha256": source_hashes(SPATIAL_IMPLEMENTATION_FILES),
            "candidate_evidence": {
                "candidate_id": candidate_id, "author_confirmed": False,
                "exact_reproduction_verified": False,
                "purpose": "SpatialNet residual adaptation; not a paper reproduction.",
                "base_candidate": QUALITY_CANDIDATE, "base_frozen": True,
                "base_inference_precision": "float32", "refiner_causal": False,
                "refiner": "Eight author SpatialNet-small blocks, C96/FFN192/heads4/squeeze8/F257",
                "source_revision": "Audio-WestlakeU/NBSS@cc42fc8ad2e6642c09b8f4169a85b4766dc22b7e",
                "features": "Raw RI for first 8 microphones plus frozen estimate RI",
                "internal_normalization": "Reference-mixture mean magnitude over valid frames per frequency, canceled at output",
                "external_waveform_normalization": "none",
                "initialization": "Zero residual decoder; exactly the frozen quality model",
                "reference_audio_used_at_inference": False,
            },
        }
    if candidate_id == QUALITY_CANDIDATE:
        return {
            "structure_candidate": candidate_id,
            "candidate_schema": 2,
            "candidate_implementation_sha256": source_hashes(QUALITY_IMPLEMENTATION_FILES),
            "candidate_evidence": {
                "candidate_id": candidate_id,
                "author_confirmed": False,
                "exact_reproduction_verified": False,
                "purpose": "Quality-oriented extension; not a reproduction of the paper architecture.",
                "base_candidate": "cbam_flat_projection64",
                "base_frozen": True,
                "base_inference_precision": "float32",
                "refiner": "Frequency U-Net with full-band bidirectional GRU and complex residual output",
                "refiner_causal": False,
                "initialization": "Zero residual output; the initial model equals the frozen baseline",
                "reference_audio_used_at_inference": False,
            },
        }
    if candidate_id not in all_reference_candidates():
        raise ValueError(f"Unknown structure candidate: {candidate_id}")
    return {"structure_candidate": candidate_id, "candidate_schema": 1,
            "candidate_implementation_sha256": source_hashes(IMPLEMENTATION_FILES),
            "candidate_evidence": reference_candidate_metadata(candidate_id)}


def read_candidate_checkpoint(path, expected_candidate=None, expected_args=None):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Candidate checkpoint must contain model_state_dict and candidate metadata")
    saved = checkpoint.get("args", {})
    if not isinstance(saved, dict):
        raise ValueError("Candidate checkpoint args must be a dictionary")
    missing = [name for name in IDENTITY_FIELDS if name not in saved]
    if missing:
        raise ValueError(f"Candidate checkpoint lacks required configuration fields: {', '.join(missing)}")
    candidate_id = saved.get("structure_candidate")
    if candidate_id not in all_model_candidates():
        raise ValueError("Checkpoint lacks a recognized structure_candidate; do not guess its architecture")
    if expected_candidate is not None and candidate_id != expected_candidate:
        raise ValueError(f"Candidate mismatch: checkpoint={candidate_id}, requested={expected_candidate}")
    for name, value in candidate_metadata(candidate_id).items():
        if saved.get(name) != value:
            raise ValueError(f"Candidate checkpoint {name} differs from the current implementation")
    if expected_args is not None:
        for name in IDENTITY_FIELDS:
            if saved.get(name) != getattr(expected_args, name):
                raise ValueError(f"Resume {name} differs from checkpoint")
    return checkpoint, candidate_id


@contextmanager
def select_model_factory(module, candidate_id):
    """Substitute only the constructor in this process, restoring it on exit."""
    original = module.EaBNet
    module.EaBNet = partial(build_model_candidate, candidate_id)
    try:
        yield
    finally:
        module.EaBNet = original

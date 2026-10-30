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
IDENTITY_FIELDS = ("sample_rate", "n_fft", "hop_length", "win_length", "power", "target_ref_mic",
                   "channels", "num_mics", "embed_dim", "cd1", "dfsmn_layers", "dfsmn_memory_size",
                   "norm_type", "bf_type", "topo_type", "is_causal")


def all_model_candidates():
    return (*all_reference_candidates(), QUALITY_CANDIDATE, SPATIAL_CANDIDATE, CAUSAL_CANDIDATE, FAST_CANDIDATE, GLOBAL_CANDIDATE)


def build_model_candidate(candidate_id, **model_kwargs):
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

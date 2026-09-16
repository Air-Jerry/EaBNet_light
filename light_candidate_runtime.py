"""Process-local model selection; the original training/data code is reused."""

from contextlib import contextmanager
from functools import partial
from pathlib import Path

import torch

from analyze_light_structure import source_hashes
from light_reference_variants import (all_reference_candidates, build_reference_candidate,
                                      reference_candidate_metadata)


IMPLEMENTATION_FILES = ("EaBNet_light.py", "light_structure_variants.py", "light_reference_variants.py")
IDENTITY_FIELDS = ("sample_rate", "n_fft", "hop_length", "win_length", "power", "target_ref_mic",
                   "channels", "num_mics", "embed_dim", "cd1", "dfsmn_layers", "dfsmn_memory_size",
                   "norm_type", "bf_type", "topo_type", "is_causal")


def candidate_metadata(candidate_id):
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
    if candidate_id not in all_reference_candidates():
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
    module.EaBNet = partial(build_reference_candidate, candidate_id)
    try:
        yield
    finally:
        module.EaBNet = original

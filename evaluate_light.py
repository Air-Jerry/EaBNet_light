"""Evaluate saved EaBNet models with the exact training frontend.

Supports metadata.csv or explicit audio pairs, candidate identity checks,
PESQ/ESTOI/SDR and noisy baselines, optional WAV export and bounded gain matching.
Training configuration is read from checkpoint args; conflicting overrides fail.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import warnings
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import soundfile as sf
import torch
from mir_eval.separation import bss_eval_sources
from pesq import pesq
from pystoi import stoi
from tqdm import tqdm

from EaBNet_light import EaBNet
from light_candidate_runtime import read_candidate_checkpoint
from light_reference_variants import all_reference_candidates, build_reference_candidate
from light_eval_inputs import (SampleRecord, load_records, resolve_record_path,
                               pairs_sha256, resolve_evaluation_inputs)

EPS = 1e-8
_WINDOW_CACHE: Dict[Tuple[int, str, int], torch.Tensor] = {}
FRONTEND_FIELDS = ("sample_rate", "n_fft", "hop_length", "win_length", "power", "target_ref_mic")
PAPER_FRONTEND = {"sample_rate": 16000, "n_fft": 512, "hop_length": 160, "win_length": 320}
METRICS = ("pesq", "stoi", "estoi", "si_snr_db", "sdr_db")
SILENCE_MEAN_SQUARE = 1e-12
SI_SNR_EPSILON = 1e-8


@dataclass
class ModelConfig:
    channels: int = 64
    num_mics: int = 8
    embed_dim: int = 64
    kd1: int = 5
    cd1: int = 64
    d_feat: int = 256
    p: int = 6
    q: int = 3
    bf_type: str = "lstm"
    topo_type: str = "mimo"
    intra_connect: str = "cat"
    norm_type: str = "BN"
    dfsmn_layers: int = 3
    dfsmn_memory_size: int = 20
    is_causal: str = "yes"
    is_u2: str = "yes"



def get_window(win_length: int, device: torch.device) -> torch.Tensor:
    cache_key = (win_length, device.type, device.index or 0)
    window = _WINDOW_CACHE.get(cache_key)
    if window is None:
        window = torch.hann_window(win_length, device=device)
        _WINDOW_CACHE[cache_key] = window
    return window



def build_stft_batch(
    mixture: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    power: float,
    device: torch.device,
) -> torch.Tensor:
    batch_size, _, num_mics = mixture.shape
    window = get_window(win_length, device)

    mixture = mixture.to(device, non_blocking=True)
    mixture_flat = mixture.transpose(1, 2).reshape(batch_size * num_mics, -1)

    mixture_stft = torch.stft(
        mixture_flat,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    freq_bins, frames = mixture_stft.shape[-2], mixture_stft.shape[-1]
    # Match train_light.build_stft_batch exactly, including sub-epsilon bins.
    scale = torch.abs(mixture_stft).clamp_min(EPS).pow(power - 1.0)
    two_channel = torch.stack([mixture_stft.real * scale, mixture_stft.imag * scale], dim=-1)
    two_channel = two_channel.view(batch_size, num_mics, freq_bins, frames, 2)
    return two_channel.permute(0, 3, 2, 1, 4).contiguous()



def reconstruct_waveform(
    estimate_ri: torch.Tensor,
    length: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    power: float,
    device: torch.device,
) -> torch.Tensor:
    # estimate_ri: (B, 2, T, F)
    window = get_window(win_length, device)

    est_real = estimate_ri[:, 0].permute(0, 2, 1).contiguous()  # (B, F, T)
    est_imag = estimate_ri[:, 1].permute(0, 2, 1).contiguous()  # (B, F, T)
    compressed = torch.complex(est_real, est_imag)

    if power <= 0:
        raise ValueError("power must be > 0")

    # Undo magnitude compression used during training.
    compressed_magnitude = torch.abs(compressed)
    # Invert the same piecewise compression: below EPS the training scale is constant.
    magnitude = torch.where(compressed_magnitude < EPS ** power,
                            compressed_magnitude * EPS ** (1.0 - power),
                            compressed_magnitude.pow(1.0 / power))
    phase = torch.angle(compressed)
    est_stft = magnitude * torch.exp(1j * phase)

    return torch.istft(
        est_stft,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        normalized=False,
        onesided=True,
        length=length,
    )



def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(device_arg)



def model_config_from_checkpoint(ckpt_args: Dict[str, object]) -> ModelConfig:
    cfg = ModelConfig()
    for field_name in cfg.__dataclass_fields__.keys():
        if field_name in ckpt_args:
            setattr(cfg, field_name, ckpt_args[field_name])
    return cfg



def create_model(cfg: ModelConfig, device: torch.device, candidate_id=None) -> EaBNet:
    constructor = EaBNet if candidate_id is None else partial(build_reference_candidate, candidate_id)
    model = constructor(
        k1=(2, 3),
        k2=(1, 3),
        c=int(cfg.channels),
        M=int(cfg.num_mics),
        embed_dim=int(cfg.embed_dim),
        kd1=int(cfg.kd1),
        cd1=int(cfg.cd1),
        d_feat=int(cfg.d_feat),
        p=int(cfg.p),
        q=int(cfg.q),
        is_causal=(str(cfg.is_causal) == "yes"),
        is_u2=(str(cfg.is_u2) == "yes"),
        bf_type=str(cfg.bf_type),
        topo_type=str(cfg.topo_type),
        intra_connect=str(cfg.intra_connect),
        norm_type=str(cfg.norm_type),
        dfsmn_layers=int(cfg.dfsmn_layers),
        dfsmn_memory_size=int(cfg.dfsmn_memory_size),
    ).to(device)
    model.eval()
    return model



def load_model(checkpoint_path: Path, device: torch.device, expected_candidate=None):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint must contain model_state_dict and saved training args; architecture cannot be guessed")
    saved = checkpoint.get("args")
    if not isinstance(saved, dict):
        raise ValueError("Checkpoint must contain saved training args")
    candidate_id = None
    if expected_candidate is not None or any(key.startswith("candidate_") or key == "structure_candidate" for key in saved):
        checkpoint, candidate_id = read_candidate_checkpoint(checkpoint_path, expected_candidate)
        saved = checkpoint["args"]
    else:
        required = ("channels", "num_mics", "embed_dim", "cd1", "dfsmn_layers", "dfsmn_memory_size",
                    "norm_type", "bf_type", "topo_type", "is_causal")
        missing = [name for name in required if name not in saved]
        if missing:
            raise ValueError(f"Checkpoint lacks model configuration: {', '.join(missing)}")
    model = create_model(model_config_from_checkpoint(saved), device, candidate_id)
    stripped = {}
    for key, value in checkpoint["model_state_dict"].items():
        name = key[len("module."):] if key.startswith("module.") else key
        if name in stripped:
            raise ValueError(f"Duplicate model key after removing module. prefix: {name}")
        stripped[name] = value
    # Never silently reshape/pad/trim weights to fit a different architecture.
    model.load_state_dict(stripped, strict=True)
    return model, saved


def resolve_frontend(args, checkpoint_args):
    """Use saved preprocessing; require explicit values for missing metadata."""
    config = {}
    for name in FRONTEND_FIELDS:
        supplied = getattr(args, name, None)
        saved = checkpoint_args.get(name)
        if saved is None and supplied is None:
            raise ValueError(f"Checkpoint lacks {name}; provide --{name.replace('_', '-')} explicitly")
        if supplied is not None and saved is not None and supplied != saved:
            raise ValueError(f"{name} differs from checkpoint: requested={supplied}, checkpoint={saved}")
        value = saved if saved is not None else supplied
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Invalid {name}: {value!r}")
        if name != "power" and value != int(value):
            raise ValueError(f"{name} must be an integer")
        config[name] = float(value) if name == "power" else int(value)
    if config["n_fft"] != 512:
        raise ValueError("EaBNet_light requires n_fft=512 (257 frequency bins)")
    if config["sample_rate"] not in (8000, 16000):
        raise ValueError("PESQ supports only saved sample_rate=8000 or 16000")
    if not 0 < config["hop_length"] <= config["win_length"] <= config["n_fft"]:
        raise ValueError("Require 0 < hop_length <= win_length <= n_fft")
    if config["power"] <= 0 or config["target_ref_mic"] < 0:
        raise ValueError("power must be positive and target_ref_mic nonnegative")
    return config



def validate_signal(signal, name):
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name}: expected a nonempty mono signal")
    if not np.isfinite(values).all():
        raise ValueError(f"{name}: non-finite samples")
    if np.mean(values * values) <= SILENCE_MEAN_SQUARE:
        raise ValueError(f"{name}: silent signal (mean square <= {SILENCE_MEAN_SQUARE})")
    return values



def compute_si_snr_db(reference, estimate):
    """Zero-mean scale-invariant SNR; epsilon is explicit in the run report."""
    reference = validate_signal(reference, "reference")
    estimate = validate_signal(estimate, "estimate")
    if reference.shape != estimate.shape:
        raise ValueError("SI-SNR requires equal-length signals")
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    energy = np.dot(reference, reference)
    if energy / reference.size <= SILENCE_MEAN_SQUARE:
        raise ValueError("reference: silent after mean removal")
    if np.mean(estimate * estimate) <= SILENCE_MEAN_SQUARE:
        raise ValueError("estimate: silent after mean removal")
    projection = np.dot(estimate, reference) / energy * reference
    residual = estimate - projection
    return float(10.0 * np.log10((np.dot(projection, projection) + SI_SNR_EPSILON)
                                 / (np.dot(residual, residual) + SI_SNR_EPSILON)))



def compute_paper_metrics(sample_rate, reference, estimate):
    reference = validate_signal(reference, "reference")
    estimate = validate_signal(estimate, "estimate")
    if sample_rate not in (8000, 16000):
        raise ValueError("PESQ supports only 8 or 16 kHz")
    if reference.shape != estimate.shape:
        raise ValueError("Metrics require equal-length signals")
    # pystoi warns and returns a sentinel for too few active frames. Treat that
    # as an evaluation failure, rather than including the sentinel in a mean.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = {
            "pesq": float(pesq(sample_rate, reference, estimate, "wb" if sample_rate == 16000 else "nb")),
            "stoi": float(stoi(reference, estimate, sample_rate, extended=False)),
            "estoi": float(stoi(reference, estimate, sample_rate, extended=True)),
            "si_snr_db": compute_si_snr_db(reference, estimate),
        }
    result["sdr_db"] = compute_sdr_db(reference, estimate)
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Metric implementation returned a non-finite value")
    return result



def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def package_versions():
    return {name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "soundfile", "pesq", "pystoi", "mir_eval")}



def compute_sdr_db(reference, estimate):
    """Original script's single-source BSS-eval SDR (512-tap distortion filter)."""
    reference = validate_signal(reference, "reference")
    estimate = validate_signal(estimate, "estimate")
    if reference.shape != estimate.shape:
        raise ValueError("SDR requires equal-length signals")
    with warnings.catch_warnings():
        # mir_eval deprecates this API; keep the requested metric definition.
        warnings.simplefilter("ignore", FutureWarning)
        sdr, _, _, _ = bss_eval_sources(reference[None, :], estimate[None, :], compute_permutation=False)
    return float(sdr[0])


def compute_energy(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    x64 = x.astype(np.float64)
    return float(np.sum(x64 * x64))



def match_estimate_level(ref: np.ndarray, est: np.ndarray, max_gain_db: float) -> Tuple[np.ndarray, float]:
    """Scale estimate to reference level with bounded gain for stable evaluation."""
    ref64 = ref.astype(np.float64)
    est64 = est.astype(np.float64)

    # Least-squares gain (projection of ref onto est).
    gain = float(np.dot(ref64, est64) / (np.dot(est64, est64) + EPS))

    # Fallback to RMS ratio if projection gain is non-positive.
    if gain <= 0:
        ref_rms = math.sqrt(float(np.mean(ref64 * ref64)) + EPS)
        est_rms = math.sqrt(float(np.mean(est64 * est64)) + EPS)
        gain = ref_rms / (est_rms + EPS)

    max_gain = 10.0 ** (max_gain_db / 20.0)
    min_gain = 1.0 / max_gain
    gain = min(max(gain, min_gain), max_gain)
    return est64 * gain, gain



def save_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate, subtype="FLOAT")



def _audio_outputs(estimate_dir, sample_id):
    return {kind: (estimate_dir / kind / f"sample_{sample_id:08d}_{kind}.wav").resolve()
            for kind in ("mixture", "estimate", "target")}


def _output_plan(args, records, protected_inputs, input_info):
    """Validate every planned output before opening files or allocating the model."""
    estimate_dir = Path(args.estimate_dir).expanduser().resolve()
    metadata_path = (estimate_dir / "metadata.csv").resolve()
    json_path = Path(args.save_json).expanduser().resolve() if args.save_json else (estimate_dir / "summary.json").resolve()
    csv_path = Path(args.save_csv).expanduser().resolve() if args.save_csv else None
    if args.max_samples < 0:
        raise ValueError("max_samples must be nonnegative; use 0 for all pairs")
    if len({record.sample_id for record in records}) != len(records):
        raise ValueError("metadata.csv contains duplicate sample_id values")
    if not math.isfinite(args.max_level_gain_db) or not 0 <= args.max_level_gain_db <= 120:
        raise ValueError("max_level_gain_db must be finite and in [0, 120]")
    if input_info["mode"] == "directories":
        for name in ("mixture_path", "target_path"):
            source_root = Path(input_info[name])
            if estimate_dir == source_root or source_root in estimate_dir.parents:
                raise ValueError("estimate-dir must be outside input audio directories to avoid contaminating future pairing")
    protected = {Path(args.checkpoint).expanduser().resolve(), *(path.resolve() for path in protected_inputs)}
    protected.update(path.resolve() for record in records for path in (record.mixture_path, record.target_path))
    selected = records[:args.max_samples] if args.max_samples > 0 else records
    outputs = [metadata_path, json_path]
    if csv_path is not None and csv_path != metadata_path:
        outputs.append(csv_path)
    if args.save_samples == "yes":
        for record in selected:
            outputs.extend(_audio_outputs(estimate_dir, record.sample_id).values())
    if len(set(outputs)) != len(outputs):
        raise ValueError("Output paths must be distinct")
    output_set = set(outputs)
    for path in outputs:
        if path in protected:
            raise ValueError(f"Output paths must not overwrite input audio, checkpoint or metadata.csv: {path}")
        if path.is_dir():
            raise ValueError(f"Output file path is an existing directory: {path}")
        if any(parent in output_set or parent.is_file() for parent in path.parents):
            raise ValueError(f"Output file/directory path collision: {path}")
    return selected, estimate_dir, metadata_path, csv_path, json_path


def evaluate(args):
    records, input_info, protected_inputs = resolve_evaluation_inputs(args)
    records, estimate_dir, metadata_path, csv_path, json_path = _output_plan(args, records, protected_inputs, input_info)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    device = get_device(args.device)
    if args.candidate is None:
        model, saved = load_model(checkpoint, device)
    else:
        model, saved = load_model(checkpoint, device, args.candidate)
    config = resolve_frontend(args, saved)
    num_mics = int(model.M)
    if saved.get("num_mics", num_mics) != num_mics:
        raise ValueError("Model microphone count differs from checkpoint")
    if not 0 <= args.mixture_ref_mic < num_mics:
        raise ValueError(f"mixture_ref_mic must be in [0, {num_mics})")
    model.eval()
    gain_matching = args.match_estimate_level == "yes"
    report = {
        "status": "running", "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "dataset": input_info.get("dataset", input_info.get("mixture_path")),
        "metadata_sha256": input_info["metadata_sha256"], "input_source": input_info,
        "selected_pairs_sha256": pairs_sha256(records),
        "selected_pairs_sha256_scope": "ordered sample IDs and resolved paths; not audio contents",
        "frontend": config, "num_mics": num_mics,
        "frontend_protocol": {"window": "periodic Hann", "center": True, "pad_mode": "reflect",
                              "normalized": False, "onesided": True, "waveform_normalization": "none",
                              "compression": "Z * abs(Z).clamp_min(1e-8) ** (power - 1)",
                              "inverse_compression": "piecewise exact inverse including the epsilon region",
                              "mixture_channels": f"first {num_mics} channels, original order",
                              "evaluation_extent": "full common-length utterance; no training crop or batch padding",
                              "inference_precision": "float32"},
        "checkpoint_model_config": {name: saved[name] for name in ModelConfig.__dataclass_fields__ if name in saved},
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "structure_candidate": saved.get("structure_candidate"),
        "candidate_implementation_sha256": saved.get("candidate_implementation_sha256"),
        "candidate_evidence": saved.get("candidate_evidence"), "exact_reproduction_verified": False,
        "mixture_ref_mic": args.mixture_ref_mic, "device": str(device),
        "expected_samples": len(records), "completed_samples": 0, "versions": package_versions(), "mean": None,
        "outputs": {"metadata_csv": str(metadata_path), "extra_csv": str(csv_path) if csv_path else None,
                    "summary_json": str(json_path), "estimate_dir": str(estimate_dir),
                    "save_samples": args.save_samples == "yes", "wav_subtype": "FLOAT",
                    "mixture_wav": "selected mixture_ref_mic (mono)",
                    "estimate_wav": "gain-matched estimate" if gain_matching else "raw model estimate"},
        "metric_protocol": {
            "pesq": "pesq wideband" if config["sample_rate"] == 16000 else "pesq narrowband",
            "estoi": "pystoi extended=True; JSON estoi=fraction, CSV estoi_pct=100*estoi",
            "stoi": "pystoi extended=False, fraction", "si_snr_db": "zero-mean projection SI-SNR",
            "si_snr_epsilon": SI_SNR_EPSILON,
            "sdr_db": "mir_eval.separation.bss_eval_sources, single source, 512-tap distortion filter; not SI-SNR",
            "target_gain_matching": gain_matching, "max_level_gain_db": args.max_level_gain_db,
            "gain_rule": "bounded least-squares projection, RMS fallback for nonpositive gain" if gain_matching else "none",
            "external_normalization_clipping_or_alignment": gain_matching,
            "length_policy": "truncate mixture and target to common length as in training Dataset; no delay correction",
            "aggregation": "paired arithmetic mean over every selected utterance; failures abort with no mean",
        },
        "limitations": ["Dataset and target convention must match the intended benchmark.",
                        "Gain matching, when enabled, uses the clean reference after inference and changes the evaluation protocol."],
    }
    print(f"Device: {device}; candidate: {saved.get('structure_candidate', 'EaBNet_light')}; samples: {len(records)}", flush=True)
    print("Training frontend: " + json.dumps(config) + f"; microphones={num_mics}; normalization=none", flush=True)
    print(f"Reference-dependent estimate gain matching: {gain_matching}", flush=True)
    fields = ["sample_id", "num_samples", "mixture_num_samples", "target_num_samples", "mixture_path", "target_path",
              "est_ref_energy_ratio", "mix_ref_energy_ratio", "estimate_level_gain", "pesq", "estoi_pct", "sdr_db",
              "pesq_mix", "estoi_mix_pct", "sdr_mix_db", "mixture_saved_path", "estimate_saved_path", "target_saved_path",
              "ref_energy", "est_energy", "mix_energy", "raw_est_energy"]
    fields += [f"{prefix}_{metric}" for prefix in ("enhanced", "noisy") for metric in METRICS]
    rows, current_id = [], None
    try:
        with ExitStack() as stack:
            writers = []
            for destination in dict.fromkeys(path for path in (metadata_path, csv_path) if path is not None):
                destination.parent.mkdir(parents=True, exist_ok=True)
                handle = stack.enter_context(destination.open("w", newline="", encoding="utf-8"))
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                handle.flush()
                writers.append((writer, handle))
            with torch.inference_mode():
                for record in tqdm(records, desc="Evaluating"):
                    current_id = record.sample_id
                    mixture, mix_sr = sf.read(record.mixture_path, dtype="float32", always_2d=True)
                    target, target_sr = sf.read(record.target_path, dtype="float32", always_2d=True)
                    if mix_sr != config["sample_rate"] or target_sr != config["sample_rate"]:
                        raise ValueError(f"sample_id={current_id}: sample rate mismatch; expected {config['sample_rate']}")
                    if mixture.shape[1] < num_mics:
                        raise ValueError(f"sample_id={current_id}: insufficient microphone channels, expected at least {num_mics}")
                    if target.shape[1] > 1 and config["target_ref_mic"] >= target.shape[1]:
                        raise ValueError(f"sample_id={current_id}: target_ref_mic out of range")
                    reference = target[:, config["target_ref_mic"] if target.shape[1] > 1 else 0]
                    mix_length, target_length = len(mixture), len(reference)
                    length = min(mix_length, target_length)
                    if length <= config["n_fft"] // 2:
                        raise ValueError(f"sample_id={current_id}: audio too short for centered STFT")
                    mixture, reference = mixture[:length, :num_mics], reference[:length]
                    if not np.isfinite(mixture).all():
                        raise ValueError(f"sample_id={current_id}: non-finite microphone samples")
                    validate_signal(reference, f"sample_id={current_id}, target")
                    noisy = validate_signal(mixture[:, args.mixture_ref_mic], f"sample_id={current_id}, mixture reference")
                    frontend = {key: config[key] for key in ("n_fft", "hop_length", "win_length", "power")}
                    features = build_stft_batch(torch.from_numpy(mixture).unsqueeze(0), device=device, **frontend)
                    estimate_ri = model(features)
                    if not torch.isfinite(estimate_ri).all():
                        raise ValueError(f"sample_id={current_id}: non-finite model output")
                    estimate = reconstruct_waveform(estimate_ri, length=length, device=device, **frontend).squeeze(0).cpu().numpy()
                    estimate = validate_signal(estimate, f"sample_id={current_id}, estimate")
                    raw_est_energy = compute_energy(estimate)
                    gain = 1.0
                    if gain_matching:
                        estimate, gain = match_estimate_level(reference, estimate, args.max_level_gain_db)
                    enhanced_metrics = compute_paper_metrics(config["sample_rate"], reference, estimate)
                    noisy_metrics = compute_paper_metrics(config["sample_rate"], reference, noisy)
                    paths = _audio_outputs(estimate_dir, current_id) if args.save_samples == "yes" else {}
                    if paths:
                        for kind, signal in (("mixture", noisy), ("estimate", estimate), ("target", reference)):
                            save_audio(paths[kind], signal.astype(np.float32), config["sample_rate"])
                    ref_energy, est_energy, mix_energy = map(compute_energy, (reference, estimate, noisy))
                    row = {
                        "sample_id": current_id, "num_samples": length,
                        "mixture_num_samples": mix_length, "target_num_samples": target_length,
                        "mixture_path": str(record.mixture_path), "target_path": str(record.target_path),
                        "estimate_level_gain": gain, "raw_est_energy": raw_est_energy,
                        "ref_energy": ref_energy, "est_energy": est_energy, "mix_energy": mix_energy,
                        "est_ref_energy_ratio": est_energy / (ref_energy + EPS), "mix_ref_energy_ratio": mix_energy / (ref_energy + EPS),
                        "pesq": enhanced_metrics["pesq"], "estoi_pct": enhanced_metrics["estoi"] * 100,
                        "sdr_db": enhanced_metrics["sdr_db"], "pesq_mix": noisy_metrics["pesq"],
                        "estoi_mix_pct": noisy_metrics["estoi"] * 100, "sdr_mix_db": noisy_metrics["sdr_db"],
                        **{f"{kind}_saved_path": str(paths[kind]) if paths else "" for kind in ("mixture", "estimate", "target")},
                        **{f"enhanced_{key}": value for key, value in enhanced_metrics.items()},
                        **{f"noisy_{key}": value for key, value in noisy_metrics.items()},
                    }
                    for writer, handle in writers:
                        writer.writerow(row)
                        handle.flush()
                    rows.append(row)
        report["status"] = "complete"
        report["mean"] = {prefix: {metric: float(np.mean([row[f"{prefix}_{metric}"] for row in rows]))
                                  for metric in METRICS} for prefix in ("enhanced", "noisy")}
    except Exception as exc:
        report.update(status="failed", failed_sample_id=current_id, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["completed_samples"] = len(rows)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report["mean"], indent=2))
    print(f"Per-sample metadata: {metadata_path}\nSummary: {json_path}")
    return report



def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--val-dir", help="Dataset directory containing metadata.csv; default ./validation_set")
    inputs.add_argument("--mixture-path", "--mixture-dir", dest="mixture_path", help="Noisy WAV/FLAC file or directory")
    parser.add_argument("--target-path", "--target-dir", dest="target_path", help="Paired clean reference file or directory")
    parser.add_argument("--mixture-suffix", default="")
    parser.add_argument("--target-suffix", default="")
    parser.add_argument("--checkpoint", default="./bestmodels_cbam_flat_projection64/best_model.pt")
    parser.add_argument("--candidate", choices=all_reference_candidates(), help="Optional expected candidate; auto-detected from checkpoint")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-samples", type=int, default=0, help="First N records; 0 means all")
    parser.add_argument("--mixture-ref-mic", type=int, default=0, help="Channel used for noisy metrics and saved mono mixture")
    parser.add_argument("--estimate-dir", default="./estimate_set_cbam_flat_projection64")
    parser.add_argument("--save-samples", choices=("yes", "no"), default="yes")
    parser.add_argument("--save-csv", default="", help="Optional additional per-sample CSV")
    parser.add_argument("--save-json", default="", help="Summary JSON; default estimate-dir/summary.json")
    parser.add_argument("--match-estimate-level", choices=("yes", "no"), default="no",
                        help="Post-inference clean-reference gain matching; changes metric/export protocol")
    parser.add_argument("--max-level-gain-db", type=float, default=20.0)
    for name in FRONTEND_FIELDS:
        parser.add_argument(f"--{name.replace('_', '-')}", type=float if name == "power" else int,
                            default=None, help="Read from checkpoint; explicit value must agree")
    args = parser.parse_args(argv)
    if args.mixture_path and not args.target_path:
        parser.error("--mixture-path requires --target-path")
    if not args.mixture_path and (args.target_path or args.mixture_suffix or args.target_suffix):
        parser.error("--target-path and suffixes require --mixture-path")
    if not args.mixture_path and not args.val_dir:
        args.val_dir = "./validation_set"
    if args.max_samples < 0:
        parser.error("--max-samples must be nonnegative; use 0 for all pairs")
    if not math.isfinite(args.max_level_gain_db) or not 0 <= args.max_level_gain_db <= 120:
        parser.error("--max-level-gain-db must be finite and in [0, 120]")
    return args


def main(argv=None):
    return evaluate(parse_args(argv))


if __name__ == "__main__":
    main()

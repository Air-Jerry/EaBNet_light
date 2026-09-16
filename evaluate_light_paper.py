"""Strict, paired evaluation using the paper's four named metrics.

The old evaluator and training pipeline are unchanged. This script uses their
metadata.csv interface and model/STFT helpers, but never uses target-dependent
gain matching. STOI/E-STOI are fractions; SI-SNR is zero-mean, in dB. The paper
does not identify its metric implementations, so exact score equivalence is not
claimed. A failed item fails the run; partial results are never averaged.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pesq import pesq
from pystoi import stoi
from tqdm import tqdm

from evaluate_light import (
    build_stft_batch,
    get_device,
    load_model,
    load_records,
    reconstruct_waveform,
)


FRONTEND_FIELDS = ("sample_rate", "n_fft", "hop_length", "win_length", "power", "target_ref_mic")
PAPER_FRONTEND = {"sample_rate": 16000, "n_fft": 512, "hop_length": 160, "win_length": 320}
METRICS = ("pesq", "stoi", "estoi", "si_snr_db")
SILENCE_MEAN_SQUARE = 1e-12
SI_SNR_EPSILON = 1e-8


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
    for name, expected in PAPER_FRONTEND.items():
        if config[name] != expected:
            raise ValueError(f"Paper evaluation requires {name}={expected}; checkpoint uses {config[name]}")
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
    if sample_rate != 16000:
        raise ValueError("Paper evaluation requires 16 kHz")
    if reference.shape != estimate.shape:
        raise ValueError("Metrics require equal-length signals")
    # pystoi warns and returns a sentinel for too few active frames. Treat that
    # as an evaluation failure, rather than including the sentinel in a mean.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = {
            "pesq": float(pesq(sample_rate, reference, estimate, "wb")),
            "stoi": float(stoi(reference, estimate, sample_rate, extended=False)),
            "estoi": float(stoi(reference, estimate, sample_rate, extended=True)),
            "si_snr_db": compute_si_snr_db(reference, estimate),
        }
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
            for name in ("torch", "numpy", "soundfile", "pesq", "pystoi")}


def evaluate(args):
    device = get_device(args.device)
    checkpoint = Path(args.checkpoint).resolve()
    dataset = Path(args.val_dir).resolve()
    csv_path, json_path = Path(args.save_csv).resolve(), Path(args.save_json).resolve()
    protected_inputs = {checkpoint, dataset / "metadata.csv"}
    if csv_path == json_path or csv_path in protected_inputs or json_path in protected_inputs:
        raise ValueError("Output paths must be distinct and must not overwrite checkpoint or metadata.csv")
    model, checkpoint_args = load_model(checkpoint, device)
    config = resolve_frontend(args, checkpoint_args)
    num_mics = int(model.M)
    if not 0 <= args.mixture_ref_mic < num_mics:
        raise ValueError(f"mixture_ref_mic must be in [0, {num_mics})")
    records = load_records(dataset)
    audio_inputs = {path.resolve() for record in records
                    for path in (record.mixture_path, record.target_path)}
    if csv_path in audio_inputs or json_path in audio_inputs:
        raise ValueError("Output paths must not overwrite input audio")
    if len({record.sample_id for record in records}) != len(records):
        raise ValueError("metadata.csv contains duplicate sample_id values")
    if args.max_samples > 0:
        records = records[:args.max_samples]
    report = {
        "status": "running", "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint), "dataset": str(dataset),
        "metadata_sha256": sha256_file(dataset / "metadata.csv"),
        "frontend": config, "num_mics": num_mics,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint_model_config": {
            name: checkpoint_args[name] for name in (
                "channels", "embed_dim", "cd1", "dfsmn_layers", "dfsmn_memory_size",
                "norm_type", "bf_type", "topo_type", "is_causal"
            ) if name in checkpoint_args
        },
        "exact_reproduction_verified": False,
        "mixture_ref_mic": args.mixture_ref_mic, "device": str(device),
        "expected_samples": len(records), "completed_samples": 0,
        "versions": package_versions(), "mean": None,
        "metric_protocol": {
            "pesq": "pesq package, wideband 16 kHz",
            "stoi": "pystoi extended=False, fraction",
            "estoi": "pystoi extended=True, fraction",
            "si_snr_db": "zero-mean projection SI-SNR, 10*log10((projection_energy+epsilon)/(residual_energy+epsilon))",
            "si_snr_epsilon": SI_SNR_EPSILON,
            "silence_mean_square_threshold": SILENCE_MEAN_SQUARE,
            "target_gain_matching": False,
            "external_normalization_clipping_or_alignment": False,
            "aggregation": "arithmetic mean over the same successfully evaluated utterances; failures abort with no mean",
            "length_policy": "truncate mixture/target to their common length, as in the existing dataset interface",
        },
        "limitations": [
            "The paper does not disclose metric software versions, exact SI-SNR convention or PESQ mode.",
            "Power compression is inherited from the checkpoint; it is not specified in this paper.",
            "Comparison with Table 1 requires the exact ConferencingSpeech2021 development test set and target convention.",
        ],
    }
    rows = []
    current_id = None
    try:
        with torch.inference_mode():
            for record in tqdm(records, desc="Paper metrics"):
                current_id = record.sample_id
                mixture, mix_sr = sf.read(record.mixture_path, dtype="float32", always_2d=True)
                target, tgt_sr = sf.read(record.target_path, dtype="float32", always_2d=True)
                if mix_sr != config["sample_rate"] or tgt_sr != config["sample_rate"]:
                    raise ValueError(f"sample_id={current_id}: sample rate mismatch")
                if mixture.shape[1] < num_mics:
                    raise ValueError(f"sample_id={current_id}: insufficient microphone channels")
                if target.shape[1] > 1:
                    if config["target_ref_mic"] >= target.shape[1]:
                        raise ValueError(f"sample_id={current_id}: target_ref_mic out of range")
                    reference = target[:, config["target_ref_mic"]]
                else:
                    reference = target[:, 0]
                mix_length, target_length = len(mixture), len(reference)
                length = min(mix_length, target_length)
                if length <= config["n_fft"] // 2:
                    raise ValueError(f"sample_id={current_id}: audio too short for centered STFT")
                mixture, reference = mixture[:length, :num_mics], reference[:length]
                for mic in range(num_mics):
                    validate_signal(mixture[:, mic], f"sample_id={current_id}, microphone={mic}")
                validate_signal(reference, f"sample_id={current_id}, target")
                frontend = {key: config[key] for key in ("n_fft", "hop_length", "win_length", "power")}
                features = build_stft_batch(torch.from_numpy(mixture).unsqueeze(0), device=device, **frontend)
                estimate_ri = model(features)
                if not torch.isfinite(estimate_ri).all():
                    raise ValueError(f"sample_id={current_id}: non-finite model output")
                estimate = reconstruct_waveform(estimate_ri, length=length, device=device, **frontend)
                estimate = estimate.squeeze(0).cpu().numpy()
                enhanced_metrics = compute_paper_metrics(config["sample_rate"], reference, estimate)
                noisy_metrics = compute_paper_metrics(config["sample_rate"], reference, mixture[:, args.mixture_ref_mic])
                rows.append({
                    "sample_id": current_id, "num_samples": length,
                    "mixture_num_samples": mix_length, "target_num_samples": target_length,
                    "mixture_path": str(record.mixture_path), "target_path": str(record.target_path),
                    **{f"enhanced_{key}": value for key, value in enhanced_metrics.items()},
                    **{f"noisy_{key}": value for key, value in noisy_metrics.items()},
                })
        report["status"] = "complete"
        report["mean"] = {
            prefix: {metric: float(np.mean([row[f"{prefix}_{metric}"] for row in rows])) for metric in METRICS}
            for prefix in ("enhanced", "noisy")
        }
    except Exception as exc:
        report.update(status="failed", failed_sample_id=current_id, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["completed_samples"] = len(rows)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["sample_id", "num_samples", "mixture_num_samples", "target_num_samples", "mixture_path", "target_path"]
        fields += [f"{prefix}_{metric}" for prefix in ("enhanced", "noisy") for metric in METRICS]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report["mean"], indent=2))
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-dir", required=True, help="Directory containing the existing metadata.csv interface")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--mixture-ref-mic", type=int, default=0)
    parser.add_argument("--save-csv", default="./logs/paper_metrics.csv")
    parser.add_argument("--save-json", default="./logs/paper_metrics.json")
    for name in FRONTEND_FIELDS:
        parser.add_argument(f"--{name.replace('_', '-')}", type=float if name == "power" else int,
                            default=None, help="Must match checkpoint; required if missing from checkpoint")
    return parser.parse_args(argv)


if __name__ == "__main__":
    evaluate(parse_args())

"""Evaluate CTS-Net with checkpoint-bound features and explicit metric failures.

The WSJ0 protocol is NB-PESQ, ESTOI (%) and BSS-Eval SDR, including at
16 kHz (paper footnote 6). DNS reports WB-/NB-PESQ, STOI (%) and SI-SDR.
VoiceBank CSIG/CBAK/COVL are not implemented by this script.

Input metadata retains the existing sample_id, mixture_path, target_path
interface. Only mixture channel zero is used. No reference-derived gain is
ever applied to the enhanced waveform.
"""

import argparse
import csv
import hashlib
import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from CTSNet import CTSNet
from cts_features import build_mixture_stft, reconstruct_waveform
from evaluate_light import (
    compute_energy,
    compute_sdr_db,
    get_device,
    load_records,
    pesq,
    save_audio,
    stoi,
)


@dataclass(frozen=True)
class FrontendConfig:
    sample_rate: int
    target_ref_mic: int
    num_mics: int
    n_fft: int
    hop_length: int
    win_length: int
    power: float

    @classmethod
    def from_checkpoint(cls, args: Dict[str, object]) -> "FrontendConfig":
        missing = sorted(set(cls.__dataclass_fields__) - set(args))
        if missing:
            raise ValueError(
                "Checkpoint does not record the complete frontend; refusing to guess: "
                + ", ".join(missing)
            )
        values = {name: args[name] for name in cls.__dataclass_fields__}
        for name in values:
            value = values[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Invalid checkpoint frontend {name}={value!r}")
            if name != "power" and (not math.isfinite(value) or int(value) != value):
                raise ValueError(f"Checkpoint frontend {name} must be an integer")
            values[name] = float(value) if name == "power" else int(value)
        cfg = cls(**values)
        if cfg.sample_rate != 16000 or cfg.n_fft != 320 or cfg.num_mics != 1:
            raise ValueError("CTS-Net requires sample_rate=16000, n_fft=320 and num_mics=1")
        if cfg.hop_length != 160 or cfg.win_length != 320:
            raise ValueError("CTS-Net requires hop_length=160 and win_length=320")
        if cfg.target_ref_mic < 0 or not math.isfinite(cfg.power) or not 0 < cfg.power <= 1:
            raise ValueError("Require target_ref_mic >= 0 and a finite power in (0, 1]")
        return cfg


def _checkpoint_bool(value: object) -> bool:
    if value is True or value == "yes":
        return True
    if value is False or value == "no":
        return False
    raise ValueError(f"Invalid checkpoint is_causal value: {value!r}")


def load_model(
    checkpoint_path: Path, device: torch.device
) -> Tuple[torch.nn.Module, FrontendConfig, Dict[str, object]]:
    # Training checkpoints contain optimizer/RNG metadata in addition to tensors.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("model_name") != "CTSNet":
        raise ValueError("Expected a CTSNet training checkpoint with model_name='CTSNet'")
    args = checkpoint.get("args")
    if not isinstance(args, dict):
        raise ValueError("CTSNet checkpoint must include its training args dictionary")
    if checkpoint.get("stage") != "joint":
        raise ValueError(
            "Full CTS-Net evaluation requires a joint-stage checkpoint; "
            "ME pretraining leaves the complex-refinement network untrained"
        )
    cfg = FrontendConfig.from_checkpoint(args)
    missing = {"is_causal", "norm_type"} - set(args)
    if missing:
        raise ValueError(f"Missing checkpoint model configuration: {sorted(missing)}")
    manifest = checkpoint.get("reproducibility_manifest") or {}
    code_hashes = manifest.get("code_sha256", {})
    verified_source_files = []
    for filename in ("CTSNet.py", "cts_features.py"):
        if filename in code_hashes:
            path = Path(__file__).resolve().parent / filename
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_hash != code_hashes[filename]:
                raise ValueError(
                    f"{filename} differs from the checkpoint source manifest; "
                    "restore the training implementation before evaluation"
                )
            verified_source_files.append(filename)
    model = CTSNet(
        is_causal=_checkpoint_bool(args["is_causal"]),
        norm_type=str(args["norm_type"]),
        return_auxiliary=False,
    ).to(device)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint must include model_state_dict")
    stripped = {key.removeprefix("module."): value for key, value in state_dict.items()}
    if len(stripped) != len(state_dict):
        raise ValueError("Checkpoint contains conflicting module-prefixed parameter names")
    model.load_state_dict(stripped, strict=True)
    model.eval()
    provenance = {
        "model_name": checkpoint["model_name"],
        "stage": checkpoint["stage"],
        "epoch": checkpoint.get("epoch"),
        "stage_epoch": checkpoint.get("stage_epoch"),
        "is_causal": args["is_causal"],
        "norm_type": args["norm_type"],
        "verified_source_files": verified_source_files,
        "reproducibility_manifest": checkpoint.get("reproducibility_manifest"),
    }
    return model, cfg, provenance


def validate_overrides(args: argparse.Namespace, cfg: FrontendConfig) -> None:
    """Accept old frontend CLI flags only when they agree with the checkpoint."""
    for name in cfg.__dataclass_fields__:
        supplied = getattr(args, name, None)
        expected = getattr(cfg, name)
        if supplied is not None and supplied != expected:
            raise ValueError(
                f"--{name.replace('_', '-')}={supplied} disagrees with checkpoint "
                f"value {expected}; evaluation must use the training frontend"
            )


def compute_si_sdr_db(ref: np.ndarray, deg: np.ndarray) -> float:
    """Zero-mean, scale-invariant SDR, distinct from BSS-Eval SDR."""
    ref = ref - ref.mean()
    deg = deg - deg.mean()
    ref_energy = float(np.dot(ref, ref))
    if ref_energy <= 0:
        raise ValueError("SI-SDR requires a nonconstant reference")
    projection = np.dot(deg, ref) * ref / ref_energy
    noise = deg - projection
    signal_energy = float(np.dot(projection, projection))
    noise_energy = float(np.dot(noise, noise))
    if signal_energy <= 0 or noise_energy <= 0:
        raise ValueError("SI-SDR is nonfinite for zero projection or zero residual")
    return float(10.0 * np.log10(signal_energy / noise_energy))


def compute_metrics(
    sample_rate: int,
    ref: np.ndarray,
    deg: np.ndarray,
    protocol: str = "wsj0",
    pesq_mode: Optional[str] = None,
) -> Dict[str, float]:
    """All requested scores must succeed on every sample; no NaN filtering."""
    ref = np.asarray(ref, dtype=np.float64)
    deg = np.asarray(deg, dtype=np.float64)
    if ref.ndim != 1 or ref.shape != deg.shape or ref.size == 0:
        raise ValueError("Metrics require nonempty, equally sized mono waveforms")
    if not np.isfinite(ref).all() or not np.isfinite(deg).all():
        raise ValueError("Metrics received nonfinite waveform samples")
    if not np.any(ref) or not np.any(deg):
        raise ValueError("Metrics are undefined for an entirely silent waveform")
    if protocol == "wsj0":
        mode = pesq_mode or "nb"
        scorers = {
            "pesq": lambda: pesq(sample_rate, ref, deg, mode),
            "estoi_pct": lambda: 100.0 * stoi(ref, deg, sample_rate, extended=True),
            "sdr_db": lambda: compute_sdr_db(ref, deg),
        }
    elif protocol == "dns":
        if pesq_mode is not None:
            raise ValueError("DNS reports both PESQ modes; do not specify --pesq-mode")
        scorers = {
            "pesq_wb": lambda: pesq(sample_rate, ref, deg, "wb"),
            "pesq_nb": lambda: pesq(sample_rate, ref, deg, "nb"),
            "stoi_pct": lambda: 100.0 * stoi(ref, deg, sample_rate, extended=False),
            "si_sdr_db": lambda: compute_si_sdr_db(ref, deg),
        }
    else:
        raise ValueError(f"Unsupported evaluation protocol: {protocol}")
    results = {}
    for name, score in scorers.items():
        try:
            with warnings.catch_warnings():
                # pystoi warns and returns a sentinel on insufficient speech.
                warnings.simplefilter("error", UserWarning)
                warnings.simplefilter("error", RuntimeWarning)
                warnings.simplefilter("ignore", FutureWarning)
                value = float(score())
            if not math.isfinite(value):
                raise ValueError(f"returned nonfinite value {value}")
        except Exception as exc:
            raise RuntimeError(f"{name} failed: {exc}") from exc
        results[name] = value
    return results


def read_waveforms(record, cfg: FrontendConfig) -> Tuple[np.ndarray, np.ndarray]:
    mixture, mix_sr = sf.read(record.mixture_path, dtype="float32", always_2d=True)
    target, target_sr = sf.read(record.target_path, dtype="float32", always_2d=True)
    if mix_sr != cfg.sample_rate or target_sr != cfg.sample_rate:
        raise ValueError(
            f"sample_id={record.sample_id}: sample rates mix={mix_sr}, target={target_sr}; "
            f"checkpoint requires {cfg.sample_rate}"
        )
    if target.shape[1] <= cfg.target_ref_mic:
        raise ValueError(f"sample_id={record.sample_id}: target_ref_mic out of range")
    if mixture.shape[0] != target.shape[0] or mixture.shape[0] == 0:
        raise ValueError(
            f"sample_id={record.sample_id}: paired waveform lengths must agree and be "
            f"nonzero; mixture={mixture.shape[0]}, target={target.shape[0]}"
        )
    mixture = mixture[:, :1].copy()
    target = target[:, cfg.target_ref_mic].copy()
    if not np.isfinite(mixture).all() or not np.isfinite(target).all():
        raise ValueError(f"sample_id={record.sample_id}: nonfinite audio")
    return mixture, target


def _baseline_key(key: str) -> str:
    if key.endswith("_pct"):
        return key[:-4] + "_mix_pct"
    if key.endswith("_db"):
        return key[:-3] + "_mix_db"
    return key + "_mix"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-dir", default="./validation_set")
    parser.add_argument("--checkpoint", default="./bestmodels_cts/best_model.pt")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--protocol", choices=["wsj0", "dns"], default="wsj0")
    parser.add_argument(
        "--pesq-mode", choices=["nb", "wb"], default=None,
        help="WSJ0 default: nb (paper footnote 6); wb is a non-paper variant",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="First N; zero means all")
    parser.add_argument("--save-csv", default="")
    parser.add_argument("--estimate-dir", default="./estimate_set_cts")
    parser.add_argument("--save-samples", choices=["yes", "no"], default="yes")
    # These familiar flags can verify a checkpoint, but cannot override its frontend.
    for name in ("sample-rate", "target-ref-mic", "num-mics", "n-fft", "hop-length", "win-length"):
        parser.add_argument(f"--{name}", type=int, default=None)
    parser.add_argument("--power", type=float, default=None)
    parser.add_argument("--mixture-ref-mic", type=int, choices=[0], default=0)
    parser.add_argument("--match-estimate-level", choices=["no"], default="no")
    args = parser.parse_args(argv)
    if args.max_samples < 0:
        parser.error("--max-samples must be >= 0")
    if args.protocol == "dns" and args.pesq_mode is not None:
        parser.error("DNS reports both PESQ modes; omit --pesq-mode")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    device = get_device(args.device)
    checkpoint_path = Path(args.checkpoint).resolve()
    model, cfg, provenance = load_model(checkpoint_path, device)
    validate_overrides(args, cfg)
    records = load_records(Path(args.val_dir).resolve())
    sample_ids = [record.sample_id for record in records]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Duplicate sample_id in metadata; output paths would collide")
    total_records = len(records)
    if args.max_samples:
        records = records[:args.max_samples]
    output_dir = Path(args.estimate_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "evaluation_status.json"
    metadata_path = output_dir / "metadata.csv"
    status = {
        "completed": False,
        "status": "running",
        "checkpoint": str(checkpoint_path),
        "checkpoint_provenance": provenance,
        "frontend": asdict(cfg),
        "protocol": args.protocol,
        "pesq_mode": (args.pesq_mode or "nb") if args.protocol == "wsj0" else "wb+nb",
        "paper_metric_protocol": args.protocol != "wsj0" or args.pesq_mode != "wb",
        "mixture_channel": 0,
        "reference_level_matching": False,
        "dataset_records": total_records,
        "requested_samples": len(records),
        "evaluated_samples": 0,
        "is_subset": len(records) < total_records,
        "metrics": {},
    }

    def write_status() -> None:
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

    write_status()
    print(f"Device: {device}; checkpoint: {checkpoint_path}")
    print(f"Frontend (from checkpoint): {asdict(cfg)}")
    print(f"Protocol: {args.protocol}; PESQ: {status['pesq_mode']}; samples: {len(records)}")
    if not status["paper_metric_protocol"]:
        print("WB-PESQ is not comparable to the paper's WSJ0 NB-PESQ tables.")
    rows = []
    metric_keys = None
    try:
        with metadata_path.open("w", newline="", encoding="utf-8") as handle, torch.inference_mode():
            writer = None
            for record in tqdm(records, desc="Evaluating CTS-Net"):
                mixture, target = read_waveforms(record, cfg)
                mix_stft = build_mixture_stft(
                    mixture=torch.from_numpy(mixture).unsqueeze(0),
                    n_fft=cfg.n_fft, hop_length=cfg.hop_length,
                    win_length=cfg.win_length, power=cfg.power, device=device,
                )
                estimate_ri = model(mix_stft)
                estimate = reconstruct_waveform(
                    estimate_ri=estimate_ri, length=len(target),
                    n_fft=cfg.n_fft, hop_length=cfg.hop_length,
                    win_length=cfg.win_length, power=cfg.power, device=device,
                ).squeeze(0).detach().cpu().numpy().astype(np.float64)
                reference = target.astype(np.float64)
                baseline = mixture[:, 0].astype(np.float64)
                try:
                    enhanced_metrics = compute_metrics(cfg.sample_rate, reference, estimate, args.protocol, args.pesq_mode)
                except Exception as exc:
                    raise RuntimeError(f"sample_id={record.sample_id}, enhanced: {exc}") from exc
                try:
                    mixture_metrics = compute_metrics(cfg.sample_rate, reference, baseline, args.protocol, args.pesq_mode)
                except Exception as exc:
                    raise RuntimeError(f"sample_id={record.sample_id}, mixture: {exc}") from exc
                scores = dict(enhanced_metrics)
                scores.update({_baseline_key(key): value for key, value in mixture_metrics.items()})
                metric_keys = list(scores)
                ref_energy = compute_energy(reference)
                est_energy = compute_energy(estimate)
                mix_energy = compute_energy(baseline)
                row = {
                    "sample_id": record.sample_id,
                    "mixture_path": str(record.mixture_path),
                    "target_path": str(record.target_path),
                    "est_ref_energy_ratio": est_energy / ref_energy,
                    "mix_ref_energy_ratio": mix_energy / ref_energy,
                    "estimate_level_gain": 1.0,
                    **scores,
                    "mixture_saved_path": "",
                    "estimate_saved_path": "",
                    "target_saved_path": "",
                    "ref_energy": ref_energy,
                    "est_energy": est_energy,
                    "mix_energy": mix_energy,
                }
                if args.save_samples == "yes":
                    for kind, audio in (("mixture", baseline), ("estimate", estimate), ("target", reference)):
                        audio_path = output_dir / kind / f"sample_{record.sample_id:08d}_{kind}.wav"
                        save_audio(audio_path, audio.astype(np.float32), cfg.sample_rate)
                        row[f"{kind}_saved_path"] = str(audio_path)
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                handle.flush()
                rows.append(row)
                status["evaluated_samples"] = len(rows)
        if len(rows) != len(records) or not rows:
            raise RuntimeError("Evaluation did not produce every requested sample")
        if args.save_csv:
            csv_path = Path(args.save_csv).resolve()
            if csv_path != metadata_path:
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                with csv_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
        status["metrics"] = {key: float(np.mean([row[key] for row in rows])) for key in metric_keys}
        status["completed"] = True
        status["status"] = "complete"
        write_status()
    except Exception as exc:
        status["status"] = "failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        write_status()
        raise
    print(f"Completed {len(rows)}/{len(records)} samples; all metrics use the same denominator.")
    for key, value in status["metrics"].items():
        print(f"{key}: {value:.4f}")
    print(f"Per-sample results: {metadata_path}")
    print(f"Protocol and completion record: {status_path}")


if __name__ == "__main__":
    main()

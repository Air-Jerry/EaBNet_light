'''
~/miniconda3/envs/EaBNet/bin/python evaluate_light.py \
  --val-dir /data/ssd1/jinrui.yang/newset_rt60_005-015_on_NOISEX92/babble/training_set \
  --checkpoint ./bestmodels/best_model.pt \
  --estimate-dir /data/ssd1/jinrui.yang/estimate_set_rt60_005-015_on_NOISEX92_test_light/babble \
  --save-samples yes \
  --match-estimate-level no
'''

'''
# 只测前100条
~/miniconda3/envs/EaBNet/bin/python evaluate_validation.py --max-samples 100

# 保存逐样本指标
~/miniconda3/envs/EaBNet/bin/python evaluate_validation.py --save-csv ./logs/val_metrics.csv
'''


import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from EaBNet_light import EaBNet


try:
    from pesq import pesq
except ImportError as exc:
    raise ImportError("Missing dependency: pesq. Install with `pip install pesq`.") from exc

try:
    from mir_eval.separation import bss_eval_sources
except ImportError as exc:
    raise ImportError("Missing dependency: mir_eval. Install with `pip install mir_eval`.") from exc

try:
    from pystoi import stoi
except ImportError as exc:
    raise ImportError("Missing dependency: pystoi. Install with `pip install pystoi`.") from exc


EPS = 1e-8
_WINDOW_CACHE: Dict[Tuple[int, str, int], torch.Tensor] = {}


@dataclass
class SampleRecord:
    sample_id: int
    mixture_path: Path
    target_path: Path


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


def resolve_record_path(root_dir: Path, relative_path: str) -> Path:
    normalized = relative_path.replace("\\", "/").strip()
    path_obj = Path(normalized)

    if path_obj.is_absolute() and path_obj.exists():
        return path_obj.resolve()

    # Case 1: path relative to val_dir (preferred layout).
    candidate = (root_dir / path_obj).resolve()
    if candidate.exists():
        return candidate

    # Case 2: path relative to current working directory / workspace root.
    cwd_candidate = Path(normalized).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    # Case 3: metadata path contains out-root + split dir prefix,
    # e.g. newset_xxx/training_set/mixture/a.wav while root_dir already points to training_set.
    parts = path_obj.parts
    for split_name in ("training_set", "validation_set"):
        if split_name in parts:
            idx = parts.index(split_name)
            tail = parts[idx + 1 :]
            if tail:
                split_candidate = (root_dir / Path(*tail)).resolve()
                if split_candidate.exists():
                    return split_candidate

    # Last-resort fallback by basename.
    fallback = (root_dir / path_obj.name).resolve()
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Cannot resolve path from metadata: {relative_path}")


def load_records(val_dir: Path) -> List[SampleRecord]:
    metadata_path = val_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found under {val_dir}")

    records: List[SampleRecord] = []
    with metadata_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            mixture_path = resolve_record_path(metadata_path.parent, row["mixture_path"])
            target_path = resolve_record_path(metadata_path.parent, row["target_path"])
            records.append(
                SampleRecord(
                    sample_id=int(row["sample_id"]),
                    mixture_path=mixture_path,
                    target_path=target_path,
                )
            )

    if not records:
        raise RuntimeError(f"No validation records found in {metadata_path}")
    return records


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
        return_complex=True,
    )

    freq_bins, frames = mixture_stft.shape[-2], mixture_stft.shape[-1]
    mixture_mag = torch.abs(mixture_stft).pow(power)
    mixture_phase = torch.angle(mixture_stft)
    compressed = mixture_mag * torch.exp(1j * mixture_phase)

    two_channel = torch.stack([compressed.real, compressed.imag], dim=-1)
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
    magnitude = torch.abs(compressed).pow(1.0 / power)
    phase = torch.angle(compressed)
    est_stft = magnitude * torch.exp(1j * phase)

    return torch.istft(
        est_stft,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
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


def create_model(cfg: ModelConfig, device: torch.device) -> EaBNet:
    model = EaBNet(
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


def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, object]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Explicitly keep weights_only=False for checkpoints that contain training args/metadata.
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        ckpt_args = checkpoint.get("args", {})
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        ckpt_args = {}
    else:
        raise RuntimeError("Unsupported checkpoint format")

    cfg = model_config_from_checkpoint(ckpt_args)
    model = create_model(cfg, device)

    stripped = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            stripped[key[len("module."):]] = value
        else:
            stripped[key] = value

    model.load_state_dict(stripped, strict=True)
    return model, ckpt_args


def compute_sdr_db(ref: np.ndarray, deg: np.ndarray) -> float:
    # Keep SDR definition consistent with batch_eval_sdr_pesq.py.
    ref_2d = ref[np.newaxis, :]
    deg_2d = deg[np.newaxis, :]
    sdr, _sir, _sar, _perm = bss_eval_sources(ref_2d, deg_2d)
    return float(sdr[0])


def compute_metrics(sample_rate: int, ref: np.ndarray, deg: np.ndarray) -> Tuple[float, float, float]:
    ref = ref.astype(np.float64)
    deg = deg.astype(np.float64)

    mode = "wb" if sample_rate == 16000 else "nb"
    try:
        pesq_score = float(pesq(sample_rate, ref, deg, mode))
    except Exception:
        pesq_score = float("nan")

    try:
        estoi_pct = float(stoi(ref, deg, sample_rate, extended=True) * 100.0)
    except Exception:
        estoi_pct = float("nan")

    sdr_db = compute_sdr_db(ref, deg)
    return pesq_score, estoi_pct, sdr_db


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EaBNet on validation set (PESQ, ESTOI%, SDRdB).")
    parser.add_argument("--val-dir", default="./validation_set", help="Validation set directory containing metadata.csv")
    parser.add_argument("--checkpoint", default="./bestmodels/best_model.pt", help="Path to checkpoint")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--target-ref-mic", type=int, default=0)
    parser.add_argument("--n-fft", type=int, default=512, help="STFT FFT size; EaBNet_light expects 512 (F=257)")
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument("--win-length", type=int, default=320)
    parser.add_argument("--power", type=float, default=0.5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-samples", type=int, default=0, help="Evaluate first N samples only; <=0 means all")
    parser.add_argument("--save-csv", default="", help="Optional path to save per-sample metrics CSV")
    parser.add_argument("--estimate-dir", default="./estimate_set", help="Directory to save mixture/estimate/target and metadata.csv")
    parser.add_argument("--save-samples", choices=["yes", "no"], default="yes", help="Whether to save mixture/estimate/target audio files")
    parser.add_argument("--mixture-ref-mic", type=int, default=0, help="Which mixture channel to save to estimate_set/mixture")
    parser.add_argument("--match-estimate-level", choices=["yes", "no"], default="yes", help="Match estimate amplitude to reference before metric computation")
    parser.add_argument("--max-level-gain-db", type=float, default=20.0, help="Max absolute gain in dB when matching estimate level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = get_device(args.device)
    val_dir = Path(args.val_dir).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    estimate_dir = Path(args.estimate_dir).resolve()
    estimate_metadata_path = estimate_dir / "metadata.csv"

    records = load_records(val_dir)
    if args.max_samples > 0:
        records = records[: args.max_samples]

    model, ckpt_args = load_model(checkpoint_path, device)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    if ckpt_args:
        print("Loaded model config from checkpoint args")
    expected_num_mics = int(getattr(model, "M", 8))
    print(f"Model microphones: {expected_num_mics}")
    print(f"Validation samples: {len(records)}")
    print(f"Estimate dir: {estimate_dir}")

    rows: List[Dict[str, object]] = []
    pesq_values: List[float] = []
    pesq_mix_values: List[float] = []
    estoi_values: List[float] = []
    estoi_mix_values: List[float] = []
    sdr_values: List[float] = []
    sdr_mix_values: List[float] = []

    estimate_dir.mkdir(parents=True, exist_ok=True)
    estimate_metadata_fieldnames = [
        "sample_id",
        "est_ref_energy_ratio",
        "mix_ref_energy_ratio",
        "estimate_level_gain",
        "pesq",
        "estoi_pct",
        "sdr_db",
        "pesq_mix",
        "estoi_mix_pct",
        "sdr_mix_db",
        "mixture_saved_path",
        "estimate_saved_path",
        "target_saved_path",
        "ref_energy",
        "est_energy",
        "mix_energy",
    ]
    with estimate_metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=estimate_metadata_fieldnames)
        writer.writeheader()

    with torch.no_grad():
        for record in tqdm(records, desc="Evaluating"):
            mixture, mix_sr = sf.read(record.mixture_path, dtype="float32", always_2d=True)
            target, tgt_sr = sf.read(record.target_path, dtype="float32", always_2d=False)

            if mixture.shape[1] < expected_num_mics:
                raise ValueError(
                    f"Expected at least {expected_num_mics} mixture channels for sample_id={record.sample_id}, got {mixture.shape[1]}"
                )
            mixture = mixture[:, :expected_num_mics]

            if mix_sr != args.sample_rate or tgt_sr != args.sample_rate:
                raise ValueError(
                    f"Sample rate mismatch at sample_id={record.sample_id}: mix={mix_sr}, tgt={tgt_sr}, expected={args.sample_rate}"
                )

            target_arr = np.asarray(target, dtype=np.float32)
            if target_arr.ndim == 2:
                if target_arr.shape[1] <= args.target_ref_mic:
                    raise ValueError(
                        f"target_ref_mic={args.target_ref_mic} out of range for sample_id={record.sample_id}, target shape={target_arr.shape}"
                    )
                target_arr = target_arr[:, args.target_ref_mic]

            valid_length = min(mixture.shape[0], target_arr.shape[0])
            mixture = mixture[:valid_length, :]
            target_arr = target_arr[:valid_length]

            mix_tensor = torch.from_numpy(mixture).unsqueeze(0)
            mix_stft = build_stft_batch(
                mixture=mix_tensor,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                win_length=args.win_length,
                power=args.power,
                device=device,
            )

            estimate_ri = model(mix_stft)
            estimate_wav = reconstruct_waveform(
                estimate_ri=estimate_ri,
                length=valid_length,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                win_length=args.win_length,
                power=args.power,
                device=device,
            )

            est_np = estimate_wav.squeeze(0).detach().cpu().numpy().astype(np.float64)
            ref_np = target_arr.astype(np.float64)
            mix_np = mixture.astype(np.float64)
            min_len = min(len(ref_np), len(est_np))
            ref_np = ref_np[:min_len]
            est_np = est_np[:min_len]
            mix_np = mix_np[:min_len, :]

            estimate_level_gain = 1.0
            if args.match_estimate_level == "yes":
                est_np, estimate_level_gain = match_estimate_level(ref_np, est_np, args.max_level_gain_db)

            if mix_np.shape[1] <= args.mixture_ref_mic:
                raise ValueError(
                    f"mixture_ref_mic={args.mixture_ref_mic} out of range for sample_id={record.sample_id}, mixture shape={mix_np.shape}"
                )
            mix_save_np = mix_np[:, args.mixture_ref_mic]

            ref_energy = compute_energy(ref_np)
            est_energy = compute_energy(est_np)
            mix_energy = compute_energy(mix_save_np)
            est_ref_energy_ratio = est_energy / (ref_energy + EPS)
            mix_ref_energy_ratio = mix_energy / (ref_energy + EPS)

            print(
                f"sample_id={record.sample_id} gain={estimate_level_gain:.4f}"
            )

            if args.save_samples == "yes":
                mix_out_path = estimate_dir / "mixture" / f"sample_{record.sample_id:08d}_mixture.wav"
                est_out_path = estimate_dir / "estimate" / f"sample_{record.sample_id:08d}_estimate.wav"
                tgt_out_path = estimate_dir / "target" / f"sample_{record.sample_id:08d}_target.wav"
                save_audio(mix_out_path, mix_save_np.astype(np.float32), args.sample_rate)
                save_audio(est_out_path, est_np.astype(np.float32), args.sample_rate)
                save_audio(tgt_out_path, ref_np.astype(np.float32), args.sample_rate)
            else:
                mix_out_path = Path("")
                est_out_path = Path("")
                tgt_out_path = Path("")

            pesq_score, estoi_pct, sdr_db = compute_metrics(args.sample_rate, ref_np, est_np)
            pesq_mix, estoi_mix_pct, sdr_mix_db = compute_metrics(args.sample_rate, ref_np, mix_save_np)

            if not math.isnan(pesq_score):
                pesq_values.append(pesq_score)
            if not math.isnan(pesq_mix):
                pesq_mix_values.append(pesq_mix)
            if not math.isnan(estoi_pct):
                estoi_values.append(estoi_pct)
            if not math.isnan(estoi_mix_pct):
                estoi_mix_values.append(estoi_mix_pct)
            sdr_values.append(sdr_db)
            sdr_mix_values.append(sdr_mix_db)

            rows.append(
                {
                    "sample_id": record.sample_id,
                    "est_ref_energy_ratio": est_ref_energy_ratio,
                    "mix_ref_energy_ratio": mix_ref_energy_ratio,
                    "estimate_level_gain": estimate_level_gain,
                    "pesq": pesq_score,
                    "estoi_pct": estoi_pct,
                    "sdr_db": sdr_db,
                    "pesq_mix": pesq_mix,
                    "estoi_mix_pct": estoi_mix_pct,
                    "sdr_mix_db": sdr_mix_db,
                    "mixture_saved_path": str(mix_out_path),
                    "estimate_saved_path": str(est_out_path),
                    "target_saved_path": str(tgt_out_path),
                    "ref_energy": ref_energy,
                    "est_energy": est_energy,
                    "mix_energy": mix_energy,
                }
            )

            with estimate_metadata_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=estimate_metadata_fieldnames)
                writer.writerow(rows[-1])

            print(
                f"sample_id={record.sample_id} PESQ(est/mix)={pesq_score:.3f}/{pesq_mix:.3f} "
                f"ESTOI(est/mix)={estoi_pct:.2f}/{estoi_mix_pct:.2f} "
                f"SDR(est/mix)={sdr_db:.2f}/{sdr_mix_db:.2f}"
            )

    avg_pesq = float(np.mean(pesq_values)) if pesq_values else float("nan")
    avg_pesq_mix = float(np.mean(pesq_mix_values)) if pesq_mix_values else float("nan")
    avg_estoi = float(np.mean(estoi_values)) if estoi_values else float("nan")
    avg_estoi_mix = float(np.mean(estoi_mix_values)) if estoi_mix_values else float("nan")
    avg_sdr = float(np.mean(sdr_values)) if sdr_values else float("nan")
    avg_sdr_mix = float(np.mean(sdr_mix_values)) if sdr_mix_values else float("nan")

    print("\n=== Validation Average Metrics ===")
    print(f"PESQ: {avg_pesq:.4f}")
    print(f"PESQ(mixture): {avg_pesq_mix:.4f}")
    print(f"ESTOI(%): {avg_estoi:.2f}")
    print(f"ESTOI(mixture,%): {avg_estoi_mix:.2f}")
    print(f"SDR(dB): {avg_sdr:.2f}")
    print(f"SDR(mixture,dB): {avg_sdr_mix:.2f}")
    print(f"Valid PESQ samples: {len(pesq_values)}/{len(rows)}")
    print(f"Valid ESTOI samples: {len(estoi_values)}/{len(rows)}")

    print(f"Estimate metadata saved to: {estimate_metadata_path}")

    if args.save_csv:
        out_csv = Path(args.save_csv).resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            fieldnames = list(rows[0].keys()) if rows else ["sample_id", "pesq", "estoi_pct", "sdr_db"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Per-sample metrics saved to: {out_csv}")


if __name__ == "__main__":
    main()

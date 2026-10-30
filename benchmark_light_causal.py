"""Measure actual ten-millisecond waveform streaming; do not infer real time from GPU utilization."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch

from light_causal_spatial_model import CausalSpatialEnhancer, CAUSAL_CANDIDATE
from light_causal_stream import CausalWaveformStream
from light_quality_loss import QualityObjective
from train_light import build_stft_batch, resolve_record_path


def run(args):
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", scope="Causality, bounded state and real CUDA waveform streaming; no quality claim")
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device("cuda:0")
        torch.manual_seed(1234)
        if args.checkpoint:
            from evaluate_light import load_model
            model, _ = load_model(args.checkpoint, device, CAUSAL_CANDIDATE)
            report["checkpoint_sha256"] = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
        else:
            model = CausalSpatialEnhancer().to(device).eval()
            with torch.no_grad():
                model.decoder.weight.normal_(std=0.005)
                model.decoder.bias.normal_(std=0.005)
        dataset = Path(args.train_dir).expanduser().resolve()
        with (dataset / "metadata.csv").open(encoding="utf-8", newline="") as handle:
            row = next(csv.DictReader(handle))
        path = resolve_record_path(dataset, row["mixture_path"])
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        assert sr == 16000 and audio.shape[1] >= 8 and audio.shape[0] >= 48000
        audio = audio[:96000, :8]
        mixture = torch.from_numpy(audio.copy()).unsqueeze(0)
        report.update(device=torch.cuda.get_device_name(0), torch_version=torch.__version__,
            mixture_path=str(path), sample_id=row["sample_id"], evaluated_samples=audio.shape[0],
            parameters=sum(p.numel() for p in model.parameters()), future_frames=0,
            buffering_delay_samples=320, buffering_delay_ms=20,
            latency_scope="CPU float32 input transfer + streaming frontend/model/synthesis + CPU output transfer; excludes audio driver")
        with torch.inference_mode():
            features, _ = build_stft_batch(mixture.to(device), mixture[..., 0].to(device),
                n_fft=512, hop_length=160, win_length=320, power=0.5, device=device)
            perturbed = features[:, :25].clone()
            perturbed[:, 12:] = torch.randn_like(perturbed[:, 12:]) * 2
            prefix = model(features[:, :25])[:, :, :12]
            changed = model(perturbed)[:, :, :12]
            future_error = (prefix - changed).abs().max().item()
            assert future_error < 1e-6, future_error
            full = QualityObjective().reconstruct(model(features), audio.shape[0]).cpu()
            stream = CausalWaveformStream(model)
            stream.warmup()
            outputs, durations = [], []
            torch.cuda.reset_peak_memory_stats(0)
            for offset in range(0, audio.shape[0], 160):
                torch.cuda.synchronize()
                start = time.perf_counter()
                value = stream.push(mixture[:, offset:offset + 160].to(device)).cpu()
                torch.cuda.synchronize()
                durations.append((time.perf_counter() - start) * 1000)
                outputs.append(value)
            outputs.append(stream.flush().cpu())
            actual = torch.cat(outputs, dim=1)
            difference = (actual - full).abs()
            assert torch.allclose(actual, full, rtol=1e-3, atol=1e-5), difference.max().item()
            steady = np.array(durations[model.context:])
            assert len(steady) >= 100
            def tensor_bytes(value):
                if isinstance(value, torch.Tensor):return value.numel() * value.element_size()
                if isinstance(value, dict):return sum(tensor_bytes(v) for v in value.values())
                if isinstance(value, (list, tuple)):return sum(tensor_bytes(v) for v in value)
                return 0
            report.update(status="measured", future_perturbation_max_error=future_error,
                full_vs_stream_waveform_max_error=difference.max().item(),
                bounded_model_cache_bytes=tensor_bytes(stream.state),
                measured_hops=len(durations), excluded_startup_hops=model.context,
                warmup_hops=model.context + 1, warmup_state_discarded=True,
                all_hop_ms=dict(mean=float(np.mean(durations)), p99=float(np.percentile(durations, 99)), max=float(max(durations))),
                all_hops_below_10ms=bool(max(durations) < 10),
                steady_hop_ms=dict(mean=float(steady.mean()), p50=float(np.percentile(steady, 50)),
                    p95=float(np.percentile(steady, 95)), p99=float(np.percentile(steady, 99)), max=float(steady.max())),
                steady_rtf=float(steady.mean() / 10), deadline_miss_fraction=float((steady > 10).mean()),
                mean_rtf_below_one=bool(steady.mean() < 10),
                p99_below_10ms=bool(np.percentile(steady, 99) < 10),
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(0), hop_ms=durations)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output / "streaming_benchmark.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k != "hop_ms"}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--train-dir", default="/data/ssd1/jinrui.yang/training_set")
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())

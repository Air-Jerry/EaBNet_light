"""Fit a fixed training subset through the existing candidate trainer and evaluator.

This is a training-subset fitting diagnostic, not a generalization benchmark,
a guarantee of PESQ > 3.4, or proof of an exact paper reproduction.
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
import gc
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import soundfile as sf
import torch

import evaluate_light as evaluator
import train_light as trainer
from light_candidate_runtime import read_candidate_checkpoint


ROOT = Path(__file__).resolve().parent
SCOPE = "Training-subset fitting only; not generalization, a PESQ > 3.4 guarantee, or proof of paper equivalence."


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_log_tail(path, max_lines=100, max_chars=12000):
    """Keep the child's actual error available without embedding its entire log."""
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        return "".join(deque(handle, maxlen=max_lines))[-max_chars:]


def is_within(path, directory):
    return path == directory or directory in path.parents


def validate_output(args, records=()):
    output = Path(args.output_dir).expanduser().resolve()
    roots = [Path(args.train_dir).expanduser().resolve(), Path(args.checkpoint).expanduser().resolve().parent]
    roots.extend(path.parent for record in records for path in (record.mixture_path, record.target_path))
    for root in roots:
        if is_within(output, root) or is_within(root, output):
            raise ValueError(f"Output must be separate from all input/checkpoint directories: {output}, {root}")
    return output


def prepare_dataset(args, saved, source=None):
    """Write reproducible FLOAT crops using the trainer's source path semantics."""
    if source is None:
        source = trainer.EnhancementDataset(Path(args.train_dir), sample_rate=saved["sample_rate"],
                                            target_ref_mic=saved["target_ref_mic"], num_mics=saved["num_mics"])
    output = validate_output(args, source.records)
    output.mkdir(parents=True, exist_ok=True)
    dataset = output / "dataset"
    dataset.mkdir()
    length = int(round(args.segment_seconds * saved["sample_rate"]))
    if length <= saved["n_fft"] // 2:
        raise ValueError("Fixed segments are too short for the saved centered STFT")
    records = list(source.records)
    rng = random.Random(args.seed)
    rng.shuffle(records)
    manifest = {"scope": SCOPE, "dataset_dir": str(dataset), "seed": args.seed,
                "segment_samples": length, "waveform_normalization": "none", "wav_subtype": "FLOAT",
                "silence_mean_square_threshold": evaluator.SILENCE_MEAN_SQUARE,
                "target_channels": "all original channels; trainer/evaluator select saved target_ref_mic",
                "metadata_path": str(source.dataset_dir / "metadata.csv"),
                "metadata_sha256": evaluator.sha256_file(source.dataset_dir / "metadata.csv"),
                "checkpoint_path": str(Path(args.checkpoint).resolve()),
                "checkpoint_sha256": evaluator.sha256_file(args.checkpoint), "selected": [], "skipped": []}
    seen = set()
    rows = []
    try:
        for record in records:
            entry = {"source_sample_id": record.sample_id, "mixture_path": str(record.mixture_path),
                     "target_path": str(record.target_path)}
            pair = (record.mixture_path, record.target_path)
            if pair in seen:
                manifest["skipped"].append({**entry, "reason": "duplicate_source_pair"})
                continue
            seen.add(pair)
            mixture, mix_sr = sf.read(record.mixture_path, dtype="float32", always_2d=True)
            target, target_sr = sf.read(record.target_path, dtype="float32", always_2d=True)
            if mix_sr != saved["sample_rate"] or target_sr != saved["sample_rate"]:
                raise ValueError(f"sample_id={record.sample_id}: sample rate mismatch")
            if mixture.shape[1] < saved["num_mics"]:
                raise ValueError(f"sample_id={record.sample_id}: insufficient mixture channels")
            if target.shape[1] > 1 and saved["target_ref_mic"] >= target.shape[1]:
                raise ValueError(f"sample_id={record.sample_id}: target_ref_mic out of range")
            if not np.isfinite(mixture).all() or not np.isfinite(target).all():
                raise ValueError(f"sample_id={record.sample_id}: non-finite audio")
            common_length = min(len(mixture), len(target))
            if common_length < length:
                manifest["skipped"].append({**entry, "reason": "shorter_than_fixed_segment", "common_samples": common_length})
                continue
            offset = rng.randint(0, common_length - length)
            mixture = mixture[offset:offset + length, :saved["num_mics"]]
            target = target[offset:offset + length]
            reference = target[:, saved["target_ref_mic"] if target.shape[1] > 1 else 0]
            if any(np.var(signal.astype(np.float64)) <= evaluator.SILENCE_MEAN_SQUARE
                   for signal in (reference, mixture[:, 0])):
                manifest["skipped"].append({**entry, "reason": "silent_fixed_segment", "offset_samples": offset})
                continue
            sample_id = len(rows) + 1
            mix_path, target_path = dataset / f"mixture_{sample_id}.wav", dataset / f"target_{sample_id}.wav"
            sf.write(mix_path, mixture, saved["sample_rate"], subtype="FLOAT")
            sf.write(target_path, target, saved["sample_rate"], subtype="FLOAT")
            rows.append({"sample_id": sample_id, "mixture_path": mix_path.name, "target_path": target_path.name})
            manifest["selected"].append({**entry, "sample_id": sample_id, "offset_samples": offset,
                                         "num_samples": length, "mixture_sha256": evaluator.sha256_file(record.mixture_path),
                                         "target_sha256": evaluator.sha256_file(record.target_path),
                                         "exported_mixture_path": str(mix_path), "exported_target_path": str(target_path),
                                         "exported_mixture_sha256": evaluator.sha256_file(mix_path),
                                         "exported_target_sha256": evaluator.sha256_file(target_path)})
            if len(rows) == args.num_samples:
                break
        if len(rows) != args.num_samples:
            raise ValueError(f"Only {len(rows)} eligible distinct source pairs; requested {args.num_samples}")
        with (dataset / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("sample_id", "mixture_path", "target_path"))
            writer.writeheader()
            writer.writerows(rows)
        manifest["exported_metadata_sha256"] = evaluator.sha256_file(dataset / "metadata.csv")
        return manifest
    finally:
        write_json(output / "subset_manifest.json", manifest)


def make_training_args(saved, args, dataset_dir, output_dir):
    """Keep architecture/frontend and original optimizer settings except the stated controls."""
    dataset_dir, output_dir = Path(dataset_dir), Path(output_dir)
    settings = vars(trainer.parse_args([])).copy()
    settings.update(saved)
    settings.update(train_dir=str(dataset_dir), val_dir=str(dataset_dir),
                    checkpoint_dir=str(output_dir / "checkpoints"), best_dir=str(output_dir / "best"),
                    log_dir=str(output_dir / "logs"), num_epochs=args.epochs, batch_size=args.batch_size,
                    learning_rate=args.learning_rate, seed=args.seed, segment_seconds=0,
                    train_random_crop="no", num_workers=0, grad_accum_steps=1,
                    use_amp="no", model_amp="no", allow_tf32="no", cudnn_benchmark="no",
                    parallel_mode="none", gpu_ids="0", strict_memory="yes", pin_memory="no",
                    persistent_workers="no", cpu_threads=args.cpu_threads, cpu_thread_scale=1.0,
                    interop_threads=1, resume="yes", resume_reset_lr="no", save_every=args.epochs,
                    train_lr_reduce_on_plateau="no", stop_on_non_finite="yes", skip_non_finite_batches="no",
                    mem_log_every_batches=0, malloc_trim_every_batches=0, gc_every_batches=0)
    return argparse.Namespace(**settings)


def write_initial_checkpoint(source_checkpoint, model, training_args, path):
    """Warm-start weights/buffers with fresh Adam, without touching the source checkpoint."""
    optimizer = torch.optim.Adam(model.parameters(), lr=training_args.learning_rate,
                                 weight_decay=training_args.weight_decay)
    state = {"epoch": 0, "best_val_loss": float("inf"), "model_state_dict": model.state_dict(),
             "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": None,
             "lr_reducer_state": {}, "args": vars(training_args).copy(),
             "diagnostic_source_epoch": int(source_checkpoint["epoch"]), "diagnostic_scope": SCOPE}
    torch.save(state, path)


def release_models():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_checkpoint(checkpoint, dataset_dir, output_dir, args):
    """Use the original validation loss and unified no-gain evaluator in FP32."""
    device = evaluator.get_device(args.device)
    model, saved = evaluator.load_model(Path(checkpoint), device)
    training_args = make_training_args(saved, args, dataset_dir, output_dir)
    dataset = trainer.EnhancementDataset(Path(dataset_dir), sample_rate=saved["sample_rate"],
                                        target_ref_mic=saved["target_ref_mic"], num_mics=saved["num_mics"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=0, collate_fn=trainer.collate_batch)
    try:
        loss = trainer.run_epoch(model, loader, None, None, device, [0] if device.type == "cuda" else [],
                                 training_args, training=False)
        if not math.isfinite(loss):
            raise ValueError("Non-finite diagnostic validation loss")
    finally:
        del model, loader
        release_models()
    eval_args = evaluator.parse_args(["--checkpoint", str(checkpoint), "--val-dir", str(dataset_dir),
                                      "--device", args.device, "--estimate-dir", str(output_dir),
                                      "--save-samples", "no", "--match-estimate-level", "no"])
    metrics = evaluator.evaluate(eval_args)
    release_models()
    checkpoint_data, _ = read_candidate_checkpoint(checkpoint)
    return {"checkpoint": str(checkpoint), "epoch": checkpoint_data["epoch"], "eval_loss": loss,
            "metrics": metrics["mean"], "summary_json": str(output_dir / "summary.json")}


def training_command(training_args):
    command = [sys.executable, "-u", str(ROOT / "train_light_candidate.py"),
               "--candidate", training_args.structure_candidate]
    for name in vars(trainer.parse_args([])):
        command.extend(["--" + name.replace("_", "-"), str(getattr(training_args, name))])
    return command


def training_environment(device, cpu_threads):
    environment = os.environ.copy()
    for name in list(environment):
        if name in {"RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK",
                    "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"} or name.startswith("TORCHELASTIC_"):
            environment.pop(name, None)
    environment.update(OMP_NUM_THREADS=str(cpu_threads), MKL_NUM_THREADS=str(cpu_threads), PYTHONIOENCODING="utf-8")
    if device.type == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    return environment


def run(args):
    output = validate_output(args)
    if output.exists():
        raise FileExistsError(f"Use a new --output-dir; refusing to overwrite {output}")
    # This constructor only resolves metadata; sample_rate is unused until audio is indexed.
    source = trainer.EnhancementDataset(Path(args.train_dir).expanduser(), sample_rate=16000)
    validate_output(args, source.records)
    output.mkdir(parents=True, exist_ok=False)
    summary = {"status": "running", "scope": SCOPE, "output_dir": str(output),
               "source_checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
               "training_log": str(output / "training.log"),
               "before": None, "after_latest": None, "after_best": None}
    try:
        if torch.distributed.is_initialized():
            raise RuntimeError("Run this diagnostic as a single process, outside an initialized process group")
        torch.set_num_threads(args.cpu_threads)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("highest")
        device = evaluator.get_device(args.device)
        args.device = device.type
        args.output_dir = str(output)
        args.checkpoint = str(Path(args.checkpoint).expanduser().resolve())
        args.train_dir = str(Path(args.train_dir).expanduser().resolve())
        checkpoint, candidate_id = read_candidate_checkpoint(args.checkpoint)
        model, saved = evaluator.load_model(Path(args.checkpoint), torch.device("cpu"), candidate_id)
        evaluator.resolve_frontend(argparse.Namespace(), saved)
        if not all(torch.isfinite(value).all() for value in model.state_dict().values() if value.is_floating_point()):
            raise ValueError("Source model contains non-finite weights or buffers")
        summary.update(source_epoch=int(checkpoint["epoch"]), source_checkpoint_sha256=evaluator.sha256_file(args.checkpoint),
                       candidate=candidate_id, device=args.device, num_samples=args.num_samples, epochs=args.epochs,
                       batch_size=args.batch_size, learning_rate=args.learning_rate, segment_seconds=args.segment_seconds,
                       frontend=evaluator.resolve_frontend(argparse.Namespace(), saved),
                       subset_manifest=str(output / "subset_manifest.json"),
                       training_config=str(output / "training_command.json"),
                       steps=args.epochs * (args.num_samples // args.batch_size))
        manifest = prepare_dataset(args, saved, source=source)
        dataset = Path(manifest["dataset_dir"])
        training_args = make_training_args(saved, args, dataset, output)
        initial = output / "initial_checkpoint.pt"
        write_initial_checkpoint(checkpoint, model, training_args, initial)
        latest = Path(training_args.checkpoint_dir) / "checkpoint_latest.pt"
        latest.parent.mkdir()
        latest.write_bytes(initial.read_bytes())
        command = training_command(training_args)
        write_json(output / "training_command.json", {"argv": command, "cwd": str(ROOT), "args": vars(training_args)})
        del checkpoint, model
        release_models()
        summary["before"] = evaluate_checkpoint(initial, dataset, output / "before", args)
        write_json(output / "diagnostic_summary.json", summary)
        print(f"Training fixed subset: {args.epochs} epochs, {summary['steps']} optimizer steps; log: {output / 'training.log'}", flush=True)
        with (output / "training.log").open("w", encoding="utf-8") as log:
            with subprocess.Popen(command, cwd=ROOT, env=training_environment(device, args.cpu_threads),
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace") as process:
                try:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
                returncode = process.wait()
                summary["training_returncode"] = returncode
                if returncode:
                    raise subprocess.CalledProcessError(returncode, command)
        final_checkpoint, _ = read_candidate_checkpoint(latest, candidate_id, training_args)
        steps = {int(state["step"].item() if torch.is_tensor(state["step"]) else state["step"])
                 for state in final_checkpoint["optimizer_state_dict"]["state"].values() if "step" in state}
        if final_checkpoint["epoch"] != args.epochs or steps != {summary["steps"]}:
            raise RuntimeError(f"Incomplete training: epoch={final_checkpoint['epoch']}, optimizer steps={steps}")
        summary["actual_optimizer_steps"] = next(iter(steps))
        del final_checkpoint
        summary["after_latest"] = evaluate_checkpoint(latest, dataset, output / "after_latest", args)
        summary["after_best"] = evaluate_checkpoint(Path(training_args.best_dir) / "best_model.pt", dataset,
                                                     output / "after_best", args)
        if evaluator.sha256_file(args.checkpoint) != summary["source_checkpoint_sha256"]:
            raise RuntimeError("Original checkpoint changed during diagnostic")
        summary["status"] = "complete"
        return summary
    except BaseException as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        try:
            if (output / "training.log").is_file():
                summary["training_log_tail"] = read_log_tail(output / "training.log")
        except OSError as log_error:
            summary["training_log_read_error"] = f"{type(log_error).__name__}: {log_error}"
        raise
    finally:
        write_json(output / "diagnostic_summary.json", summary)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--output-dir", required=True, help="New directory outside input and original checkpoint directories")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--segment-seconds", type=float, default=6.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args(argv)
    if min(args.num_samples, args.epochs, args.batch_size, args.cpu_threads) < 1:
        parser.error("num-samples, epochs, batch-size and cpu-threads must be positive")
    if args.num_samples % args.batch_size:
        parser.error("num-samples must be divisible by batch-size")
    if not math.isfinite(args.segment_seconds) or args.segment_seconds <= 0:
        parser.error("segment-seconds must be finite and positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive")
    return args


def main(argv=None):
    result = run(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

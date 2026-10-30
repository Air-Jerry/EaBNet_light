"""Verify real-data dual-CUDA updates before full causal-model training."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from train_light import resolve_record_path


ROOT = Path(__file__).resolve().parent


def run(args):
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", scope=f"{args.num_samples} fixed training records; CUDA training correctness, not generalization")
    sampler = None
    try:
        assert torch.cuda.device_count() == 2 and torch.cuda.is_bf16_supported()
        assert all(torch.cuda.mem_get_info(i)[0] > 10 * 2**30 for i in range(2)), "Both GPUs must be free"
        data = Path(args.train_dir).expanduser().resolve()
        with (data / "metadata.csv").open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, []
            for _, row in zip(range(args.num_samples), reader):
                for key in ("mixture_path", "target_path"):
                    row[key] = str(resolve_record_path(data, row[key]))
                rows.append(row)
        assert args.num_samples >= 32 and args.num_samples % 8 == 0
        assert len(rows) == args.num_samples
        subset = output / "dataset"
        subset.mkdir()
        with (subset / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        checkpoints = output / "checkpoints"
        command = [sys.executable, "-u", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
            args.entry, "--train-dir", str(subset), "--val-dir", str(subset),
            "--checkpoint-dir", str(checkpoints), "--best-dir", str(output / "best"),
            "--log-dir", str(output / "tensorboard"), "--num-epochs", "2", "--parallel-mode", "ddp",
            "--batch-size", str(args.batch_size), "--grad-accum-steps", str(4 // args.batch_size), "--segment-seconds", "6",
            "--train-random-crop", "no", "--num-workers", "4", "--strict-memory", "no",
            "--cpu-threads", "2", "--cpu-thread-scale", "1", "--use-amp", "no", "--model-amp", "no",
            "--allow-tf32", "no", "--stop-on-non-finite", "yes", "--skip-non-finite-batches", "no", "--save-every", "1"]
        if args.warmstart:
            if args.entry!='train_light_causal_global.py':raise ValueError('Only the global warm-start entry accepts --warmstart')
            command.extend(['--warmstart',str(Path(args.warmstart).expanduser().resolve())])
        environment = os.environ.copy()
        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            environment.pop(key, None)
        report.update(command=command, sample_ids=[row["sample_id"] for row in rows],
                      batch_size_per_gpu=args.batch_size, grad_accum_steps=4 // args.batch_size,
                      effective_global_batch=8,
                      torch_version=torch.__version__, devices=[torch.cuda.get_device_name(i) for i in range(2)])
        (output / "cuda_check.json").write_text(json.dumps(report, indent=2))
        start = time.perf_counter()
        with (output / "gpu_utilization.log").open("w") as gpu, (output / "training.log").open("w") as log:
            sampler = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits", "-l", "5"], stdout=gpu, stderr=subprocess.STDOUT)
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        report.update(training_exit_code=result.returncode, elapsed_seconds=time.perf_counter() - start)
        assert result.returncode == 0, f"Training failed; inspect {output / 'training.log'}"
        zero = torch.load(checkpoints / "model_epoch_0.pt", map_location="cpu", weights_only=False)
        state = torch.load(checkpoints / "model_epoch_2.pt", map_location="cpu", weights_only=False)
        weights = state["model_state_dict"]
        assert state["epoch"] == 2 and not any(name.startswith("base.") for name in weights)
        assert all(torch.isfinite(value).all() for value in weights.values())
        optimizer = state["optimizer_state_dict"]["state"]
        expected_steps = args.num_samples // 4
        assert optimizer and {int(value["step"]) for value in optimizer.values()} == {expected_steps}
        assert all(torch.isfinite(value).all() for item in optimizer.values()
                   for value in item.values() if isinstance(value, torch.Tensor))
        changed = sum(not torch.equal(value, zero["model_state_dict"][name]) for name, value in weights.items())
        assert changed > len(weights) * 0.7
        report.update(status="passed", epochs=2, generator_adam_step=expected_steps,
            all_weights_and_adam_finite=True, noncausal_frozen_backbones=0,
            state_tensors=len(weights), changed_state_tensors=changed,
            finite_gradients="Checked on both ranks before every optimizer step",
            checkpoint_sha256=hashlib.sha256((checkpoints / "model_epoch_2.pt").read_bytes()).hexdigest())
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if sampler is not None:
            sampler.terminate()
            sampler.wait(timeout=15)
            samples = {0: [], 1: []}
            with (output / "gpu_utilization.log").open() as handle:
                for row in csv.reader(handle):
                    if len(row) == 5:
                        samples[int(row[1])].append(tuple(float(value) for value in row[2:]))
            report["gpu_observations"] = [{"index": i, "samples": len(v),
                "utilization_mean_pct": sum(x[0] for x in v) / len(v),
                "sampled_peak_memory_mib": max(x[1] for x in v)} for i, v in samples.items() if v]
        (output / "cuda_check.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", default="/data/ssd1/jinrui.yang/training_set")
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--entry", choices=("train_light_causal.py", "train_light_causal_fast.py", "train_light_causal_global.py", "train_light_causal_recurrent.py", "train_light_causal_recurrent65.py"), default="train_light_causal.py")
    parser.add_argument("--warmstart")
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())

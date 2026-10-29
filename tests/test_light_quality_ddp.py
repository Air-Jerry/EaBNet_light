"""Run one actual two-rank CPU/Gloo update through the quality training CLI."""

import csv
import hashlib
import math
import os
from pathlib import Path
import socket
import subprocess
import sys

import numpy as np
import soundfile as sf
import torch

import evaluate_light
from light_candidate_runtime import candidate_metadata, read_candidate_checkpoint
from light_quality_model import BASE_CANDIDATE, QUALITY_CANDIDATE
import train_light


ROOT = Path(__file__).resolve().parents[1]


def run_two_windows_gloo_ranks(tmp_path, training_arguments, environment):
    """Bypass the Windows elastic wrapper's hardcoded libuv TCPStore default."""
    wrapper = tmp_path / "quality_gloo_rank.py"
    wrapper.write_text(
        "from datetime import timedelta\n"
        "import os\n"
        "import sys\n"
        "import torch.distributed as dist\n"
        "sys.path.insert(0, os.environ['QUALITY_TEST_ROOT'])\n"
        "dist.init_process_group(backend='gloo', rank=int(os.environ['RANK']), "
        "world_size=2, timeout=timedelta(seconds=30), "
        "init_method='tcp://127.0.0.1:' + os.environ['MASTER_PORT'] + '?use_libuv=0')\n"
        "import train_light_quality\n"
        "raise SystemExit(train_light_quality.main())\n",
        encoding="utf-8",
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    processes = []
    results = []
    try:
        for rank in range(2):
            rank_environment = environment.copy()
            rank_environment.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2",
                                    MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                                    QUALITY_TEST_ROOT=str(ROOT))
            rank_environment.pop("TORCHELASTIC_USE_AGENT_STORE", None)
            processes.append(subprocess.Popen(
                [sys.executable, str(wrapper), *training_arguments], cwd=ROOT,
                env=rank_environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
            ))
        for process in processes:
            stdout, stderr = process.communicate(timeout=90)
            results.append((process.returncode, stdout, stderr))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=10)
    return subprocess.CompletedProcess(
        args=[str(wrapper)], returncode=next((code for code, _, _ in results if code), 0),
        stdout="\n".join(stdout for _, stdout, _ in results),
        stderr="\n".join(stderr for _, _, stderr in results),
    )


def test_real_two_rank_gloo_quality_update(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    dataset = tmp_path / "training_set"
    (dataset / "mixture").mkdir(parents=True)
    (dataset / "target").mkdir()
    rows = []
    time = np.arange(3200, dtype=np.float64) / 16000
    for index in range(2):
        speech = (0.1 * np.sin(2 * np.pi * (140 + index * 31) * time)).astype(np.float32)
        noise = np.random.default_rng(7 + index).normal(0, 0.01, (len(time), 8))
        mixture = (speech[:, None] + noise).astype(np.float32)
        mixture_name = f"sample_{index:08d}_8ch.wav"
        target_name = f"sample_{index:08d}_target.wav"
        sf.write(dataset / "mixture" / mixture_name, mixture, 16000, subtype="FLOAT")
        sf.write(dataset / "target" / target_name, speech, 16000, subtype="FLOAT")
        rows.append([index, f"old_export/training_set/mixture/{mixture_name}",
                     f"old_export/training_set/target/{target_name}"])
    with (dataset / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "mixture_path", "target_path"])
        writer.writerows(rows)

    args = train_light.parse_args(["--train-dir", str(dataset), "--val-dir", str(dataset)])
    for name, value in candidate_metadata(BASE_CANDIDATE).items():
        setattr(args, name, value)
    base = evaluate_light.create_model(evaluate_light.model_config_from_checkpoint(vars(args)),
                                       torch.device("cpu"), BASE_CANDIDATE)
    for module in base.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean.fill_(0.03)
            module.running_var.fill_(0.95)
            module.num_batches_tracked.fill_(40)
    source_dir = tmp_path / "original_checkpoints"
    source_dir.mkdir()
    source = source_dir / "model_epoch_40.pt"
    torch.save({"epoch": 40, "args": vars(args), "model_state_dict": base.state_dict(),
                "best_val_loss": 0.02}, source)
    original_source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       PYTHONIOENCODING="utf-8")
    if os.name == "nt":
        # Windows CPU wheels can omit libuv; retain the real TCP/Gloo transport.
        environment["USE_LIBUV"] = "0"
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(name, None)
    checkpoint_dir = tmp_path / "quality_checkpoints"
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
        "--nproc_per_node=2", "--rdzv_conf=join_timeout=30,read_timeout=30",
        str(ROOT / "train_light_quality.py"), "--warmstart", str(source),
        "--train-dir", str(dataset), "--val-dir", str(dataset),
        "--checkpoint-dir", str(checkpoint_dir), "--best-dir", str(tmp_path / "quality_best"),
        "--log-dir", str(tmp_path / "quality_logs"), "--num-epochs", "1",
        "--num-workers", "0", "--parallel-mode", "ddp", "--cpu-threads", "2",
        "--interop-threads", "1", "--use-amp", "no", "--model-amp", "no",
        "--batch-size", "1", "--grad-accum-steps", "1", "--segment-seconds", "0.2",
        "--train-random-crop", "no", "--mem-log-every-batches", "0",
        "--gc-every-batches", "0", "--malloc-trim-every-batches", "0",
        "--log-perf", "no", "--stop-on-non-finite", "yes", "--save-every", "1",
    ]
    completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True,
                               text=True, encoding="utf-8", errors="replace", timeout=180)
    output = completed.stdout + "\n" + completed.stderr
    if (os.name == "nt" and completed.returncode != 0
            and "PyTorch was built without libuv support" in output):
        (tmp_path / "torchrun_libuv_failure.log").write_text(output, encoding="utf-8")
        entry_index = command.index(str(ROOT / "train_light_quality.py"))
        completed = run_two_windows_gloo_ranks(tmp_path, command[entry_index + 1:], environment)
        output = completed.stdout + "\n" + completed.stderr
    (tmp_path / "quality_ddp.log").write_text(output, encoding="utf-8")
    assert completed.returncode == 0, output
    assert "DDP world_size/rank/local_rank: 2/0/0" in output
    assert "Enabled DistributedDataParallel" in output
    assert "[QUALITY][train][rank0]" in output

    initial, _ = read_candidate_checkpoint(checkpoint_dir / "model_epoch_0.pt", QUALITY_CANDIDATE)
    trained, _ = read_candidate_checkpoint(checkpoint_dir / "checkpoint_latest.pt", QUALITY_CANDIDATE)
    assert trained["epoch"] == 1
    assert math.isfinite(trained["train_loss"]) and math.isfinite(trained["val_loss"])
    optimizer_states = trained["optimizer_state_dict"]["state"]
    assert optimizer_states
    assert {int(state["step"].item()) for state in optimizer_states.values()} == {1}
    for state in optimizer_states.values():
        assert torch.isfinite(state["exp_avg"]).all()
        assert torch.isfinite(state["exp_avg_sq"]).all()
    for name, value in base.state_dict().items():
        torch.testing.assert_close(trained["model_state_dict"]["base." + name], value, rtol=0, atol=0)
    for value in trained["model_state_dict"].values():
        assert torch.isfinite(value).all()
    assert any(not torch.equal(value, initial["model_state_dict"][name])
               for name, value in trained["model_state_dict"].items() if name.startswith("refiner."))
    restored, saved = evaluate_light.load_model(checkpoint_dir / "checkpoint_latest.pt", torch.device("cpu"),
                                                QUALITY_CANDIDATE)
    assert saved["structure_candidate"] == QUALITY_CANDIDATE
    assert set(restored.state_dict()) == set(trained["model_state_dict"])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_source_hash

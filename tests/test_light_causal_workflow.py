"""Actual causal train/resume/inference and DDP through the unchanged loop."""

import csv
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf
import torch

import evaluate_light
from light_candidate_runtime import read_candidate_checkpoint
from light_causal_spatial_model import CAUSAL_CANDIDATE
from test_light_quality_workflow import run_cli, ROOT
from test_light_quality_ddp import run_two_windows_gloo_ranks
import train_light_causal as causal


@pytest.fixture
def argv(tmp_path):
    data = tmp_path / "dataset"
    data.mkdir()
    speech = (0.1 * np.sin(2 * np.pi * 143 * np.arange(4000) / 16000)).astype(np.float32)
    mixture = speech[:, None] + np.random.default_rng(8).normal(0, 0.01, (4000, 8))
    sf.write(data / "mixture.wav", mixture.astype(np.float32), 16000, subtype="FLOAT")
    sf.write(data / "target.wav", speech, 16000, subtype="FLOAT")
    with (data / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "mixture_path", "target_path"])
        writer.writerows([[1, "mixture.wav", "target.wav"], [2, "mixture.wav", "target.wav"]])
    return ["--train-dir", str(data), "--val-dir", str(data),
        "--checkpoint-dir", str(tmp_path / "checkpoints"), "--best-dir", str(tmp_path / "best"),
        "--log-dir", str(tmp_path / "logs"), "--num-workers", "0", "--parallel-mode", "none",
        "--cpu-threads", "2", "--batch-size", "1", "--grad-accum-steps", "1",
        "--segment-seconds", "0.25", "--train-random-crop", "no", "--log-perf", "no",
        "--mem-log-every-batches", "0", "--gc-every-batches", "0", "--malloc-trim-every-batches", "0"]


def test_causal_real_training_resume_and_saved_inference(argv):
    run_cli("train_light_causal.py", [*argv, "--num-epochs", "1"])
    directory = Path(argv[argv.index("--checkpoint-dir") + 1])
    first, _ = read_candidate_checkpoint(directory / "checkpoint_latest.pt", CAUSAL_CANDIDATE)
    assert first["epoch"] == 1
    assert {int(v["step"]) for v in first["optimizer_state_dict"]["state"].values()} == {2}
    assert not any(name.startswith("base.") for name in first["model_state_dict"])
    assert all(torch.isfinite(v).all() for v in first["model_state_dict"].values())
    assert torch.count_nonzero(first["model_state_dict"]["decoder.weight"]) > 0
    model, _ = evaluate_light.load_model(directory / "checkpoint_latest.pt", torch.device("cpu"), CAUSAL_CANDIDATE)
    assert model.is_causal
    value = torch.randn(1, 4, 257, 8, 2) * 0.01
    with torch.no_grad():
        output = model(value)
    assert output.shape == (1, 2, 4, 257) and torch.isfinite(output).all()
    run_cli("train_light_causal.py", [*argv, "--num-epochs", "2"])
    second, _ = read_candidate_checkpoint(directory / "checkpoint_latest.pt", CAUSAL_CANDIDATE)
    assert second["epoch"] == 2
    assert {int(v["step"]) for v in second["optimizer_state_dict"]["state"].values()} == {4}
    with pytest.raises(ValueError, match="target_ref_mic differs"):
        causal.parse_args([*argv, "--target-ref-mic", "1"])


def test_actual_two_rank_causal_update(argv, tmp_path):
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONIOENCODING="utf-8")
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(name, None)
    arguments = [*argv, "--num-epochs", "1", "--parallel-mode", "ddp"]
    if os.name == "nt":
        result = run_two_windows_gloo_ranks(tmp_path, arguments, environment, entry_module="train_light_causal")
    else:
        result = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
            "--nproc_per_node=2", "train_light_causal.py", *arguments], cwd=ROOT, env=environment,
            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    directory = Path(argv[argv.index("--checkpoint-dir") + 1])
    state, _ = read_candidate_checkpoint(directory / "checkpoint_latest.pt", CAUSAL_CANDIDATE)
    assert {int(v["step"]) for v in state["optimizer_state_dict"]["state"].values()} == {1}


@pytest.mark.parametrize("fault", [None, "partial", "gain", "noncausal", "changed_mics"])
def test_launcher_requires_complete_causal_same_protocol_scores(tmp_path, monkeypatch, fault):
    report = dict(status="complete", completed_samples=600, expected_samples=600,
        frontend={"n_fft": 512}, frontend_protocol={"normalization": "none"}, num_mics=8,
        selected_pairs_sha256="same", metadata_sha256="same", metric_protocol={"target_gain_matching": False},
        structure_candidate=CAUSAL_CANDIDATE, candidate_evidence={"model_causal": True, "future_frames": 0},
        candidate_implementation_sha256={"model": "fixed"}, checkpoint="model.pt", checkpoint_sha256="fixed",
        mean={"enhanced": {"pesq": 1.08}, "noisy": {"pesq": 1.08}})
    legacy = tmp_path / "logs_cbam_metric_v1/metric_20261030_052113_1573812/pesq_selection/model_epoch_0/summary.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps(report))
    for epoch in (0, 1):
        value = copy.deepcopy(report)
        if epoch == 1:
            value["mean"]["enhanced"]["pesq"] = 3.45
            if fault == "partial": value["completed_samples"] = 599
            if fault == "gain": value["metric_protocol"]["target_gain_matching"] = True
            if fault == "noncausal": value["candidate_evidence"]["model_causal"] = False
            if fault == "changed_mics": value["num_mics"] = 7
        path = tmp_path / f"run/pesq_selection/model_epoch_{epoch}/summary.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(value))
    shell = (ROOT / "run_light_causal.sh").read_text()
    code = shell.split("<<'PY_RESULTS'\n", 1)[1].split("\nPY_RESULTS", 1)[0]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["-", str(tmp_path / "run")])
    if fault:
        with pytest.raises(AssertionError):exec(compile(code, "causal full600 gate", "exec"), {})
    else:
        exec(compile(code, "causal full600 gate", "exec"), {})
        ranked = json.loads((tmp_path / "run/pesq_selection/checkpoint_ranking.json").read_text())
        assert ranked["target_reached"] and ranked["model_is_causal"]
        assert ranked["final_full600_waveform_streaming_audit_required"]

"""Source identity, actual updates, strict resume and inference for the new candidate."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

import evaluate_light
import train_light_quality as quality
import train_light_spatial as spatial
from light_candidate_runtime import read_candidate_checkpoint
from light_spatial_model import SPATIAL_CANDIDATE
from test_light_quality_workflow import source_case, arguments, digest, run_cli, ROOT
from test_light_quality_ddp import run_two_windows_gloo_ranks


@pytest.fixture
def spatial_case(source_case, tmp_path):
    args, source = quality.parse_args(arguments(source_case, tmp_path / "quality"))
    quality.prepare_checkpoint(args, source)
    checkpoint = tmp_path / "quality/checkpoints/model_epoch_0.pt"
    argv = arguments(source_case, tmp_path / "spatial")
    argv[1] = str(checkpoint)
    return checkpoint, argv


def test_real_spatial_train_resume_and_inference(spatial_case):
    warmstart, argv = spatial_case
    before = digest(warmstart)
    original = torch.load(warmstart, weights_only=False)
    run_cli("train_light_spatial.py", [*argv, "--num-epochs", "1"])
    directory = Path(argv[argv.index("--checkpoint-dir") + 1])
    first, _ = read_candidate_checkpoint(directory / "checkpoint_latest.pt", SPATIAL_CANDIDATE)
    zero, _ = read_candidate_checkpoint(directory / "model_epoch_0.pt", SPATIAL_CANDIDATE)
    assert {int(value["step"]) for value in first["optimizer_state_dict"]["state"].values()} == {1}
    for name, value in original["model_state_dict"].items():
        torch.testing.assert_close(first["model_state_dict"]["base." + name], value, rtol=0, atol=0)
    assert first["args"]["target_ref_mic"] == original["args"]["target_ref_mic"]
    model, _ = evaluate_light.load_model(directory / "model_epoch_0.pt", torch.device("cpu"), SPATIAL_CANDIDATE)
    source_model, _ = evaluate_light.load_model(warmstart, torch.device("cpu"), "cbam_refiner_v1")
    value = torch.randn(1, 7, 257, 8, 2) * 0.1
    with torch.no_grad():
        torch.testing.assert_close(model(value), source_model(value), rtol=0, atol=0)
    assert not torch.equal(first["model_state_dict"]["refiner.network.decoder.weight"],
                           zero["model_state_dict"]["refiner.network.decoder.weight"])
    run_cli("train_light_spatial.py", [*argv, "--num-epochs", "2"])
    second, _ = read_candidate_checkpoint(directory / "checkpoint_latest.pt", SPATIAL_CANDIDATE)
    assert second["epoch"] == 2
    assert {int(value["step"]) for value in second["optimizer_state_dict"]["state"].values()} == {2}
    assert all(torch.isfinite(value).all() for value in second["model_state_dict"].values())
    assert digest(warmstart) == before


def test_spatial_resume_rejects_frontend_and_training_changes(spatial_case):
    _, argv = spatial_case
    run_cli("train_light_spatial.py", [*argv, "--num-epochs", "0"])
    with pytest.raises(ValueError, match="differs"):
        spatial.parse_args([*argv, "--hop-length", "128"])
    latest = Path(argv[argv.index("--checkpoint-dir") + 1]) / "checkpoint_latest.pt"
    state = torch.load(latest, weights_only=False)
    state["args"]["spatial_training_precision"] = "changed"
    torch.save(state, latest)
    with pytest.raises(ValueError, match="spatial_training_precision"):
        spatial.parse_args(argv)


def test_actual_two_rank_spatial_update(spatial_case, tmp_path):
    _, argv = spatial_case
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONIOENCODING="utf-8")
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(name, None)
    argv = [*argv, "--num-epochs", "1", "--parallel-mode", "ddp"]
    if os.name == "nt":
        result = run_two_windows_gloo_ranks(tmp_path, argv, environment, entry_module="train_light_spatial")
    else:
        result = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
            "--nproc_per_node=2", "train_light_spatial.py", *argv], cwd=ROOT, env=environment,
            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    directory = Path(argv[argv.index("--checkpoint-dir") + 1])
    result, _ = read_candidate_checkpoint(directory / "checkpoint_latest.pt", SPATIAL_CANDIDATE)
    assert result["epoch"] == 1
    assert {int(v["step"]) for v in result["optimizer_state_dict"]["state"].values()} == {1}


@pytest.mark.parametrize("fault", [None, "partial_high_score", "changed_gain", "changed_initial_output"])
def test_spatial_launcher_full600_and_source_gates(tmp_path, monkeypatch, fault):
    report = dict(status="complete", completed_samples=600, expected_samples=600,
        frontend={"n_fft": 512}, frontend_protocol={"waveform_normalization": "none"},
        num_mics=8, selected_pairs_sha256="same", metadata_sha256="same",
        metric_protocol={"target_gain_matching": False}, checkpoint="source.pt", checkpoint_sha256="source",
        mean={"enhanced": {"pesq": 2.865}, "noisy": {"pesq": 1.088}})
    source = tmp_path / "source.json"
    source.write_text(json.dumps(report))
    for epoch in (0, 1):
        candidate = copy.deepcopy(report)
        if epoch == 1:
            candidate["mean"]["enhanced"]["pesq"] = 3.45
        if fault == "partial_high_score" and epoch == 1:
            candidate["completed_samples"] = 599
        if fault == "changed_gain":
            candidate["metric_protocol"]["target_gain_matching"] = True
        if fault == "changed_initial_output" and epoch == 0:
            candidate["mean"]["enhanced"]["pesq"] += 0.1
        path = tmp_path / f"pesq_selection/model_epoch_{epoch}/summary.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(candidate))
    shell = (ROOT / "run_light_spatial.sh").read_text()
    code = shell.split("<<'PY_RESULTS'\n", 1)[1].split("\nPY_RESULTS", 1)[0]
    monkeypatch.setattr(sys, "argv", ["-", str(tmp_path), str(source)])
    if fault:
        with pytest.raises(AssertionError):
            exec(compile(code, "spatial result gate", "exec"), {})
    else:
        exec(compile(code, "spatial result gate", "exec"), {})
        ranked = json.loads((tmp_path / "pesq_selection/checkpoint_ranking.json").read_text())
        assert ranked["target_reached"] and ranked["best_completed_epoch"] == 1

"""Fresh G/D initialization, real updates, full-state resume and two-rank sync."""

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

import evaluate_light
import train_light_metric as metric
import train_light_quality as quality
from light_candidate_runtime import read_candidate_checkpoint
from light_quality_model import QUALITY_CANDIDATE
from test_light_quality_workflow import source_case, arguments, digest, run_cli
from test_light_quality_ddp import run_two_windows_gloo_ranks


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def metric_case(source_case, tmp_path):
    dataset = tmp_path / "dataset"
    shutil.copytree(source_case["dataset"], dataset)
    with (dataset / "metadata.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    rows.append([18, *rows[1][1:]])
    with (dataset / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    case = dict(source_case, dataset=dataset)
    initial_args = arguments(case, tmp_path / "quality")
    args, source = quality.parse_args(initial_args)
    quality.prepare_checkpoint(args, source)
    warmstart = tmp_path / "quality/checkpoints/model_epoch_0.pt"
    argv = arguments(case, tmp_path / "metric")
    argv[1] = str(warmstart)
    argv.extend(("--segment-seconds", "0", "--grad-accum-steps", "2"))
    return warmstart, argv


def test_metric_real_train_resume_and_inference(metric_case):
    warmstart, argv = metric_case
    source, _ = read_candidate_checkpoint(warmstart, QUALITY_CANDIDATE)
    source_hash = digest(warmstart)
    run_cli("train_light_metric.py", [*argv, "--num-epochs", "1"])
    directory = Path(argv[argv.index("--checkpoint-dir") + 1])
    initialized, _ = read_candidate_checkpoint(directory / "model_epoch_0.pt", QUALITY_CANDIDATE)
    first, _ = read_candidate_checkpoint(directory / "checkpoint_latest.pt", QUALITY_CANDIDATE)
    for name, value in source["model_state_dict"].items():
        torch.testing.assert_close(initialized["model_state_dict"][name], value, rtol=0, atol=0)
    assert not initialized["optimizer_state_dict"]["state"]
    assert not initialized["metric_state"]["optimizer"]["state"]
    assert first["metric_state"]["updates"] == 2
    assert {int(v["step"]) for v in first["optimizer_state_dict"]["state"].values()} == {1}
    assert {int(v["step"]) for v in first["metric_state"]["optimizer"]["state"].values()} == {2}
    run_cli("train_light_metric.py", [*argv, "--num-epochs", "2"])
    second, _ = read_candidate_checkpoint(directory / "checkpoint_latest.pt", QUALITY_CANDIDATE)
    assert second["epoch"] == 2 and second["metric_state"]["updates"] == 4
    assert {int(v["step"]) for v in second["optimizer_state_dict"]["state"].values()} == {2}
    assert {int(v["step"]) for v in second["metric_state"]["optimizer"]["state"].values()} == {4}
    for name, value in source["model_state_dict"].items():
        if name.startswith("base."):
            torch.testing.assert_close(second["model_state_dict"][name], value, rtol=0, atol=0)
    model, _ = evaluate_light.load_model(directory / "checkpoint_latest.pt", torch.device("cpu"), QUALITY_CANDIDATE)
    assert set(model.state_dict()) == set(second["model_state_dict"])
    assert not any("discriminator" in name for name in model.state_dict())
    assert digest(warmstart) == source_hash


def test_metric_resume_rejects_changed_objective(metric_case):
    _, argv = metric_case
    run_cli("train_light_metric.py", [*argv, "--num-epochs", "0"])
    with pytest.raises(ValueError, match="metric_weight differs"):
        metric.parse_args([*argv, "--metric-weight", "0.1"])
    with pytest.raises(ValueError, match="FP32"):
        metric.parse_args([*argv, "--use-amp", "yes"])


def test_actual_two_rank_metric_update(metric_case, tmp_path):
    _, argv = metric_case
    argv = [*argv, "--num-epochs", "1", "--parallel-mode", "ddp"]
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONIOENCODING="utf-8")
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(name, None)
    if os.name == "nt":
        result = run_two_windows_gloo_ranks(tmp_path, argv, environment, entry_module="train_light_metric")
    else:
        result = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                                 "--nproc_per_node=2", "train_light_metric.py", *argv],
                                cwd=ROOT, env=environment, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "[METRIC][train]" in result.stdout
    directory = Path(argv[argv.index("--checkpoint-dir") + 1])
    initialized, _ = read_candidate_checkpoint(directory / "model_epoch_0.pt", QUALITY_CANDIDATE)
    trained, _ = read_candidate_checkpoint(directory / "checkpoint_latest.pt", QUALITY_CANDIDATE)
    assert trained["epoch"] == 1 and trained["metric_state"]["updates"] == 1
    assert {int(v["step"]) for v in trained["optimizer_state_dict"]["state"].values()} == {1}
    assert {int(v["step"]) for v in trained["metric_state"]["optimizer"]["state"].values()} == {1}
    assert all(torch.isfinite(v).all() for v in trained["metric_state"]["discriminator"].values())
    assert any(not torch.equal(v, initialized["metric_state"]["discriminator"][k])
               for k, v in trained["metric_state"]["discriminator"].items())


def test_discriminator_ranks_agree_when_one_rank_has_no_label(tmp_path):
    worker = tmp_path / "metric_ddp_capture.py"
    worker.write_text('''
import os
from pathlib import Path
import torch
import torch.distributed as dist
from light_metric_loss import MetricObjective

def main():
    torch.set_num_threads(2)
    if not dist.is_initialized():
        dist.init_process_group('gloo')
    rank = dist.get_rank()
    torch.manual_seed(42)
    objective = MetricObjective('cpu', score_fn=lambda *args: 3.0 if rank == 0 else None)
    objective.training = True
    time = torch.arange(8000) / 16000
    target = (0.1 * torch.sin(2 * torch.pi * 149 * time))[None]
    mixture = target[..., None].repeat(1, 1, 8)
    _, reference = objective.build_features(mixture, target, 512, 160, 320, 0.5,
                                           torch.device('cpu'), [8000])
    estimate = (reference * 0.8).detach().requires_grad_()
    objective(estimate, reference, [51]).backward()
    assert torch.isfinite(estimate.grad).all()
    stats = objective.metric_stats()
    torch.save(dict(state=objective.state_dict(), stats=stats),
               Path(__file__).with_name(f'rank_{rank}.pt'))
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
''', encoding="utf-8")
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       PYTHONPATH=os.pathsep.join((str(tmp_path), str(ROOT))))
    if os.name == "nt":
        result = run_two_windows_gloo_ranks(tmp_path, [], environment, entry_module="metric_ddp_capture")
    else:
        result = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                                 "--nproc_per_node=2", str(worker)], cwd=ROOT, env=environment,
                                capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    one, two = [torch.load(tmp_path / f"rank_{rank}.pt", weights_only=False) for rank in range(2)]
    assert one["stats"]["valid_pesq_labels"] == 1 and one["stats"]["invalid_pesq_labels"] == 1
    assert one["state"]["updates"] == two["state"]["updates"] == 1
    for name, value in one["state"]["discriminator"].items():
        torch.testing.assert_close(value, two["state"]["discriminator"][name], rtol=0, atol=0)
    for first, second in zip(one["state"]["optimizer"]["state"].values(), two["state"]["optimizer"]["state"].values()):
        for name in first:
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)


@pytest.mark.parametrize("fault", ["gain_matching", "partial_high_score"])
def test_launch_gates_do_not_accept_changed_protocol_or_partial_score(tmp_path, monkeypatch, fault):
    shell = (ROOT / "run_light_metric.sh").read_text()
    marker = "PY_BASELINE" if fault == "gain_matching" else "PY_TARGET"
    code = shell.split(f"<<'{marker}'\n", 1)[1].split(f"\n{marker}", 1)[0]
    report = dict(status="complete", completed_samples=600, expected_samples=600,
                  frontend={"n_fft": 512}, frontend_protocol={"waveform_normalization": "none"},
                  num_mics=8, selected_pairs_sha256="same", metadata_sha256="same",
                  metric_protocol={"target_gain_matching": False}, model_parameters=2_625_088,
                  candidate_implementation_sha256={"same": "implementation"},
                  mean={"enhanced": {"pesq": 4.0}, "noisy": {"pesq": 1.1}})
    source = tmp_path / "source.json"
    candidate = tmp_path / "candidate.json"
    source.write_text(json.dumps(report))
    if fault == "gain_matching":
        report["metric_protocol"]["target_gain_matching"] = True
    else:
        report["completed_samples"] = 599
    candidate.write_text(json.dumps(report))
    order = [source, candidate] if fault == "gain_matching" else [candidate, source]
    monkeypatch.setattr(sys, "argv", ["-", *map(str, order)])
    with pytest.raises(AssertionError):
        exec(compile(code, "metric launcher gate", "exec"), {})

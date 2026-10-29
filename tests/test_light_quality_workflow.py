"""Quality warm-start, isolation and real train/resume/evaluation plumbing."""

import csv
import hashlib
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
from light_candidate_runtime import candidate_metadata, read_candidate_checkpoint
from light_quality_model import BASE_CANDIDATE, QUALITY_CANDIDATE
import train_light
import train_light_quality as quality


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def source_case(tmp_path_factory):
    torch.set_num_threads(2)
    directory = tmp_path_factory.mktemp("quality_source")
    dataset = directory / "dataset"
    dataset.mkdir()
    time = np.arange(32000) / 16000
    speech = (0.1 * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * time)) *
              (np.sin(2 * np.pi * 143 * time) + 0.4 * np.sin(2 * np.pi * 286 * time))).astype(np.float32)
    mixture = speech[:, None] + np.random.default_rng(4).normal(0, 0.01, (len(time), 9))
    target = np.stack([speech * 0.5, speech * 0.8, speech], axis=1)
    sf.write(dataset / "mixture.wav", mixture.astype(np.float32), 16000, subtype="FLOAT")
    sf.write(dataset / "target.wav", target, 16000, subtype="FLOAT")
    with (dataset / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "mixture_path", "target_path"])
        writer.writerow([17, "mixture.wav", "target.wav"])
    args = train_light.parse_args(["--train-dir", str(dataset), "--val-dir", str(dataset),
                                  "--target-ref-mic", "2"])
    for name, value in candidate_metadata(BASE_CANDIDATE).items():
        setattr(args, name, value)
    model = evaluate_light.create_model(evaluate_light.model_config_from_checkpoint(vars(args)),
                                        torch.device("cpu"), BASE_CANDIDATE)
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean.fill_(0.03)
            module.running_var.fill_(0.95)
            module.num_batches_tracked.fill_(40)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0000625)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.0001)
    optimizer.step()
    source_dir = directory / "original_checkpoints"
    source_dir.mkdir()
    checkpoint_path = source_dir / "model_epoch_40.pt"
    torch.save({"epoch": 40, "args": vars(args), "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "best_val_loss": 0.02}, checkpoint_path)
    return dict(directory=directory, dataset=dataset, path=checkpoint_path,
                checkpoint=torch.load(checkpoint_path, map_location="cpu", weights_only=False))


def arguments(source, directory):
    return ["--warmstart", str(source["path"]), "--train-dir", str(source["dataset"]),
            "--val-dir", str(source["dataset"]), "--checkpoint-dir", str(directory / "checkpoints"),
            "--best-dir", str(directory / "best"), "--log-dir", str(directory / "logs"),
            "--num-workers", "0", "--parallel-mode", "none", "--cpu-threads", "2",
            "--interop-threads", "1", "--use-amp", "no", "--model-amp", "no",
            "--batch-size", "1", "--grad-accum-steps", "1", "--segment-seconds", "0.25",
            "--train-random-crop", "no", "--mem-log-every-batches", "0",
            "--gc-every-batches", "0", "--malloc-trim-every-batches", "0", "--log-perf", "no"]


def test_warmstart_inherits_frontend_and_base_exactly_with_fresh_adam(source_case, tmp_path):
    before = digest(source_case["path"])
    args, source = quality.parse_args(arguments(source_case, tmp_path))
    assert args.target_ref_mic == 2
    assert args.quality_refiner_causal is False and args.is_causal == "yes"
    assert args.quality_source_sha256 == before
    quality.prepare_checkpoint(args, source)
    checkpoint_path = tmp_path / "checkpoints" / "checkpoint_latest.pt"
    checkpoint, candidate = read_candidate_checkpoint(checkpoint_path, QUALITY_CANDIDATE)
    assert candidate == QUALITY_CANDIDATE
    assert checkpoint["epoch"] == 0 and checkpoint["best_val_loss"] == float("inf")
    assert checkpoint["optimizer_state_dict"]["state"] == {}
    assert checkpoint["lr_reducer_state"] == {}
    assert checkpoint["scaler_state_dict"] == {}
    for name, tensor in source["model_state_dict"].items():
        torch.testing.assert_close(checkpoint["model_state_dict"]["base." + name], tensor, rtol=0, atol=0)
    assert (tmp_path / "checkpoints" / "model_epoch_0.pt").exists()
    assert digest(source_case["path"]) == before


@pytest.mark.parametrize("override", [["--power", "1.0"], ["--num-mics=7"]])
def test_warmstart_rejects_explicit_frontend_or_model_conflicts(source_case, tmp_path, override):
    with pytest.raises(ValueError, match="differs from the source checkpoint"):
        quality.parse_args([*arguments(source_case, tmp_path), *override])
    assert not (tmp_path / "checkpoints").exists()


def test_outputs_cannot_overlap_original_dataset_or_checkpoint(source_case, tmp_path):
    for protected in (source_case["dataset"], source_case["path"].parent):
        with pytest.raises(ValueError, match="overlaps source checkpoint or dataset"):
            quality.parse_args([*arguments(source_case, tmp_path), "--checkpoint-dir", str(protected / "new")])


@pytest.mark.parametrize("field,value", [
    ("quality_objective", "different_loss"), ("quality_source_sha256", "0" * 64),
    ("quality_refiner_causal", True),
])
def test_resume_rejects_changed_quality_protocol(source_case, tmp_path, field, value):
    argv = arguments(source_case, tmp_path)
    args, source = quality.parse_args(argv)
    quality.prepare_checkpoint(args, source)
    path = tmp_path / "checkpoints" / "checkpoint_latest.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["args"][field] = value
    torch.save(checkpoint, path)
    before = digest(path)
    with pytest.raises(ValueError, match="differs from checkpoint"):
        quality.parse_args(argv)
    assert digest(path) == before


def test_existing_output_is_not_reinitialized(source_case, tmp_path):
    argv = arguments(source_case, tmp_path)
    args, source = quality.parse_args(argv)
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    (directory / "notes.txt").write_text("Keep me", encoding="utf-8")
    with pytest.raises(FileExistsError, match="empty output directory"):
        quality.prepare_checkpoint(args, source)
    assert (directory / "notes.txt").read_text(encoding="utf-8") == "Keep me"
    assert not (directory / "checkpoint_latest.pt").exists()


@pytest.mark.parametrize("fault", ["schema", "implementation", "missing_head"])
def test_evaluation_rejects_incompatible_quality_checkpoints(source_case, tmp_path, fault):
    args, source = quality.parse_args(arguments(source_case, tmp_path))
    quality.prepare_checkpoint(args, source)
    path = tmp_path / "checkpoints" / "checkpoint_latest.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if fault == "schema":
        checkpoint["args"]["candidate_schema"] = 1
    elif fault == "implementation":
        checkpoint["args"]["candidate_implementation_sha256"]["light_quality_model.py"] = "0" * 64
    else:
        del checkpoint["model_state_dict"]["refiner.head.weight"]
    torch.save(checkpoint, path)
    with pytest.raises((ValueError, RuntimeError)):
        evaluate_light.load_model(path, torch.device("cpu"), QUALITY_CANDIDATE)


def run_cli(script, argv):
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       PYTHONIOENCODING="utf-8")
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(name, None)
    result = subprocess.run([sys.executable, str(ROOT / script), *map(str, argv)], cwd=ROOT,
                            env=environment, text=True, encoding="utf-8", errors="replace",
                            capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    return result


def test_real_train_resume_and_five_metric_evaluation(source_case, tmp_path):
    before = digest(source_case["path"])
    argv = arguments(source_case, tmp_path)
    first_log = run_cli("train_light_quality.py", [*argv, "--num-epochs", "1"])
    latest = tmp_path / "checkpoints" / "checkpoint_latest.pt"
    first = torch.load(latest, map_location="cpu", weights_only=False)
    assert first["epoch"] == 1
    first_steps = {int(value["step"].item()) for value in first["optimizer_state_dict"]["state"].values()}
    assert first_steps == {1}
    resumed = [value for index, value in enumerate(argv) if index not in (0, 1)]
    second_log = run_cli("train_light_quality.py", [*resumed, "--num-epochs", "2"])
    second = torch.load(latest, map_location="cpu", weights_only=False)
    assert second["epoch"] == 2
    second_steps = {int(value["step"].item()) for value in second["optimizer_state_dict"]["state"].values()}
    assert second_steps == {2}
    assert "[QUALITY][train][rank0]" in first_log.stdout and "at epoch 1" in second_log.stdout
    for name, tensor in source_case["checkpoint"]["model_state_dict"].items():
        torch.testing.assert_close(second["model_state_dict"]["base." + name], tensor, rtol=0, atol=0)
    assert any(not torch.equal(first["model_state_dict"][name], tensor)
               for name, tensor in second["model_state_dict"].items() if not name.startswith("base."))
    restored, _ = evaluate_light.load_model(latest, torch.device("cpu"), QUALITY_CANDIDATE)
    for name, tensor in second["model_state_dict"].items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)
    report_path = tmp_path / "metrics.json"
    run_cli("evaluate_light.py", ["--checkpoint", latest, "--val-dir", source_case["dataset"],
            "--device", "cpu", "--save-samples", "no", "--match-estimate-level", "no",
            "--estimate-dir", tmp_path / "estimates", "--save-json", report_path])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "complete" and report["completed_samples"] == 1
    assert report["structure_candidate"] == QUALITY_CANDIDATE
    assert report["model_parameters"] == 2_625_088
    assert report["quality_training"]["quality_source_sha256"] == before
    assert report["quality_training"]["quality_refiner_causal"] is False
    assert report["frontend"]["target_ref_mic"] == 2
    assert report["metric_protocol"]["target_gain_matching"] is False
    assert set(report["mean"]["enhanced"]) == {"pesq", "stoi", "estoi", "si_snr_db", "sdr_db"}
    assert digest(source_case["path"]) == before

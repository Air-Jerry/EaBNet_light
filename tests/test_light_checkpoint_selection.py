"""Checkpoint selection must compare complete, independent validation runs."""

import csv
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf
import torch

import evaluate_light as evaluation
from light_candidate_runtime import candidate_metadata
from light_reference_variants import build_reference_candidate
import train_light


ROOT = Path(__file__).resolve().parents[1]


def write_dataset(directory, count=2):
    directory.mkdir()
    time = np.arange(32000) / 16000
    envelope = 0.15 * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * time))
    reference = (envelope * (np.sin(2 * np.pi * 143 * time)
                            + 0.4 * np.sin(2 * np.pi * 286 * time)
                            + 0.2 * np.sin(2 * np.pi * 429 * time))).astype(np.float32)
    rows = []
    for sample_id in range(1, count + 1):
        mixture = reference[:, None] + np.random.default_rng(sample_id).normal(0, 0.01, (32000, 8))
        mixture_path, target_path = directory / f"mix_{sample_id}.wav", directory / f"target_{sample_id}.wav"
        sf.write(mixture_path, mixture, 16000, subtype="FLOAT")
        sf.write(target_path, reference, 16000, subtype="FLOAT")
        rows.append([sample_id, mixture_path.name, target_path.name])
    with (directory / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "mixture_path", "target_path"])
        writer.writerows(rows)
    return directory


@pytest.fixture
def selection_case(tmp_path):
    dataset = write_dataset(tmp_path / "validation")
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoints = []
    for name in ("model_epoch_5.pt", "model_epoch_10.pt", "checkpoint_latest.pt", "best_model.pt"):
        path = checkpoint_dir / name
        path.write_bytes(name.encode("utf-8"))
        checkpoints.append(path)
    (checkpoint_dir / "notes.txt").write_text("Ignore unrelated files", encoding="utf-8")
    output = tmp_path / "selection"
    arguments = ["--val-dir", str(dataset), "--checkpoint-dir", str(checkpoint_dir),
                 "--estimate-dir", str(output), "--device", "cpu"]
    return dataset, checkpoint_dir, checkpoints, output, arguments


def complete_report(args, pesq_score):
    records, source, _ = evaluation.resolve_evaluation_inputs(args)
    if args.max_samples:
        records = records[:args.max_samples]
    checkpoint = Path(args.checkpoint).resolve()
    noisy = {"pesq": 1.2, "stoi": 0.6, "estoi": 0.4, "si_snr_db": -1.0, "sdr_db": -0.5}
    enhanced = {"pesq": pesq_score, "stoi": 0.94, "estoi": 0.87, "si_snr_db": 13.0, "sdr_db": 14.0}
    return {
        "status": "complete", "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "dataset": str(Path(args.val_dir).resolve()), "input_source": source,
        "metadata_sha256": source["metadata_sha256"],
        "selected_pairs_sha256": evaluation.pairs_sha256(records),
        "frontend": {"sample_rate": 16000, "n_fft": 512, "hop_length": 160,
                     "win_length": 320, "power": 0.5, "target_ref_mic": 0},
        "frontend_protocol": {"normalized": False, "waveform_normalization": "none"},
        "num_mics": 8, "checkpoint_model_config": {"num_mics": 8, "channels": 64},
        "model_parameters": 735070, "structure_candidate": "cbam_flat_projection64",
        "candidate_implementation_sha256": {"model.py": "unchanged"},
        "candidate_evidence": {"author_confirmed": False},
        "mixture_ref_mic": 0, "device": "cpu", "versions": {"pesq": "test"},
        "metric_protocol": {"pesq": "pesq wideband", "target_gain_matching": False,
                            "external_normalization_clipping_or_alignment": False},
        "expected_samples": len(records), "completed_samples": len(records),
        "mean": {"enhanced": enhanced, "noisy": noisy},
    }


def install_evaluator(monkeypatch, transform=None):
    calls = []
    scores = {"model_epoch_5.pt": 3.1, "model_epoch_10.pt": 3.5,
              "checkpoint_latest.pt": 3.2, "best_model.pt": 3.0}

    def fake_evaluate(args):
        calls.append(deepcopy(args))
        report = complete_report(args, scores[Path(args.checkpoint).name])
        if transform is not None:
            report = transform(report, len(calls))
        return report

    monkeypatch.setattr(evaluation, "evaluate", fake_evaluate)
    return calls


def test_selection_ranks_validation_pesq_and_isolates_each_checkpoint(selection_case, monkeypatch):
    dataset, _, checkpoints, output, arguments = selection_case
    calls = install_evaluator(monkeypatch)
    result = evaluation.main(arguments)
    report = json.loads((output / "checkpoint_ranking.json").read_text(encoding="utf-8"))
    assert result == report
    assert report["status"] == "complete"
    assert Path(report["best_checkpoint"]) == checkpoints[1]
    assert len(report["results"]) == len(calls) == len(checkpoints)
    assert [row["mean"]["enhanced"]["pesq"] for row in report["results"]] == [3.5, 3.2, 3.1, 3.0]
    assert {Path(call.checkpoint) for call in calls} == set(checkpoints)
    assert len({str(call.estimate_dir) for call in calls}) == len(checkpoints)
    for call in calls:
        assert Path(call.val_dir) == dataset
        assert call.save_samples == "no" and call.match_estimate_level == "no"
        assert output in Path(call.estimate_dir).parents
        assert Path(call.estimate_dir).name == Path(call.checkpoint).stem
        assert not call.save_csv or Path(call.save_csv) != output / "checkpoint_ranking.csv"
        assert not call.save_json or Path(call.save_json) != output / "checkpoint_ranking.json"
    with (output / "checkpoint_ranking.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(checkpoints)
    assert Path(rows[0]["checkpoint"]) == checkpoints[1]


@pytest.mark.parametrize("change", [
    "selected_pairs", "metadata", "frontend", "microphones", "metric_protocol",
    "noisy_baseline", "candidate", "sample_count", "incomplete", "nonfinite",
])
def test_selection_rejects_incomparable_or_incomplete_results(selection_case, monkeypatch, change):
    _, _, _, output, arguments = selection_case

    def mutate(report, call_number):
        if call_number != 2:
            return report
        if change == "selected_pairs":
            report["selected_pairs_sha256"] = "different ordering or selected files"
        elif change == "metadata":
            report["metadata_sha256"] = "changed references"
        elif change == "frontend":
            report["frontend"]["power"] = 1.0
        elif change == "microphones":
            report["num_mics"] = 4
        elif change == "metric_protocol":
            report["metric_protocol"]["target_gain_matching"] = True
        elif change == "noisy_baseline":
            report["mean"]["noisy"]["pesq"] += 0.1
        elif change == "candidate":
            report["structure_candidate"] = "literal_per_frequency"
        elif change == "sample_count":
            report["expected_samples"] = report["completed_samples"] = 1
        elif change == "incomplete":
            report["completed_samples"] -= 1
        elif change == "nonfinite":
            report["mean"]["enhanced"]["pesq"] = float("nan")
        return report

    calls = install_evaluator(monkeypatch, mutate)
    with pytest.raises((ValueError, RuntimeError)):
        evaluation.main(arguments)
    report = json.loads((output / "checkpoint_ranking.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed" and report["best_checkpoint"] is None
    assert len(calls) == 2
    assert "error" in report


def test_failed_checkpoint_cannot_leave_a_successful_partial_winner(selection_case, monkeypatch):
    _, _, _, output, arguments = selection_case

    def fail_second(report, count):
        if count == 2:
            raise RuntimeError("damaged checkpoint")
        return report

    install_evaluator(monkeypatch, fail_second)
    with pytest.raises(RuntimeError, match="damaged checkpoint"):
        evaluation.main(arguments)
    report = json.loads((output / "checkpoint_ranking.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["best_checkpoint"] is None
    assert "damaged checkpoint" in report["error"]


@pytest.mark.parametrize("destination", ["checkpoint", "metadata", "unselected_audio"])
@pytest.mark.parametrize("output_flag", ["--save-json", "--save-csv"])
def test_selection_protects_all_inputs_before_evaluating(selection_case, monkeypatch, destination, output_flag):
    dataset, _, checkpoints, _, arguments = selection_case
    path = {"checkpoint": checkpoints[-1], "metadata": dataset / "metadata.csv",
            "unselected_audio": dataset / "target_2.wav"}[destination]
    original = path.read_bytes()
    calls = install_evaluator(monkeypatch)
    with pytest.raises(ValueError):
        evaluation.main([*arguments, "--max-samples", "1", output_flag, str(path)])
    assert not calls
    assert path.read_bytes() == original


def test_per_checkpoint_output_cannot_overwrite_dataset_metadata(selection_case, monkeypatch):
    dataset, _, _, output, arguments = selection_case
    conflicting = output / "checkpoints" / "best_model" / "metadata.csv"
    conflicting.parent.mkdir(parents=True)
    conflicting.write_text("input audio content", encoding="utf-8")
    with (dataset / "metadata.csv").open("a", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow([3, str(conflicting), "target_2.wav"])
    calls = install_evaluator(monkeypatch)
    with pytest.raises(ValueError):
        evaluation.main([*arguments, "--max-samples", "1"])
    assert not calls
    assert conflicting.read_text(encoding="utf-8") == "input audio content"


@pytest.mark.parametrize("filename", ["checkpoint_ranking.csv", "checkpoint_ranking.json"])
def test_default_ranking_symlinks_cannot_overwrite_checkpoint(selection_case, monkeypatch, filename):
    _, _, checkpoints, output, arguments = selection_case
    output.mkdir()
    original = checkpoints[0].read_bytes()
    try:
        (output / filename).symlink_to(checkpoints[0])
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Filesystem does not permit symlink creation: {exc}")
    calls = install_evaluator(monkeypatch)
    with pytest.raises(ValueError, match="overwrite"):
        evaluation.main(arguments)
    assert not calls
    assert checkpoints[0].read_bytes() == original


@pytest.mark.parametrize("kind", ["missing", "empty", "file"])
def test_invalid_checkpoint_directory_fails_before_evaluating(tmp_path, monkeypatch, kind):
    dataset = write_dataset(tmp_path / "validation", count=1)
    checkpoint_dir = tmp_path / "checkpoints"
    if kind == "empty":
        checkpoint_dir.mkdir()
        (checkpoint_dir / "unrelated.pt").write_bytes(b"not a training checkpoint")
    elif kind == "file":
        checkpoint_dir.write_bytes(b"not a directory")
    calls = install_evaluator(monkeypatch)
    with pytest.raises((ValueError, FileNotFoundError, NotADirectoryError)):
        evaluation.main(["--val-dir", str(dataset), "--checkpoint-dir", str(checkpoint_dir),
                         "--estimate-dir", str(tmp_path / "results")])
    assert not calls


def test_single_checkpoint_cli_keeps_original_defaults(monkeypatch):
    args = evaluation.parse_args([])
    assert args.checkpoint == "./bestmodels_cbam_flat_projection64/best_model.pt"
    assert args.val_dir == "./validation_set" and args.save_samples == "yes"
    received = []
    monkeypatch.setattr(evaluation, "evaluate", lambda args: received.append(args) or "single result")
    assert evaluation.main(["--checkpoint", "trained.pt"]) == "single result"
    assert received[0].checkpoint == "trained.pt"


@pytest.mark.parametrize("arguments", [
    ["--checkpoint-dir", "checkpoints"],
    ["--checkpoint-dir", "checkpoints", "--checkpoint", "single.pt", "--val-dir", "validation"],
    ["--checkpoint-dir", "checkpoints", "--mixture-path", "mix.wav", "--target-path", "target.wav"],
    ["--checkpoint-dir", "checkpoints", "--val-dir", "validation", "--save-samples", "yes"],
    ["--checkpoint-dir", "checkpoints", "--val-dir", "validation", "--match-estimate-level", "yes"],
])
def test_unsafe_or_ambiguous_sweep_cli_is_rejected(arguments):
    with pytest.raises(SystemExit):
        evaluation.parse_args(arguments)


def test_real_candidate_checkpoint_sweep_runs_all_metrics_in_a_separate_process(tmp_path):
    dataset = write_dataset(tmp_path / "validation", count=1)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    candidate_id = "cbam_flat_projection64"
    torch.manual_seed(91)
    model = build_reference_candidate(candidate_id)
    saved_args = vars(train_light.parse_args([]))
    saved_args.update(candidate_metadata(candidate_id))
    for epoch in (5, 10):
        torch.save({"args": saved_args, "epoch": epoch, "model_state_dict": model.state_dict()},
                   checkpoint_dir / f"model_epoch_{epoch}.pt")
    output = tmp_path / "selection"
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONIOENCODING="utf-8")
    result = subprocess.run([sys.executable, str(ROOT / "evaluate_light.py"),
                             "--val-dir", str(dataset), "--checkpoint-dir", str(checkpoint_dir),
                             "--estimate-dir", str(output), "--device", "cpu"],
                            cwd=ROOT, env=environment, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=120)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    report = json.loads((output / "checkpoint_ranking.json").read_text(encoding="utf-8"))
    assert report["status"] == "complete" and len(report["results"]) == 2
    assert Path(report["best_checkpoint"]).is_file()
    assert not list(output.rglob("*.wav"))
    for entry in report["results"]:
        run = output / "checkpoints" / Path(entry["checkpoint"]).stem
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "complete"
        assert summary["completed_samples"] == summary["expected_samples"] == 1
        assert summary["structure_candidate"] == candidate_id and summary["num_mics"] == 8
        assert summary["frontend"]["power"] == 0.5
        assert summary["metric_protocol"]["target_gain_matching"] is False
        assert summary["outputs"]["save_samples"] is False
        for kind in ("enhanced", "noisy"):
            assert set(summary["mean"][kind]) == set(evaluation.METRICS)
            assert all(math.isfinite(value) for value in summary["mean"][kind].values())
        with (run / "metadata.csv").open(newline="", encoding="utf-8") as handle:
            assert len(list(csv.DictReader(handle))) == 1

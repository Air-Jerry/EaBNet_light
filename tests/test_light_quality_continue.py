"""Execute the launcher's actual resume and evaluation gates without a GPU."""

import hashlib
import json
from pathlib import Path
import sys

import pytest

import light_candidate_runtime
import train_light


SCRIPT = (Path(__file__).resolve().parents[1] / "continue_light_quality.sh").read_text()
PLAN = SCRIPT.split("<<'PY_PLAN'\n", 1)[1].split("\nPY_PLAN", 1)[0]
RESULT = SCRIPT.split("<<'PY_RESULT'\n", 1)[1].split("\nPY_RESULT", 1)[0]


@pytest.fixture
def run_case(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / "checkpoints_cbam_refiner_v1_quality_test"
    best = tmp_path / "bestmodels_cbam_refiner_v1_quality_test"
    logs = tmp_path / "logs_cbam_refiner_v1/quality_test"
    for path in (directory, best, logs):
        path.mkdir(parents=True)
    stable = directory / "model_epoch_10.pt"
    stable.write_bytes(b"saved model and Adam state")
    validation = tmp_path / "validation"
    validation.mkdir()
    (validation / "metadata.csv").write_bytes(b"same validation metadata")
    args = vars(train_light.parse_args([]))
    args.update(checkpoint_dir=str(directory), best_dir=str(best), log_dir=str(logs),
                quality_source_checkpoint=str(tmp_path / "original40.pt"), train_dir="/data/train",
                val_dir=str(validation), batch_size=1, grad_accum_steps=4, cpu_threads=8)
    checkpoint = {"epoch": 10, "args": args}
    monkeypatch.setattr(light_candidate_runtime, "read_candidate_checkpoint", lambda *a: (checkpoint, "cbam_refiner_v1"))
    report = dict(status="complete", completed_samples=600, expected_samples=600,
                  frontend={"n_fft": 512}, frontend_protocol={"waveform_normalization": "none"},
                  num_mics=8, selected_pairs_sha256="same-pairs",
                  metadata_sha256=hashlib.sha256((validation / "metadata.csv").read_bytes()).hexdigest(),
                  metric_protocol={"target_gain_matching": False},
                  checkpoint_sha256=hashlib.sha256(stable.read_bytes()).hexdigest(),
                  mean={"enhanced": {"pesq": 2.8211450682083767}})
    paths = []
    for epoch in (0, 10):
        path = logs / f"pesq_selection/model_epoch_{epoch}/summary.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(report))
        paths.append(path)
    return directory, checkpoint, paths


def test_resume_starts_after_saved_epoch_and_retains_settings(run_case, monkeypatch, capsys):
    directory, checkpoint, _ = run_case
    monkeypatch.setattr(sys, "argv", ["-", str(directory), "20"])
    exec(compile(PLAN, "resume preflight", "exec"), {})
    values = capsys.readouterr().out.splitlines()
    assert values[6] == "11"
    options = dict(zip(values[7::2], values[8::2]))
    assert options["--batch-size"] == "1" and options["--grad-accum-steps"] == "4"
    assert options["--cpu-threads"] == "8" and options["--cpu-thread-scale"] == "1.0"
    assert checkpoint["epoch"] == 10


@pytest.mark.parametrize("fault", ["completed_limit", "different_dataset", "incomplete", "checkpoint_changed", "output_changed", "validation_csv_changed"])
def test_resume_rejects_unusable_or_different_run(run_case, monkeypatch, fault):
    directory, checkpoint, paths = run_case
    total = "10" if fault == "completed_limit" else "20"
    report = json.loads(paths[1].read_text())
    if fault == "different_dataset":
        report["selected_pairs_sha256"] = "different"
    elif fault == "incomplete":
        report["completed_samples"] = 599
    elif fault == "checkpoint_changed":
        (directory / "model_epoch_10.pt").write_bytes(b"different checkpoint")
    elif fault == "output_changed":
        checkpoint["args"]["log_dir"] = str(directory)
    elif fault == "validation_csv_changed":
        (Path(checkpoint["args"]["val_dir"]) / "metadata.csv").write_bytes(b"changed metadata")
    paths[1].write_text(json.dumps(report))
    monkeypatch.setattr(sys, "argv", ["-", str(directory), total])
    with pytest.raises((ValueError, AssertionError)):
        exec(compile(PLAN, "resume preflight", "exec"), {})


@pytest.mark.parametrize("score,stop", [(3.399, "0"), (3.4, "1")])
def test_stop_requires_full_600_sample_target(run_case, monkeypatch, capsys, score, stop):
    _, _, paths = run_case
    report = json.loads(paths[1].read_text())
    report["mean"]["enhanced"]["pesq"] = score
    paths[1].write_text(json.dumps(report))
    monkeypatch.setattr(sys, "argv", ["-", str(paths[1]), str(paths[0])])
    exec(compile(RESULT, "evaluation gate", "exec"), {})
    assert capsys.readouterr().out.strip() == stop


@pytest.mark.parametrize("fault", ["incomplete", "gain_matching", "nan"])
def test_invalid_score_cannot_trigger_success(run_case, monkeypatch, fault):
    _, _, paths = run_case
    report = json.loads(paths[1].read_text())
    if fault == "incomplete":
        report["completed_samples"] = 599
    elif fault == "gain_matching":
        report["metric_protocol"]["target_gain_matching"] = True
    else:
        report["mean"]["enhanced"]["pesq"] = float("nan")
    paths[1].write_text(json.dumps(report))
    monkeypatch.setattr(sys, "argv", ["-", str(paths[1]), str(paths[0])])
    with pytest.raises(AssertionError):
        exec(compile(RESULT, "evaluation gate", "exec"), {})

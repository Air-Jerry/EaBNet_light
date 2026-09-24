"""Resume the same fixed-pair experiment without resetting its evidence or Adam."""

import json
from pathlib import Path
import subprocess

import pytest
import torch

import diagnose_light_overfit as diagnostic
from light_candidate_runtime import read_candidate_checkpoint
from test_light_overfit_diagnostic import (
    CANDIDATE,
    arguments,
    digest,
    source_checkpoint,
)


def optimizer_steps(checkpoint):
    return {
        int(state["step"].item() if torch.is_tensor(state["step"]) else state["step"])
        for state in checkpoint["optimizer_state_dict"]["state"].values()
        if "step" in state
    }


@pytest.fixture
def interrupted_run(source_checkpoint, tmp_path, monkeypatch):
    """Interrupt the real trainer only after its first completed epoch was saved."""
    output = tmp_path / "experiment"
    args = arguments(source_checkpoint, output, None, "--epochs", "2")
    real_popen = subprocess.Popen

    class StopAfterFirstEpoch:
        def __init__(self, *args, **kwargs):
            self.process = real_popen(*args, **kwargs)
            self.stdout = self.lines()

        def lines(self):
            for line in self.process.stdout:
                if "Epoch 2/2" in line:
                    self.process.terminate()
                    self.process.wait(timeout=15)
                    yield line
                    return
                yield line

        def __enter__(self):
            self.process.__enter__()
            return self

        def __exit__(self, *args):
            return self.process.__exit__(*args)

        def wait(self, **kwargs):
            return self.process.wait(**kwargs)

        def terminate(self):
            self.process.terminate()

        def kill(self):
            self.process.kill()

    with monkeypatch.context() as patched:
        patched.setattr(diagnostic.subprocess, "Popen", StopAfterFirstEpoch)
        with pytest.raises(subprocess.CalledProcessError):
            diagnostic.run(args)
    checkpoint, _ = read_candidate_checkpoint(output / "checkpoints" / "checkpoint_latest.pt", CANDIDATE)
    assert checkpoint["epoch"] == 1
    assert optimizer_steps(checkpoint) == {1}
    summary = json.loads((output / "diagnostic_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["epochs"] == 2 and summary["steps"] == 2
    assert summary["before"]["epoch"] == 0
    return output


def test_real_cpu_resume_preserves_baseline_dataset_and_adam_history(interrupted_run, monkeypatch):
    output = interrupted_run
    summary_before = json.loads((output / "diagnostic_summary.json").read_text(encoding="utf-8"))
    evidence = [output / "initial_checkpoint.pt", output / "subset_manifest.json",
                output / "training_command.json"]
    evidence += [path for directory in (output / "dataset", output / "before")
                 for path in directory.rglob("*") if path.is_file()]
    hashes = {path: digest(path) for path in evidence}
    old_log = (output / "training.log").read_bytes()
    real_evaluate = diagnostic.evaluate_checkpoint
    evaluations = []

    def evaluate_final_only(checkpoint, *args, **kwargs):
        assert Path(checkpoint).name != "initial_checkpoint.pt", "Baseline must not be recomputed"
        evaluations.append(Path(checkpoint).name)
        return real_evaluate(checkpoint, *args, **kwargs)

    monkeypatch.setattr(diagnostic, "evaluate_checkpoint", evaluate_final_only)
    result = diagnostic.run(diagnostic.parse_args(["--resume-run", str(output), "--device", "cpu"]))
    checkpoint, _ = read_candidate_checkpoint(output / "checkpoints" / "checkpoint_latest.pt", CANDIDATE)
    assert result["status"] == "complete"
    assert result["actual_optimizer_steps"] == result["steps"] == 2
    assert checkpoint["epoch"] == 2 and optimizer_steps(checkpoint) == {2}
    assert result["before"] == summary_before["before"]
    assert result["after_latest"]["epoch"] == 2
    assert result["after_best"] is not None
    assert set(evaluations) == {"checkpoint_latest.pt", "best_model.pt"}
    assert all(digest(path) == expected for path, expected in hashes.items())
    new_log = (output / "training.log").read_bytes()
    assert new_log.startswith(old_log)
    appended = new_log[len(old_log):].decode("utf-8")
    assert "at epoch 1" in appended and "Epoch 2/2" in appended
    assert "Epoch 1/2" not in appended
    completed_summary_hash = digest(output / "diagnostic_summary.json")
    repeated = diagnostic.run(diagnostic.parse_args(["--resume-run", str(output), "--device", "cpu"]))
    assert repeated == result
    assert len(evaluations) == 2
    assert (output / "training.log").read_bytes() == new_log
    assert digest(output / "diagnostic_summary.json") == completed_summary_hash


@pytest.mark.parametrize("fault", ["audio", "metadata", "checkpoint_identity", "training_batch"])
def test_resume_rejects_changed_experiment_without_overwriting_report(interrupted_run, monkeypatch, fault):
    output = interrupted_run
    if fault == "audio":
        path = next((output / "dataset").glob("mixture_*.wav"))
        contents = bytearray(path.read_bytes())
        contents[-1] ^= 1
        path.write_bytes(contents)
    elif fault == "metadata":
        path = output / "dataset" / "metadata.csv"
        path.write_bytes(path.read_bytes() + b"\n")
    elif fault == "checkpoint_identity":
        path = output / "checkpoints" / "checkpoint_latest.pt"
        checkpoint, _ = read_candidate_checkpoint(path, CANDIDATE)
        checkpoint["args"]["candidate_schema"] = 999
        torch.save(checkpoint, path)
    elif fault == "training_batch":
        path = output / "training_command.json"
        command = json.loads(path.read_text(encoding="utf-8"))
        command["args"]["batch_size"] = 2
        command["argv"][command["argv"].index("--batch-size") + 1] = "2"
        path.write_text(json.dumps(command), encoding="utf-8")
    summary_hash = digest(output / "diagnostic_summary.json")
    log_hash = digest(output / "training.log")

    def no_process(*args, **kwargs):
        pytest.fail("A changed experiment must be rejected before spawning training")

    monkeypatch.setattr(diagnostic.subprocess, "Popen", no_process)
    with pytest.raises((ValueError, RuntimeError)):
        diagnostic.run(diagnostic.parse_args(["--resume-run", str(output), "--device", "cpu"]))
    assert digest(output / "diagnostic_summary.json") == summary_hash
    assert digest(output / "training.log") == log_hash


@pytest.mark.parametrize("extra", [["--checkpoint", "other.pt"], ["--epochs", "2"], ["--batch-size", "1"]])
def test_resume_cli_rejects_experiment_overrides(tmp_path, extra):
    with pytest.raises(SystemExit) as error:
        diagnostic.parse_args(["--resume-run", str(tmp_path), "--device", "cpu", *extra])
    assert error.value.code == 2


def test_active_run_lock_blocks_duplicate_resume_and_releases_on_close(tmp_path):
    with diagnostic.acquire_run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="lock|active"):
            diagnostic.run(diagnostic.parse_args(["--resume-run", str(tmp_path), "--device", "cpu"]))
    assert not (tmp_path / "diagnostic_summary.json").exists()
    with diagnostic.acquire_run_lock(tmp_path):
        pass

"""Behavioral checks for the isolated fixed-training-pair diagnostic.

The synthetic signals verify plumbing and numerical preservation; they are not
evidence that the trained model reaches any speech quality target.
"""

import csv
from copy import deepcopy
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
import train_light


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = "cbam_flat_projection64"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_dataset(directory, *, duration=2.0, count=1, silence=False):
    directory.mkdir(parents=True)
    time = np.arange(round(duration * 16000), dtype=np.float64) / 16000
    speech = (0.6 + 0.4 * np.sin(2 * np.pi * 3 * time)) * (
        np.sin(2 * np.pi * 143 * time)
        + 0.4 * np.sin(2 * np.pi * 286 * time)
        + 0.2 * np.sin(2 * np.pi * 429 * time))
    rows = []
    for index in range(count):
        clean = (1.2 * speech * (1 + index / 20)).astype(np.float32)
        if silence:
            clean.fill(0)
        target = np.stack([0.5 * clean, -0.6 * clean, clean, 0.8 * clean], axis=1)
        noise = np.random.default_rng(43 + index).normal(0, 0.015, (len(time), 10))
        mixture = np.stack([(0.7 + mic / 20) * clean for mic in range(10)], axis=1) + noise
        mixture = mixture.astype(np.float32)
        mixture_path = directory / f"mixture_{index}.wav"
        target_path = directory / f"target_{index}.wav"
        sf.write(mixture_path, mixture, 16000, subtype="FLOAT")
        sf.write(target_path, target, 16000, subtype="FLOAT")
        rows.append([index + 17, mixture_path.name, target_path.name])
    with (directory / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "mixture_path", "target_path"])
        writer.writerows(rows)
    return directory


@pytest.fixture(scope="module")
def source_checkpoint(tmp_path_factory):
    torch.set_num_threads(2)
    torch.manual_seed(81)
    directory = tmp_path_factory.mktemp("overfit_source")
    dataset = write_dataset(directory / "training")
    saved = vars(train_light.parse_args([
        "--target-ref-mic", "2", "--train-dir", str(dataset),
        "--val-dir", str(dataset), "--batch-size", "4", "--grad-accum-steps", "4",
        "--segment-seconds", "6", "--num-epochs", "40", "--parallel-mode", "ddp",
    ]))
    saved.update(candidate_metadata(CANDIDATE))
    model = evaluate_light.create_model(evaluate_light.model_config_from_checkpoint(saved),
                                       torch.device("cpu"), CANDIDATE)
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean.fill_(0.02)
            module.running_var.fill_(0.9)
            module.num_batches_tracked.fill_(37)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000125)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.0001)
    optimizer.step()
    for state in optimizer.state.values():
        state["step"].fill_(700)
    checkpoint = {
        "epoch": 40, "train_loss": 0.02, "val_loss": 0.03, "best_val_loss": 0.01,
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": {"scale": 8192.0},
        "lr_reducer_state": {"best_plateau_loss_for_lr": 0.01,
                             "stale_train_epochs": 1, "metric": "val"},
        "args": saved,
    }
    path = directory / "source_epoch_40.pt"
    torch.save(checkpoint, path)
    return {"path": path, "dataset": dataset, "checkpoint": checkpoint, "saved": saved}


def arguments(source, output, dataset=None, *extra):
    import diagnose_light_overfit as diagnostic

    return diagnostic.parse_args([
        "--checkpoint", str(source["path"]), "--train-dir", str(dataset or source["dataset"]),
        "--output-dir", str(output), "--num-samples", "1", "--segment-seconds", "1",
        "--epochs", "1", "--batch-size", "1", "--device", "cpu", *map(str, extra),
    ])


def metadata_rows(directory):
    with (directory / "metadata.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolved_audio(directory, row, name):
    path = Path(row[name])
    return path if path.is_absolute() else directory / path


def test_training_config_reuses_original_frontend_and_isolates_the_experiment(source_checkpoint, tmp_path):
    import diagnose_light_overfit as diagnostic

    args = arguments(source_checkpoint, tmp_path / "experiment")
    dataset = tmp_path / "experiment" / "dataset"
    training = diagnostic.make_training_args(source_checkpoint["saved"], args, dataset, Path(args.output_dir))
    for name in ("sample_rate", "n_fft", "hop_length", "win_length", "power", "target_ref_mic",
                 "channels", "num_mics", "embed_dim", "dfsmn_layers", "dfsmn_memory_size"):
        assert getattr(training, name) == source_checkpoint["saved"][name]
    assert Path(training.train_dir) == Path(training.val_dir) == dataset
    assert training.segment_seconds == 0
    assert training.train_random_crop == "no"
    assert training.batch_size == training.grad_accum_steps == 1
    assert training.parallel_mode == "none"
    assert training.use_amp == training.model_amp == training.allow_tf32 == "no"
    assert training.train_lr_reduce_on_plateau == "no"
    assert training.resume == "yes"
    assert training.num_epochs == 1
    for name in ("checkpoint_dir", "best_dir", "log_dir"):
        assert Path(getattr(training, name)).is_relative_to(Path(args.output_dir))


def test_initial_checkpoint_keeps_every_model_tensor_but_clears_training_history(source_checkpoint, tmp_path):
    import diagnose_light_overfit as diagnostic

    args = arguments(source_checkpoint, tmp_path / "experiment")
    training = diagnostic.make_training_args(source_checkpoint["saved"], args,
                                             Path(args.output_dir) / "dataset", Path(args.output_dir))
    model, _ = evaluate_light.load_model(source_checkpoint["path"], torch.device("cpu"))
    initial_path = tmp_path / "fresh.pt"
    before = digest(source_checkpoint["path"])
    diagnostic.write_initial_checkpoint(source_checkpoint["checkpoint"], model, training, initial_path)
    initial, candidate = read_candidate_checkpoint(initial_path, CANDIDATE, training)
    assert candidate == CANDIDATE
    assert initial["epoch"] == 0
    assert initial["best_val_loss"] == float("inf")
    assert initial["optimizer_state_dict"]["state"] == {}
    assert initial.get("scaler_state_dict") in ({}, None)
    reducer = initial.get("lr_reducer_state", {})
    assert reducer.get("stale_train_epochs", 0) == 0
    assert reducer.get("best_plateau_loss_for_lr", float("inf")) == float("inf")
    assert initial["model_state_dict"].keys() == source_checkpoint["checkpoint"]["model_state_dict"].keys()
    for name, value in source_checkpoint["checkpoint"]["model_state_dict"].items():
        assert torch.equal(initial["model_state_dict"][name], value), name
    groups = initial["optimizer_state_dict"]["param_groups"]
    assert len(groups) == 1 and groups[0]["lr"] == training.learning_rate
    assert digest(source_checkpoint["path"]) == before


def test_fixed_exports_keep_waveform_amplitude_channel_order_and_reference(source_checkpoint, tmp_path):
    import diagnose_light_overfit as diagnostic

    output = tmp_path / "experiment"
    args = arguments(source_checkpoint, output)
    original_hashes = {path: digest(path) for path in source_checkpoint["dataset"].iterdir()}
    checkpoint_hash = digest(source_checkpoint["path"])
    manifest = diagnostic.prepare_dataset(args, source_checkpoint["saved"])
    dataset = output / "dataset"
    rows = metadata_rows(dataset)
    assert len(rows) == 1
    mixture_path = resolved_audio(dataset, rows[0], "mixture_path")
    target_path = resolved_audio(dataset, rows[0], "target_path")
    mixture, sr = sf.read(mixture_path, dtype="float32", always_2d=True)
    target, target_sr = sf.read(target_path, dtype="float32", always_2d=True)
    raw_mix, _ = sf.read(source_checkpoint["dataset"] / "mixture_0.wav", dtype="float32", always_2d=True)
    raw_target, _ = sf.read(source_checkpoint["dataset"] / "target_0.wav", dtype="float32", always_2d=True)
    assert sr == target_sr == 16000
    assert mixture.shape == (16000, 8)
    assert len(target) == 16000
    assert sf.info(mixture_path).subtype == sf.info(target_path).subtype == "FLOAT"
    assert abs(mixture).max() > 1 and abs(target).max() > 1
    matching_offsets = [offset for offset in range(len(raw_mix) - len(mixture) + 1)
                        if np.array_equal(raw_mix[offset, :8], mixture[0])]
    assert len(matching_offsets) == 1
    offset = matching_offsets[0]
    np.testing.assert_array_equal(mixture, raw_mix[offset:offset + 16000, :8])
    chosen = target[:, 2] if target.shape[1] > 1 else target[:, 0]
    np.testing.assert_array_equal(chosen, raw_target[offset:offset + 16000, 2])
    assert all(digest(path) == value for path, value in original_hashes.items())
    assert digest(source_checkpoint["path"]) == checkpoint_hash
    assert manifest["selected"][0]["offset_samples"] == offset
    assert manifest["selected"][0]["num_samples"] == 16000
    assert manifest["selected"][0]["source_sample_id"] == 17


def test_same_seed_exports_identical_segments(source_checkpoint, tmp_path):
    import diagnose_light_overfit as diagnostic

    for name in ("first", "second"):
        diagnostic.prepare_dataset(arguments(source_checkpoint, tmp_path / name), source_checkpoint["saved"])
    first, second = tmp_path / "first" / "dataset", tmp_path / "second" / "dataset"
    left, right = metadata_rows(first), metadata_rows(second)
    assert [row["sample_id"] for row in left] == [row["sample_id"] for row in right]
    for lrow, rrow in zip(left, right):
        for name in ("mixture_path", "target_path"):
            assert digest(resolved_audio(first, lrow, name)) == digest(resolved_audio(second, rrow, name))


@pytest.mark.parametrize("fault", ["short", "silence", "duplicate_pair"])
def test_unusable_or_duplicate_training_records_are_rejected(source_checkpoint, tmp_path, fault):
    import diagnose_light_overfit as diagnostic

    dataset = write_dataset(tmp_path / "source", duration=0.2 if fault == "short" else 2,
                            silence=fault == "silence")
    if fault == "duplicate_pair":
        with (dataset / "metadata.csv").open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow([99, "mixture_0.wav", "target_0.wav"])
    source_hashes = {path: digest(path) for path in dataset.iterdir()}
    args = arguments(source_checkpoint, tmp_path / "experiment", dataset)
    if fault == "duplicate_pair":
        args.num_samples = 2
    with pytest.raises((ValueError, RuntimeError)):
        diagnostic.prepare_dataset(args, source_checkpoint["saved"])
    assert all(digest(path) == value for path, value in source_hashes.items())


def test_invalid_candidate_identity_is_rejected_without_modifying_source(source_checkpoint, tmp_path):
    import diagnose_light_overfit as diagnostic

    source = deepcopy(source_checkpoint)
    source["checkpoint"]["args"]["candidate_schema"] = 999
    source["path"] = tmp_path / "source" / "bad.pt"
    source["path"].parent.mkdir()
    torch.save(source["checkpoint"], source["path"])
    before = digest(source["path"])
    with pytest.raises(ValueError, match="candidate_schema"):
        diagnostic.run(arguments(source, tmp_path / "experiment"))
    assert digest(source["path"]) == before


def test_existing_output_directory_is_not_reused_or_cleaned(source_checkpoint, tmp_path):
    import diagnose_light_overfit as diagnostic

    output = tmp_path / "experiment"
    output.mkdir()
    sentinel = output / "existing_result.txt"
    sentinel.write_text("keep this", encoding="utf-8")
    with pytest.raises((ValueError, FileExistsError, RuntimeError)):
        diagnostic.run(arguments(source_checkpoint, output))
    assert sentinel.read_text(encoding="utf-8") == "keep this"


def test_output_inside_input_dataset_is_rejected(source_checkpoint):
    import diagnose_light_overfit as diagnostic

    output = source_checkpoint["dataset"] / "diagnostic_must_not_write_here"
    with pytest.raises((ValueError, FileExistsError, RuntimeError)):
        diagnostic.run(arguments(source_checkpoint, output))
    assert not output.exists()


def test_output_cannot_pollute_audio_directories_referenced_outside_dataset(source_checkpoint, tmp_path):
    import diagnose_light_overfit as diagnostic

    audio = write_dataset(tmp_path / "external_audio")
    dataset = tmp_path / "metadata_only"
    dataset.mkdir()
    with (dataset / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "mixture_path", "target_path"])
        writer.writerow([1, str(audio / "mixture_0.wav"), str(audio / "target_0.wav")])
    output = audio / "do_not_create"
    with pytest.raises((ValueError, FileExistsError, RuntimeError)):
        diagnostic.run(arguments(source_checkpoint, output, dataset))
    assert not output.exists()


def test_training_command_calls_existing_candidate_entrypoint(source_checkpoint, tmp_path):
    import diagnose_light_overfit as diagnostic

    args = arguments(source_checkpoint, tmp_path / "experiment")
    training = diagnostic.make_training_args(source_checkpoint["saved"], args,
                                             tmp_path / "experiment" / "dataset", tmp_path / "experiment")
    command = diagnostic.training_command(training)
    assert command[:3] == [sys.executable, "-u", str(ROOT / "train_light_candidate.py")]
    candidate_index = command.index("--candidate")
    assert command[candidate_index + 1] == CANDIDATE
    original_arguments = command[3:candidate_index] + command[candidate_index + 2:]
    parsed = train_light.parse_args(original_arguments)
    assert parsed.train_dir == parsed.val_dir == training.train_dir
    assert parsed.num_epochs == 1 and parsed.segment_seconds == 0
    assert parsed.grad_accum_steps == 1 and parsed.batch_size == 1
    assert parsed.use_amp == parsed.model_amp == parsed.train_lr_reduce_on_plateau == "no"
    assert parsed.learning_rate == args.learning_rate
    assert parsed.resume == "yes" and parsed.resume_reset_lr == "no"


def test_failure_writes_no_successful_metrics_or_training_result(source_checkpoint, tmp_path, monkeypatch):
    import diagnose_light_overfit as diagnostic

    def fail_evaluation(*args, **kwargs):
        raise RuntimeError("injected baseline metric failure")

    monkeypatch.setattr(diagnostic, "evaluate_checkpoint", fail_evaluation)
    output = tmp_path / "experiment"
    before = digest(source_checkpoint["path"])
    with pytest.raises(RuntimeError, match="injected baseline metric failure"):
        diagnostic.run(arguments(source_checkpoint, output))
    summary = json.loads((output / "diagnostic_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert "injected baseline metric failure" in summary["error"]
    assert summary["before"] is summary["after_latest"] is summary["after_best"] is None
    assert "actual_optimizer_steps" not in summary
    assert digest(source_checkpoint["path"]) == before
    assert not (output / "training.log").exists()


def test_zero_exit_status_without_optimizer_updates_is_reported_as_failed(source_checkpoint, tmp_path, monkeypatch):
    import diagnose_light_overfit as diagnostic

    class NoTrainingProcess:
        def __init__(self, *args, **kwargs):
            self.stdout = iter(["pretend process finished\n"])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def wait(self, **kwargs):
            return 0

    monkeypatch.setattr(diagnostic, "evaluate_checkpoint", lambda *args, **kwargs: {"epoch": 0})
    monkeypatch.setattr(diagnostic.subprocess, "Popen", NoTrainingProcess)
    output = tmp_path / "experiment"
    with pytest.raises(RuntimeError, match="Incomplete training"):
        diagnostic.run(arguments(source_checkpoint, output))
    summary = json.loads((output / "diagnostic_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["before"] == {"epoch": 0}
    assert summary["after_latest"] is summary["after_best"] is None
    assert "actual_optimizer_steps" not in summary


def test_real_cpu_pipeline_evaluates_trains_and_reports_actual_fresh_optimizer_steps(source_checkpoint, tmp_path):
    output = tmp_path / "experiment"
    source_hashes = {path: digest(path) for path in source_checkpoint["dataset"].iterdir()}
    source_hashes[source_checkpoint["path"]] = digest(source_checkpoint["path"])
    core_hashes = {ROOT / name: digest(ROOT / name)
                   for name in ("train_light.py", "train_light_candidate.py", "evaluate_light.py", "EaBNet_light.py")}
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       PYTHONIOENCODING="utf-8")
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(name, None)
    result = subprocess.run([
        sys.executable, str(ROOT / "diagnose_light_overfit.py"),
        "--checkpoint", str(source_checkpoint["path"]), "--train-dir", str(source_checkpoint["dataset"]),
        "--output-dir", str(output), "--num-samples", "1", "--batch-size", "1",
        "--segment-seconds", "1", "--epochs", "1", "--device", "cpu", "--cpu-threads", "2",
    ], cwd=ROOT, env=environment, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    summary = json.loads((output / "diagnostic_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    assert summary["source_epoch"] == 40
    assert summary["steps"] == summary["actual_optimizer_steps"] == 1
    assert summary["before"]["epoch"] == 0
    assert summary["after_latest"]["epoch"] == summary["after_best"]["epoch"] == 1
    for phase in ("before", "after_latest", "after_best"):
        report = summary[phase]
        assert np.isfinite(report["eval_loss"]) and report["eval_loss"] > 0
        for kind in ("enhanced", "noisy"):
            assert set(report["metrics"][kind]) == {"pesq", "stoi", "estoi", "si_snr_db", "sdr_db"}
            assert all(np.isfinite(value) for value in report["metrics"][kind].values())
        evaluation = json.loads(Path(report["summary_json"]).read_text(encoding="utf-8"))
        assert evaluation["completed_samples"] == evaluation["expected_samples"] == 1
        assert evaluation["metric_protocol"]["target_gain_matching"] is False
        assert evaluation["frontend"]["target_ref_mic"] == 2
    for name, value in summary["before"]["metrics"]["noisy"].items():
        assert summary["after_latest"]["metrics"]["noisy"][name] == pytest.approx(value, abs=1e-10)
    initial, _ = read_candidate_checkpoint(output / "initial_checkpoint.pt", CANDIDATE)
    final, _ = read_candidate_checkpoint(output / "checkpoints" / "checkpoint_latest.pt", CANDIDATE)
    assert initial["epoch"] == 0 and initial["optimizer_state_dict"]["state"] == {}
    for name, value in source_checkpoint["checkpoint"]["model_state_dict"].items():
        assert torch.equal(initial["model_state_dict"][name], value), name
    steps = {int(state["step"].item()) for state in final["optimizer_state_dict"]["state"].values()}
    assert steps == {1}
    changed = [name for name, value in initial["model_state_dict"].items()
               if value.is_floating_point() and not torch.equal(value, final["model_state_dict"][name])]
    assert any(name.endswith("weight") for name in changed)
    assert any(name.endswith("running_mean") for name in changed)
    assert final["optimizer_state_dict"]["param_groups"][0]["lr"] == 1e-4
    training_log = (output / "training.log").read_text(encoding="utf-8")
    assert "at epoch 0" in training_log and "Training finished." in training_log
    assert all(digest(path) == value for path, value in {**source_hashes, **core_hashes}.items())

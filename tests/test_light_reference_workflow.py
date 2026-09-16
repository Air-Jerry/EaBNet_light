"""Independent-process checks for reference-derived candidate experiments.

The signals are synthetic plumbing fixtures, not the paper's benchmark data.
"""

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


ROOT = Path(__file__).resolve().parents[1]
CORE_FILES = ("EaBNet_light.py", "train_light.py", "evaluate_light.py")


def core_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in CORE_FILES}


def run_cli(script, arguments, *, expected_success=True):
    environment = os.environ.copy()
    environment.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                        "PYTHONIOENCODING": "utf-8"})
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(name, None)
    result = subprocess.run([sys.executable, str(ROOT / script), *map(str, arguments)], cwd=ROOT,
                            env=environment, text=True, encoding="utf-8", errors="replace",
                            capture_output=True, timeout=120)
    if expected_success:
        assert result.returncode == 0, result.stdout + "\n" + result.stderr
    else:
        assert result.returncode != 0, result.stdout + "\n" + result.stderr
    return result


def write_synthetic_dataset(directory):
    directory.mkdir()
    time = np.arange(32000) / 16000
    envelope = 0.15 * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * time))
    reference = (envelope * (np.sin(2 * np.pi * 143 * time)
                            + 0.4 * np.sin(2 * np.pi * 286 * time)
                            + 0.2 * np.sin(2 * np.pi * 429 * time))).astype(np.float32)
    mixture = reference[:, None] + np.random.default_rng(23).normal(0, 0.01, (32000, 8))
    sf.write(directory / "mixture.wav", mixture, 16000, subtype="FLOAT")
    sf.write(directory / "target.wav", reference, 16000, subtype="FLOAT")
    with (directory / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "mixture_path", "target_path"])
        writer.writerow([1, "mixture.wav", "target.wav"])
    return directory


def training_arguments(dataset, directory):
    return ["--train-dir", dataset, "--val-dir", dataset,
            "--checkpoint-dir", directory / "checkpoints", "--best-dir", directory / "best",
            "--log-dir", directory / "logs", "--batch-size", 1, "--num-workers", 0,
            "--grad-accum-steps", 1, "--use-amp", "no", "--model-amp", "no",
            "--parallel-mode", "none", "--cpu-threads", 2, "--interop-threads", 1,
            "--segment-seconds", 0.2, "--train-random-crop", "no", "--save-every", 1,
            "--mem-log-every-batches", 0, "--malloc-trim-every-batches", 0,
            "--gc-every-batches", 0, "--log-perf", "no"]


def optimizer_steps(checkpoint):
    states = checkpoint["optimizer_state_dict"]["state"].values()
    return {int(state["step"].item()) for state in states if "step" in state}


@pytest.fixture(scope="module")
def completed_candidate_training(tmp_path_factory):
    directory = tmp_path_factory.mktemp("candidate_training")
    dataset = write_synthetic_dataset(directory / "dataset")
    candidate_id = "cbam_flat_projection64"
    arguments = ["--candidate", candidate_id, *training_arguments(dataset, directory)]
    before = core_hashes()
    first_result = run_cli("train_light_candidate.py", [*arguments, "--num-epochs", 1, "--resume", "no"])
    checkpoint_path = directory / "checkpoints" / "checkpoint_latest.pt"
    first = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    second_result = run_cli("train_light_candidate.py", [*arguments, "--num-epochs", 2, "--resume", "yes"])
    second = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert core_hashes() == before
    return dict(directory=directory, dataset=dataset, candidate_id=candidate_id, arguments=arguments,
                checkpoint_path=checkpoint_path, first=first, second=second,
                first_result=first_result, second_result=second_result)


def test_separate_process_train_and_resume_preserve_optimizer_and_candidate_identity(completed_candidate_training):
    run = completed_candidate_training
    first, second = run["first"], run["second"]
    assert first["epoch"] == 1 and second["epoch"] == 2
    assert optimizer_steps(first) == {1}
    assert optimizer_steps(second) == {2}
    assert "Resumed from" in run["second_result"].stdout
    assert "at epoch 1" in run["second_result"].stdout
    for checkpoint in (first, second):
        assert checkpoint["args"]["structure_candidate"] == "cbam_flat_projection64"
        assert checkpoint["args"]["candidate_schema"] == 1
        assert checkpoint["args"]["candidate_evidence"]["author_confirmed"] is False
        assert checkpoint["args"]["candidate_evidence"]["target_equation_11_matches"] is False
        assert math.isfinite(checkpoint["train_loss"]) and checkpoint["train_loss"] > 0
        assert math.isfinite(checkpoint["val_loss"]) and checkpoint["val_loss"] > 0
        assert all(torch.isfinite(value).all() for value in checkpoint["model_state_dict"].values())
    assert any(not torch.equal(first["model_state_dict"][name], value)
               for name, value in second["model_state_dict"].items() if value.is_floating_point())
    assert (run["directory"] / "best" / "best_model.pt").exists()


def test_checkpoint_candidate_is_inferred_for_four_real_metrics(completed_candidate_training):
    run = completed_candidate_training
    output = run["directory"] / "metrics.json"
    csv_output = run["directory"] / "metrics.csv"
    run_cli("evaluate_light_candidate.py", ["--val-dir", run["dataset"],
            "--checkpoint", run["checkpoint_path"], "--device", "cpu",
            "--save-json", output, "--save-csv", csv_output])
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "complete"
    assert report["structure_candidate"] == "cbam_flat_projection64"
    assert report["model_parameters"] == 735070
    assert report["expected_samples"] == report["completed_samples"] == 1
    assert report["exact_reproduction_verified"] is False
    assert report["metric_protocol"]["target_gain_matching"] is False
    assert report["frontend"]["power"] == 0.5
    for signal in ("enhanced", "noisy"):
        metrics = report["mean"][signal]
        assert set(metrics) == {"pesq", "stoi", "estoi", "si_snr_db"}
        assert all(math.isfinite(value) for value in metrics.values())
        assert 0 <= metrics["stoi"] <= 1 and 0 <= metrics["estoi"] <= 1
    with csv_output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1 and rows[0]["num_samples"] == "32000"


@pytest.mark.parametrize("change,error", [
    ("candidate", "Candidate mismatch"),
    ("frontend", "Resume power differs from checkpoint"),
])
def test_resume_rejects_wrong_architecture_or_frontend_before_writing(completed_candidate_training, change, error):
    run = completed_candidate_training
    arguments = list(run["arguments"])
    if change == "candidate":
        arguments[arguments.index("--candidate") + 1] = "literal_per_frequency"
    else:
        arguments.extend(["--power", 1.0])
    digest = hashlib.sha256(run["checkpoint_path"].read_bytes()).hexdigest()
    result = run_cli("train_light_candidate.py", [*arguments, "--num-epochs", 3, "--resume", "yes"],
                     expected_success=False)
    assert error in result.stdout + result.stderr
    assert hashlib.sha256(run["checkpoint_path"].read_bytes()).hexdigest() == digest


def test_explicit_evaluation_candidate_cannot_override_checkpoint_identity(completed_candidate_training, tmp_path):
    run = completed_candidate_training
    result = run_cli("evaluate_light_candidate.py", ["--candidate", "literal_per_frequency",
                     "--val-dir", run["dataset"], "--checkpoint", run["checkpoint_path"],
                     "--device", "cpu", "--save-json", tmp_path / "metrics.json",
                     "--save-csv", tmp_path / "metrics.csv"], expected_success=False)
    assert "Candidate mismatch" in result.stdout + result.stderr
    assert not (tmp_path / "metrics.json").exists()


@pytest.mark.parametrize("fault,error", [
    ("schema", "candidate_schema differs"),
    ("source_hash", "candidate_implementation_sha256 differs"),
    ("missing_identity", "lacks a recognized structure_candidate"),
    ("evidence", "candidate_evidence differs"),
])
def test_saved_metadata_rejects_incompatible_or_unidentified_checkpoints(completed_candidate_training, tmp_path,
                                                                     fault, error):
    from light_candidate_runtime import read_candidate_checkpoint

    checkpoint = deepcopy(completed_candidate_training["first"])
    if fault == "schema":
        checkpoint["args"]["candidate_schema"] = 2
    elif fault == "source_hash":
        checkpoint["args"]["candidate_implementation_sha256"]["EaBNet_light.py"] = "0" * 64
    elif fault == "evidence":
        checkpoint["args"]["candidate_evidence"]["author_confirmed"] = True
    else:
        checkpoint["args"].pop("structure_candidate")
    path = tmp_path / "incompatible.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match=error):
        read_candidate_checkpoint(path)


def test_compatibility_checkpoint_alone_cannot_be_overwritten_by_another_candidate(completed_candidate_training,
                                                                                 tmp_path):
    run = completed_candidate_training
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    compatibility_copy = checkpoint_dir / "best_model.pt"
    torch.save(run["first"], compatibility_copy)
    before = hashlib.sha256(compatibility_copy.read_bytes()).hexdigest()
    assert not (checkpoint_dir / "checkpoint_latest.pt").exists()
    assert not (tmp_path / "best" / "best_model.pt").exists()
    result = run_cli("train_light_candidate.py", ["--candidate", "literal_per_frequency",
                     *training_arguments(run["dataset"], tmp_path), "--num-epochs", 1, "--resume", "no"],
                     expected_success=False)
    assert "Candidate mismatch" in result.stdout + result.stderr
    assert hashlib.sha256(compatibility_copy.read_bytes()).hexdigest() == before
    assert not (checkpoint_dir / "checkpoint_latest.pt").exists()


@pytest.mark.parametrize("missing_field", ["dfsmn_layers", "dfsmn_memory_size", "power", "num_mics"])
def test_evaluation_requires_saved_identity_even_when_tensor_shapes_still_load(completed_candidate_training,
                                                                            tmp_path, missing_field):
    from evaluate_light_candidate import load_candidate_model

    checkpoint = deepcopy(completed_candidate_training["first"])
    checkpoint["args"].pop(missing_field)
    path = tmp_path / "missing_identity_field.pt"
    torch.save(checkpoint, path)
    # No requested identity is supplied here: inference must not silently use
    # old-loader defaults, notably for a shared cell's unencoded repeat depth.
    with pytest.raises(ValueError, match=missing_field):
        load_candidate_model(path, torch.device("cpu"))


def test_candidate_loading_is_strict_and_restores_default_constructor_on_failure(completed_candidate_training,
                                                                               tmp_path):
    import evaluate_light
    from evaluate_light_candidate import load_candidate_model

    checkpoint = deepcopy(completed_candidate_training["first"])
    checkpoint["model_state_dict"].pop(next(iter(checkpoint["model_state_dict"])))
    path = tmp_path / "missing_tensor.pt"
    torch.save(checkpoint, path)
    original_constructor = evaluate_light.EaBNet
    with pytest.raises(RuntimeError, match="Missing key"):
        load_candidate_model(path, torch.device("cpu"))
    assert evaluate_light.EaBNet is original_constructor


def test_module_prefix_state_loads_identically_for_candidate(completed_candidate_training, tmp_path):
    from evaluate_light_candidate import load_candidate_model

    run = completed_candidate_training
    checkpoint = deepcopy(run["second"])
    checkpoint["model_state_dict"] = {"module." + name: value
                                      for name, value in checkpoint["model_state_dict"].items()}
    path = tmp_path / "ddp_prefix.pt"
    torch.save(checkpoint, path)
    prefixed, _ = load_candidate_model(path, torch.device("cpu"))
    plain, _ = load_candidate_model(run["checkpoint_path"], torch.device("cpu"))
    sample = torch.randn(1, 3, 257, 8, 2, generator=torch.Generator().manual_seed(5))
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        with torch.no_grad():
            torch.testing.assert_close(prefixed(sample), plain(sample), atol=0, rtol=0)
    finally:
        torch.set_num_threads(previous_threads)


def test_tcn_constraint_is_independent_of_the_two_other_matching_counts():
    import analyze_light_reference as analysis

    # Original TCM: 57,344 pointwise + 40,960 temporal + 576 BN/PReLU
    # parameters per block. Eighteen blocks total 1,779,840 parameters.
    assert analysis.original_tcn_parameters() == 1779840
    correct = analysis.count_constraints(735070, 731680, 59200)
    assert correct == {"full_match": True, "without_skip_match": True,
                       "tcn_replacement_match": True, "tcn_replacement_parameters": 2455710}
    wrong_memory = analysis.count_constraints(735070, 731680, 9664)
    assert wrong_memory["full_match"] is True and wrong_memory["without_skip_match"] is True
    assert wrong_memory["tcn_replacement_match"] is False
    assert wrong_memory["tcn_replacement_parameters"] == 2505246


@pytest.mark.parametrize("count,accepted", [
    (2454999, False), (2455000, True), (2464999, True), (2465000, False),
])
def test_tcn_half_open_rounding_boundary(count, accepted):
    import analyze_light_reference as analysis

    result = analysis.count_constraints(735000, 725000, 735000 + 1779840 - count)
    assert result["tcn_replacement_parameters"] == count
    assert result["tcn_replacement_match"] is accepted


def test_five_candidate_screen_distinguishes_count_match_from_equation_agreement(tmp_path):
    before = core_hashes()
    result = run_cli("analyze_light_reference.py", ["--output-dir", tmp_path, "--frames", 1,
                     "--steps", 1, "--cpu-threads", 2, "--require-exact"], expected_success=False)
    assert result.returncode == 2
    assert core_hashes() == before
    report = json.loads((tmp_path / "reference_candidates.json").read_text(encoding="utf-8"))
    with (tmp_path / "reference_candidates.csv").open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    expected_counts = {
        "literal_per_frequency": (800674, 758904, 9664, 2570850),
        "literal_flat_projection64": (850210, 808440, 59200, 2570850),
        "cbam_per_frequency": (685534, 682144, 9664, 2455710),
        "cbam_flat_projection64": (735070, 731680, 59200, 2455710),
        "cbam_flat_hidden64": (742238, 738848, 66368, 2455710),
    }
    assert len(csv_rows) == len(report["candidates"]) == len(report["details"]) == 5
    assert report["count_compatible_candidates"] == ["cbam_flat_projection64"]
    assert report["equation_and_count_compatible_candidates"] == []
    assert report["exact_reproduction_verified"] is False
    assert report["default_model_replaced"] is False
    assert report["full_benchmark_training_performed"] is False
    assert report["tcn_protocol"]["actual_tcn_ablation_forward_or_training_performed"] is False
    assert report["core_sources_before"] == report["core_sources_after"] == before
    for csv_row, row, detail in zip(csv_rows, report["candidates"], report["details"]):
        candidate_id = row["candidate_id"]
        assert csv_row["candidate_id"] == detail["candidate_id"] == candidate_id
        observed = tuple(row[name] for name in ("parameters", "without_skip_parameters",
                                                "dfsmn_parameters", "tcn_replacement_parameters"))
        assert observed == expected_counts[candidate_id]
        assert int(csv_row["parameters"]) == row["parameters"]
        assert row["author_confirmed"] is False
        for name in ("full_forward", "without_skip_forward"):
            assert detail[name]["output_shape"] == [1, 2, 1, 257]
            assert detail[name]["dfsmn_calls"] == 3
            assert detail[name]["dfsmn_unique_called_modules"] == 1
            assert detail[name]["finite"] is True
        learning = detail["synthetic_learning"]
        assert learning["steps"] == len(learning["losses_before_updates"]) == 1
        assert learning["all_parameter_gradients_finite"] is True
        assert learning["quality_metric_or_generalization_evidence"] is False
        assert math.isfinite(learning["loss_after_last_update"])

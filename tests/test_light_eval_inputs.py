"""Independent pairing and end-to-end checks for explicit evaluation inputs."""

import csv
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

import evaluate_light as paper
from evaluate_light import load_records
from light_eval_inputs import resolve_evaluation_inputs


def input_args(**changes):
    values = dict(val_dir=None, mixture_path=None, target_path=None,
                  mixture_suffix="", target_suffix="")
    values.update(changes)
    return Namespace(**values)


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"path-only fixture; resolver must not decode audio")
    return path


def metadata(root, rows):
    path = root / "metadata.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "mixture_path", "target_path"])
        writer.writerows(rows)
    return path


def test_metadata_preserves_existing_path_resolution_and_sample_ids(tmp_path):
    mix = touch(tmp_path / "audio" / "different_mix.wav")
    target = touch(tmp_path / "reference" / "different_target.wav")
    manifest = metadata(tmp_path, [(71, "audio/different_mix.wav", "reference/different_target.wav")])
    records, info, protected = resolve_evaluation_inputs(input_args(val_dir=str(tmp_path)))
    assert records == load_records(tmp_path)
    assert records[0].sample_id == 71
    assert manifest.resolve() in set(protected)
    assert isinstance(info, dict)


def test_explicit_files_pair_different_names_and_formats(tmp_path):
    mix = touch(tmp_path / "field_recording.WAV")
    target = touch(tmp_path / "studio_reference.flac")
    records, _, _ = resolve_evaluation_inputs(
        input_args(mixture_path=str(mix), target_path=str(target)))
    assert len(records) == 1
    assert records[0].mixture_path == mix.resolve()
    assert records[0].target_path == target.resolve()


def test_recursive_pairing_keeps_subdirectories_and_has_deterministic_order(tmp_path):
    mix_root, target_root = tmp_path / "mixture", tmp_path / "target"
    expected = {}
    for stem in ("speaker_z/same", "speaker_a/same", "speaker_a/other"):
        mix = touch(mix_root / (stem + ".wav"))
        target = touch(target_root / (stem + ".flac"))
        expected[mix.resolve()] = target.resolve()
    touch(mix_root / "README.txt")
    touch(target_root / "ignored.json")
    args = input_args(mixture_path=str(mix_root), target_path=str(target_root))
    records, _, _ = resolve_evaluation_inputs(args)
    again, _, _ = resolve_evaluation_inputs(args)
    assert records == again
    assert len({row.sample_id for row in records}) == 3
    assert {row.mixture_path: row.target_path for row in records} == expected
    keys = [row.mixture_path.relative_to(mix_root).with_suffix("").as_posix() for row in records]
    assert keys == sorted(keys)


def test_suffix_removal_applies_to_final_stem_without_dropping_nested_identity(tmp_path):
    mix_root, target_root = tmp_path / "mixture", tmp_path / "target"
    expected = {}
    for parent in ("speaker_one", "speaker_two"):
        mix = touch(mix_root / parent / "speech_mix.wav")
        target = touch(target_root / parent / "speech_clean.wav")
        expected[mix.resolve()] = target.resolve()
    records, _, _ = resolve_evaluation_inputs(input_args(
        mixture_path=str(mix_root), target_path=str(target_root),
        mixture_suffix="_mix", target_suffix="_clean"))
    assert {row.mixture_path: row.target_path for row in records} == expected


@pytest.mark.parametrize("extra_side", ["mixture", "target"])
def test_directory_pairing_rejects_partial_intersection(tmp_path, extra_side):
    mix_root, target_root = tmp_path / "mixture", tmp_path / "target"
    touch(mix_root / "paired.wav")
    touch(target_root / "paired.wav")
    touch(tmp_path / extra_side / "unpaired.wav")
    with pytest.raises(ValueError):
        resolve_evaluation_inputs(input_args(mixture_path=str(mix_root), target_path=str(target_root)))


def test_pairing_does_not_silently_match_only_the_basename(tmp_path):
    mix_root, target_root = tmp_path / "mixture", tmp_path / "target"
    touch(mix_root / "speaker_a" / "same.wav")
    touch(target_root / "speaker_b" / "same.wav")
    with pytest.raises(ValueError):
        resolve_evaluation_inputs(input_args(mixture_path=str(mix_root), target_path=str(target_root)))


@pytest.mark.parametrize("duplicate_side", ["mixture", "target"])
def test_different_extensions_with_same_relative_stem_are_ambiguous(tmp_path, duplicate_side):
    mix_root, target_root = tmp_path / "mixture", tmp_path / "target"
    touch(mix_root / "speech.wav")
    touch(target_root / "speech.wav")
    touch(tmp_path / duplicate_side / "speech.flac")
    with pytest.raises(ValueError):
        resolve_evaluation_inputs(input_args(mixture_path=str(mix_root), target_path=str(target_root)))


@pytest.mark.parametrize("mode", ["empty", "files_vs_dir", "missing", "unsupported"])
def test_invalid_explicit_locations_fail_before_evaluation(tmp_path, mode):
    mix, target = tmp_path / "mixture", tmp_path / "target"
    if mode == "empty":
        mix.mkdir()
        target.mkdir()
    elif mode == "files_vs_dir":
        mix = touch(tmp_path / "mixture.wav")
        target.mkdir()
    elif mode == "missing":
        target = touch(tmp_path / "target.wav")
    else:
        mix = touch(tmp_path / "mixture.txt")
        target = touch(tmp_path / "target.txt")
    with pytest.raises((ValueError, FileNotFoundError)):
        resolve_evaluation_inputs(input_args(mixture_path=str(mix), target_path=str(target)))


@pytest.mark.parametrize("options", [
    ["--mixture-path", "mixture"],
    ["--target-path", "target"],
    ["--val-dir", "data", "--mixture-path", "mixture", "--target-path", "target"],
    ["--val-dir", "data", "--target-path", "target"],
    ["--val-dir", "data", "--mixture-suffix", "_mix"],
])
def test_cli_rejects_incomplete_or_conflicting_input_modes(options):
    with pytest.raises(SystemExit) as error:
        paper.parse_args(["--checkpoint", "model.pt", *options])
    assert error.value.code == 2


def test_directory_cli_aliases_use_same_destinations():
    args = paper.parse_args(["--checkpoint", "model.pt", "--mixture-dir", "mixture",
                             "--target-dir", "target"])
    assert args.mixture_path == "mixture"
    assert args.target_path == "target"
    assert args.val_dir is None


def test_no_input_options_keep_original_validation_directory_default():
    args = paper.parse_args(["--checkpoint", "model.pt"])
    assert args.val_dir == "./validation_set"


class ReferencePassThrough(torch.nn.Module):
    M = 8

    def forward(self, features):
        return features[:, :, :, 0, :].permute(0, 3, 1, 2)


def saved_frontend():
    return {**paper.PAPER_FRONTEND, "power": 0.5, "target_ref_mic": 0}


@pytest.fixture
def real_audio_case(tmp_path, monkeypatch):
    t = np.arange(32000) / 16000
    envelope = 0.15 * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    reference = (envelope * (np.sin(2 * np.pi * 143 * t)
                            + 0.4 * np.sin(2 * np.pi * 286 * t)
                            + 0.2 * np.sin(2 * np.pi * 429 * t))).astype(np.float32)
    noisy = reference + np.random.default_rng(19).normal(0, 0.01, len(t)).astype(np.float32)
    mix, target = tmp_path / "mixture" / "one.wav", tmp_path / "target" / "one.wav"
    mix.parent.mkdir()
    target.parent.mkdir()
    sf.write(mix, np.tile(noisy[:, None], (1, 8)), 16000, subtype="FLOAT")
    sf.write(target, reference, 16000, subtype="FLOAT")
    metadata(tmp_path, [(1, "mixture/one.wav", "target/one.wav")])
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"deterministic checkpoint hash fixture")
    monkeypatch.setattr(paper, "load_model", lambda *_, **__: (ReferencePassThrough().eval(), saved_frontend()))
    return tmp_path, mix, target, checkpoint


def evaluation_cli(case, name, inputs):
    root, _, _, checkpoint = case
    return ["--checkpoint", str(checkpoint), "--device", "cpu", *inputs,
            "--estimate-dir", str(root / (name + "_estimates")),
            "--save-csv", str(root / (name + ".csv")),
            "--save-json", str(root / (name + ".json"))]


def test_real_metrics_are_identical_for_metadata_files_and_directories(real_audio_case):
    root, mix, target, _ = real_audio_case
    modes = {
        "metadata": ["--val-dir", str(root)],
        "files": ["--mixture-path", str(mix), "--target-path", str(target)],
        "directories": ["--mixture-path", str(mix.parent), "--target-path", str(target.parent)],
    }
    reports = {}
    for name, inputs in modes.items():
        output_name = "metrics_" + name
        report = paper.evaluate(paper.parse_args(evaluation_cli(real_audio_case, output_name, inputs)))
        reports[name] = report
        assert report["status"] == "complete"
        assert report["expected_samples"] == report["completed_samples"] == 1
        assert report["metric_protocol"]["target_gain_matching"] is False
        assert isinstance(report["input_source"], dict)
        assert len(report["selected_pairs_sha256"]) == 64
        assert json.loads((root / (output_name + ".json")).read_text())["mean"] == report["mean"]
        rows = list(csv.DictReader((root / (output_name + ".csv")).open(encoding="utf-8")))
        assert len(rows) == 1
        assert Path(rows[0]["mixture_path"]) == mix.resolve()
        assert Path(rows[0]["target_path"]) == target.resolve()
        for metric in paper.METRICS:
            assert report["mean"]["enhanced"][metric] == pytest.approx(report["mean"]["noisy"][metric], abs=1e-4)
    assert reports["metadata"]["metadata_sha256"] is not None
    for name in ("files", "directories"):
        assert reports[name]["metadata_sha256"] is None
        assert reports[name]["dataset"] == str(mix.resolve() if name == "files" else mix.parent.resolve())
        for prefix in ("enhanced", "noisy"):
            for metric in paper.METRICS:
                assert reports[name]["mean"][prefix][metric] == pytest.approx(
                    reports["metadata"]["mean"][prefix][metric], abs=1e-8)


def test_unified_entrypoint_accepts_explicit_candidate_and_files(real_audio_case, monkeypatch):
    root, mix, target, _ = real_audio_case
    seen = []
    def candidate_loader(path, device, expected_candidate):
        seen.append(expected_candidate)
        return ReferencePassThrough().eval(), saved_frontend()
    monkeypatch.setattr(paper, "load_model", candidate_loader)
    report = paper.main(["--candidate", "cbam_flat_projection64", *evaluation_cli(
        real_audio_case, "candidate", ["--mixture-path", str(mix), "--target-path", str(target)])])
    assert report["status"] == "complete"
    assert seen == ["cbam_flat_projection64"]


@pytest.mark.parametrize("destination", ["mixture", "target", "checkpoint", "unselected_mixture", "unselected_target"])
@pytest.mark.parametrize("output_option", ["--save-csv", "--save-json"])
def test_output_paths_cannot_overwrite_any_input_even_after_sample_limit(
        real_audio_case, destination, output_option):
    root, mix, target, checkpoint = real_audio_case
    second_mix = touch(mix.parent / "two.wav")
    second_target = touch(target.parent / "two.wav")
    destinations = {"mixture": mix, "target": target, "checkpoint": checkpoint,
                    "unselected_mixture": second_mix, "unselected_target": second_target}
    protected = destinations[destination]
    before = protected.read_bytes()
    options = evaluation_cli(real_audio_case, "protected", [
        "--mixture-path", str(mix.parent), "--target-path", str(target.parent), "--max-samples", "1"])
    options[options.index(output_option) + 1] = str(protected)
    with pytest.raises(ValueError, match="overwrite|distinct"):
        paper.evaluate(paper.parse_args(options))
    assert protected.read_bytes() == before
    assert not (root / "protected.csv").exists()
    assert not (root / "protected.json").exists()


def test_estimate_metadata_cannot_overwrite_input_manifest(real_audio_case):
    root, _, _, _ = real_audio_case
    before = (root / "metadata.csv").read_bytes()
    options = evaluation_cli(real_audio_case, "collision", ["--val-dir", str(root)])
    options[options.index("--estimate-dir") + 1] = str(root)
    with pytest.raises(ValueError, match="overwrite|distinct"):
        paper.evaluate(paper.parse_args(options))
    assert (root / "metadata.csv").read_bytes() == before
    assert not (root / "collision.json").exists()


@pytest.mark.parametrize("saved_role", ["mixture", "estimate", "target"])
def test_export_cannot_overwrite_an_unselected_input_audio(real_audio_case, saved_role):
    root, mix, target, _ = real_audio_case
    output_dir = root / "dangerous_output"
    protected = touch(output_dir / saved_role / f"sample_00000001_{saved_role}.wav")
    metadata(root, [(1, str(mix), str(target)), (2, str(protected), str(target))])
    original = protected.read_bytes()
    options = evaluation_cli(real_audio_case, "collision", ["--val-dir", str(root), "--max-samples", "1"])
    options[options.index("--estimate-dir") + 1] = str(output_dir)
    with pytest.raises(ValueError, match="overwrite|distinct"):
        paper.evaluate(paper.parse_args(options))
    assert protected.read_bytes() == original
    assert not (root / "collision.json").exists()


@pytest.mark.parametrize("option", ["--save-csv", "--save-json"])
def test_report_cannot_be_written_below_an_input_file(real_audio_case, option):
    root, mix, target, _ = real_audio_case
    original = mix.read_bytes()
    options = evaluation_cli(real_audio_case, "nested", [
        "--mixture-path", str(mix), "--target-path", str(target)])
    options[options.index(option) + 1] = str(mix / "result.txt")
    with pytest.raises(ValueError, match="overwrite|distinct|input|ancestor|file"):
        paper.evaluate(paper.parse_args(options))
    assert mix.read_bytes() == original
    assert not (root / "nested_estimates" / "metadata.csv").exists()


def test_max_samples_does_not_decode_unselected_audio(real_audio_case):
    root, mix, target, _ = real_audio_case
    second_mix = touch(mix.parent / "two.wav")
    second_target = touch(target.parent / "two.wav")
    report = paper.evaluate(paper.parse_args(evaluation_cli(real_audio_case, "first", [
        "--mixture-path", str(mix.parent), "--target-path", str(target.parent), "--max-samples", "1"])))
    assert report["status"] == "complete"
    assert report["expected_samples"] == report["completed_samples"] == 1
    assert second_mix.read_bytes() == second_target.read_bytes() == b"path-only fixture; resolver must not decode audio"

import csv
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

import evaluate_light as paper
import train_light


def frontend_args(**changes):
    values = dict(sample_rate=None, n_fft=None, hop_length=None, win_length=None, power=None, target_ref_mic=None)
    values.update(changes)
    return Namespace(**values)


def saved_frontend():
    return {**paper.PAPER_FRONTEND, "power": 0.5, "target_ref_mic": 0}


def test_frontend_requires_checkpoint_agreement_and_missing_value_acknowledgement():
    assert paper.resolve_frontend(frontend_args(), saved_frontend()) == saved_frontend()
    with pytest.raises(ValueError, match="differs from checkpoint"):
        paper.resolve_frontend(frontend_args(power=1.0), saved_frontend())
    missing = saved_frontend()
    missing.pop("power")
    with pytest.raises(ValueError, match="provide --power explicitly"):
        paper.resolve_frontend(frontend_args(), missing)
    assert paper.resolve_frontend(frontend_args(power=0.5), missing)["power"] == 0.5
    with pytest.raises(ValueError, match="requires n_fft=512"):
        paper.resolve_frontend(frontend_args(), {**saved_frontend(), "n_fft": 1024})


def test_frontend_uses_checkpoint_window_hop_and_power_without_default_overrides():
    saved = {**saved_frontend(), "hop_length": 80, "win_length": 256, "power": 1.0, "target_ref_mic": 2}
    assert paper.resolve_frontend(frontend_args(), saved) == saved
    with pytest.raises(ValueError, match="differs from checkpoint"):
        paper.resolve_frontend(frontend_args(hop_length=160), saved)


@pytest.mark.parametrize("reference_scale", [1e-3, 1e3])
def test_optional_gain_matching_respects_symmetric_db_bound(reference_scale):
    estimate = synthetic_speech().astype(np.float64)
    matched, gain = paper.match_estimate_level(estimate * reference_scale, estimate, max_gain_db=6.0)
    expected = 10 ** ((-6.0 if reference_scale < 1 else 6.0) / 20)
    assert gain == pytest.approx(expected)
    np.testing.assert_array_equal(matched, estimate * gain)


def test_si_snr_matches_analytic_projection_and_is_scale_offset_invariant():
    rng = np.random.default_rng(41)
    reference, noise = rng.normal(size=(2, 16000))
    reference -= reference.mean()
    noise -= noise.mean()
    noise -= np.dot(noise, reference) / np.dot(reference, reference) * reference
    estimate = 0.7 * reference + noise
    expected = 10 * np.log10((0.49 * np.dot(reference, reference) + paper.SI_SNR_EPSILON)
                            / (np.dot(noise, noise) + paper.SI_SNR_EPSILON))
    assert paper.compute_si_snr_db(reference, estimate) == pytest.approx(expected, abs=1e-12)
    assert paper.compute_si_snr_db(reference + 3, estimate * 2 - 4) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("bad", [np.zeros(1600), np.full(1600, np.nan), np.full(1600, np.inf)])
def test_bad_reference_and_estimate_fail(bad):
    signal = np.sin(np.arange(1600) * 0.1)
    with pytest.raises(ValueError):
        paper.compute_paper_metrics(16000, bad, signal)
    with pytest.raises(ValueError):
        paper.compute_paper_metrics(16000, signal, bad)


def synthetic_speech():
    t = np.arange(32000) / 16000
    envelope = 0.15 * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    return (envelope * (np.sin(2 * np.pi * 143 * t) + 0.4 * np.sin(2 * np.pi * 286 * t)
                        + 0.2 * np.sin(2 * np.pi * 429 * t))).astype(np.float32)


def test_real_metrics_return_five_finite_values_and_fraction_intelligibility():
    reference = synthetic_speech()
    estimate = reference + np.random.default_rng(3).normal(0, 0.01, len(reference))
    result = paper.compute_paper_metrics(16000, reference, estimate)
    assert set(result) == set(paper.METRICS)
    assert set(result) == {"pesq", "stoi", "estoi", "si_snr_db", "sdr_db"}
    assert all(np.isfinite(value) for value in result.values())
    assert 0 <= result["stoi"] <= 1
    assert 0 <= result["estoi"] <= 1


class ReferencePassThrough(torch.nn.Module):
    M = 8

    def forward(self, features):
        return features[:, :, :, 0, :].permute(0, 3, 1, 2)


@pytest.mark.parametrize("silent", [False, True])
def test_complete_evaluator_reports_raw_paired_metrics_or_explicit_failure(tmp_path, monkeypatch, silent):
    reference = synthetic_speech()
    noisy = reference + np.random.default_rng(13).normal(0, 0.01, len(reference)).astype(np.float32)
    if silent:
        reference[:] = 0
    sf.write(tmp_path / "mix.wav", np.tile(noisy[:, None], (1, 8)), 16000, subtype="FLOAT")
    sf.write(tmp_path / "target.wav", reference, 16000, subtype="FLOAT")
    with (tmp_path / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "mixture_path", "target_path"])
        writer.writerow([1, "mix.wav", "target.wav"])
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic checkpoint placeholder for hash test")
    monkeypatch.setattr(paper, "load_model", lambda *_, **__: (ReferencePassThrough().eval(), saved_frontend()))
    args = paper.parse_args(["--val-dir", str(tmp_path), "--checkpoint", str(checkpoint), "--device", "cpu",
                             "--estimate-dir", str(tmp_path / "estimates"),
                             "--save-csv", str(tmp_path / "metrics.csv"), "--save-json", str(tmp_path / "metrics.json")])
    if silent:
        with pytest.raises(ValueError, match="silent signal"):
            paper.evaluate(args)
        report = json.loads((tmp_path / "metrics.json").read_text())
        assert report["status"] == "failed" and report["mean"] is None
        assert report["failed_sample_id"] == 1 and report["completed_samples"] == 0
    else:
        report = paper.evaluate(args)
        assert report["status"] == "complete"
        assert report["expected_samples"] == report["completed_samples"] == 1
        assert report["metric_protocol"]["target_gain_matching"] is False
        for metric in paper.METRICS:
            assert report["mean"]["enhanced"][metric] == pytest.approx(report["mean"]["noisy"][metric], abs=1e-4)
        rows = list(csv.DictReader((tmp_path / "metrics.csv").open()))
        assert len(rows) == 1 and rows[0]["sample_id"] == "1"


@pytest.fixture
def waveform_case(tmp_path, monkeypatch):
    """Amplified audio makes any hidden peak normalization or PCM clipping observable."""
    reference = synthetic_speech() * 14
    noisy = reference + np.random.default_rng(61).normal(0, 0.01, reference.size).astype(np.float32)
    mixture = np.stack([noisy * (index + 1) / 8 for index in range(10)], axis=1)
    targets = np.stack([reference * 0.4, reference * 0.7, reference], axis=1)
    mixture_path, target_path = tmp_path / "mix.wav", tmp_path / "target.wav"
    sf.write(mixture_path, mixture, 16000, subtype="FLOAT")
    sf.write(target_path, targets, 16000, subtype="FLOAT")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic checkpoint placeholder for output test")
    observed = {"features": [], "metrics": []}

    class CaptureModel(ReferencePassThrough):
        def forward(self, features):
            observed["features"].append(features.detach().clone())
            return features[:, :, :, 7, :].permute(0, 3, 1, 2)

    def capture_metrics(sample_rate, ref, estimate):
        assert sample_rate == 16000
        observed["metrics"].append((np.array(ref), np.array(estimate)))
        return {metric: 0.5 for metric in paper.METRICS}

    monkeypatch.setattr(paper, "load_model", lambda *_, **__: (
        CaptureModel().eval(), {**saved_frontend(), "target_ref_mic": 2}))
    monkeypatch.setattr(paper, "compute_paper_metrics", capture_metrics)
    options = ["--checkpoint", str(checkpoint), "--mixture-path", str(mixture_path),
               "--target-path", str(target_path), "--device", "cpu",
               "--mixture-ref-mic", "5", "--estimate-dir", str(tmp_path / "out")]
    return tmp_path, mixture, targets, observed, options


def test_audio_export_preserves_amplitude_channels_and_training_features(waveform_case):
    root, mixture, targets, observed, options = waveform_case
    report = paper.evaluate(paper.parse_args(options))
    assert report["status"] == "complete"
    kwargs = dict(n_fft=512, hop_length=160, win_length=320, power=0.5, device=torch.device("cpu"))
    expected, _ = train_light.build_stft_batch(
        torch.from_numpy(mixture[:, :8]).unsqueeze(0), torch.from_numpy(targets[:, 2]).unsqueeze(0), **kwargs)
    assert len(observed["features"]) == 1
    torch.testing.assert_close(observed["features"][0], expected, rtol=0, atol=0)
    row, = list(csv.DictReader((root / "out" / "metadata.csv").open(encoding="utf-8")))
    assert float(row["estimate_level_gain"]) == 1
    assert float(row["estoi_pct"]) == 50
    assert float(row["estoi_mix_pct"]) == 50
    for field, expected_audio, tolerance in (
            ("mixture_saved_path", mixture[:, 5], 0),
            ("target_saved_path", targets[:, 2], 0),
            ("estimate_saved_path", mixture[:, 7], 2e-6)):
        path = Path(row[field])
        audio, sample_rate = sf.read(path, dtype="float32")
        assert sf.info(path).subtype == "FLOAT"
        assert sample_rate == 16000 and audio.shape == (32000,)
        assert np.max(np.abs(audio)) > 1
        np.testing.assert_allclose(audio, expected_audio, atol=tolerance, rtol=1e-6 if tolerance else 0)
    assert len(observed["metrics"]) == 2
    np.testing.assert_array_equal(observed["metrics"][0][0], targets[:, 2])
    np.testing.assert_allclose(observed["metrics"][0][1], mixture[:, 7], atol=2e-6, rtol=1e-6)
    np.testing.assert_array_equal(observed["metrics"][1][1], mixture[:, 5])
    summary = json.loads((root / "out" / "summary.json").read_text(encoding="utf-8"))
    assert summary["metric_protocol"]["target_gain_matching"] is False


def test_optional_level_matching_changes_only_estimate_and_is_reported(waveform_case, monkeypatch):
    root, mixture, targets, observed, options = waveform_case

    class QuietModel(ReferencePassThrough):
        def forward(self, features):
            # power=0.5; halving compressed magnitude quarters waveform level.
            return features[:, :, :, 7, :].permute(0, 3, 1, 2) * 0.5

    monkeypatch.setattr(paper, "load_model", lambda *_, **__: (
        QuietModel().eval(), {**saved_frontend(), "target_ref_mic": 2}))
    report = paper.evaluate(paper.parse_args([*options, "--match-estimate-level", "yes"]))
    assert report["metric_protocol"]["target_gain_matching"] is True
    row, = list(csv.DictReader((root / "out" / "metadata.csv").open(encoding="utf-8")))
    gain = float(row["estimate_level_gain"])
    raw_estimate = mixture[:, 7].astype(np.float64) * 0.25
    expected_gain = np.dot(targets[:, 2].astype(np.float64), raw_estimate) / (np.dot(raw_estimate, raw_estimate) + paper.EPS)
    assert gain == pytest.approx(expected_gain, rel=2e-6)
    np.testing.assert_allclose(observed["metrics"][0][1], raw_estimate * gain, atol=3e-6, rtol=2e-6)
    np.testing.assert_array_equal(observed["metrics"][1][1], mixture[:, 5])
    saved_estimate, _ = sf.read(row["estimate_saved_path"], dtype="float32")
    np.testing.assert_allclose(saved_estimate, observed["metrics"][0][1], atol=2e-7, rtol=1e-7)


def test_disabling_wav_export_still_writes_complete_metadata_and_summary(waveform_case):
    root, _, _, _, options = waveform_case
    report = paper.evaluate(paper.parse_args([*options, "--save-samples", "no"]))
    assert report["status"] == "complete"
    assert not list((root / "out").rglob("*.wav"))
    row, = list(csv.DictReader((root / "out" / "metadata.csv").open(encoding="utf-8")))
    for field in ("mixture_saved_path", "estimate_saved_path", "target_saved_path"):
        assert row[field] == ""
    assert (root / "out" / "summary.json").exists()


@pytest.mark.parametrize("change,value,error", [
    ("mixture_rate", 8000, "sample rate"),
    ("target_rate", 8000, "sample rate"),
    ("channels", 7, "channels"),
])
def test_wrong_audio_rate_or_microphone_count_fails_without_metrics(waveform_case, change, value, error):
    root, mixture, targets, observed, options = waveform_case
    sf.write(root / "mix.wav", mixture[:, :value] if change == "channels" else mixture,
             value if change == "mixture_rate" else 16000, subtype="FLOAT")
    sf.write(root / "target.wav", targets, value if change == "target_rate" else 16000, subtype="FLOAT")
    with pytest.raises(ValueError, match=error):
        paper.evaluate(paper.parse_args(options))
    assert observed["features"] == observed["metrics"] == []
    report = json.loads((root / "out" / "summary.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed" and report["mean"] is None


def test_saved_frontend_override_fails_before_model_inference_or_output(waveform_case):
    root, _, _, observed, options = waveform_case
    with pytest.raises(ValueError, match="differs from checkpoint"):
        paper.evaluate(paper.parse_args([*options, "--power", "1"]))
    assert observed["features"] == observed["metrics"] == []
    assert not (root / "out" / "metadata.csv").exists()

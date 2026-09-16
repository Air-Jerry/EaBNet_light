import csv
import json
from argparse import Namespace

import numpy as np
import pytest
import soundfile as sf
import torch

import evaluate_light_paper as paper


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


def test_real_metrics_return_four_finite_values_and_fraction_intelligibility():
    reference = synthetic_speech()
    estimate = reference + np.random.default_rng(3).normal(0, 0.01, len(reference))
    result = paper.compute_paper_metrics(16000, reference, estimate)
    assert set(result) == set(paper.METRICS)
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
    monkeypatch.setattr(paper, "load_model", lambda *_: (ReferencePassThrough().eval(), saved_frontend()))
    args = paper.parse_args(["--val-dir", str(tmp_path), "--checkpoint", str(checkpoint), "--device", "cpu",
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

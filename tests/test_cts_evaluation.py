"""Guard evaluation comparability, channel selection, and failure accounting."""

import csv
import json
import warnings
from dataclasses import asdict

import numpy as np
import pytest
import soundfile as sf
import torch

import evaluate_cts as evaluation
from evaluate_light import SampleRecord


@pytest.fixture
def frontend():
    return evaluation.FrontendConfig(16000, 0, 1, 320, 160, 320, 1.0)


def test_frontend_requires_checkpoint_metadata_and_rejects_512(frontend):
    args = asdict(frontend)
    assert evaluation.FrontendConfig.from_checkpoint(args) == frontend
    with pytest.raises(ValueError, match="refusing to guess"):
        evaluation.FrontendConfig.from_checkpoint({key: value for key, value in args.items() if key != "power"})
    with pytest.raises(ValueError, match="n_fft=320"):
        evaluation.FrontendConfig.from_checkpoint({**args, "n_fft": 512})
    with pytest.raises(ValueError, match="disagrees with checkpoint"):
        evaluation.validate_overrides(evaluation.parse_args(["--power", "0.5"]), frontend)


def test_wsj0_uses_narrowband_and_extended_stoi(monkeypatch):
    calls = []
    monkeypatch.setattr(evaluation, "pesq", lambda sr, ref, deg, mode: calls.append(("pesq", sr, mode)) or 2.5)
    monkeypatch.setattr(evaluation, "stoi", lambda ref, deg, sr, extended: calls.append(("stoi", extended)) or 0.7)
    monkeypatch.setattr(evaluation, "compute_sdr_db", lambda ref, deg: 4.0)
    audio = np.arange(1, 100, dtype=np.float64)
    scores = evaluation.compute_metrics(16000, audio, audio * 0.7)
    assert calls == [("pesq", 16000, "nb"), ("stoi", True)]
    assert scores == {"pesq": 2.5, "estoi_pct": 70.0, "sdr_db": 4.0}


@pytest.mark.parametrize("failure", ["exception", "nan", "warning"])
def test_metric_failures_do_not_disappear_from_averages(monkeypatch, failure):
    def broken(*args):
        if failure == "exception":
            raise ValueError("insufficient speech")
        if failure == "warning":
            warnings.warn("Not enough STFT frames", RuntimeWarning)
        return float("nan")

    monkeypatch.setattr(evaluation, "pesq", broken)
    audio = np.arange(1, 100, dtype=np.float64)
    with pytest.raises(RuntimeError, match="pesq failed"):
        evaluation.compute_metrics(16000, audio, audio)


def test_dns_reports_both_pesq_modes_and_stoi(monkeypatch):
    calls = []
    monkeypatch.setattr(evaluation, "pesq", lambda sr, ref, deg, mode: calls.append(mode) or 2.5)
    monkeypatch.setattr(evaluation, "stoi", lambda ref, deg, sr, extended: calls.append(extended) or 0.7)
    audio = np.random.default_rng(9).normal(size=1600)
    deg = audio + np.random.default_rng(10).normal(scale=0.1, size=audio.shape)
    scores = evaluation.compute_metrics(16000, audio, deg, protocol="dns")
    assert calls == ["wb", "nb", False]
    assert set(scores) == {"pesq_wb", "pesq_nb", "stoi_pct", "si_sdr_db"}
    assert scores["si_sdr_db"] == pytest.approx(evaluation.compute_si_sdr_db(audio, deg * 20), abs=1e-10)


def test_only_mixture_first_channel_is_used_and_lengths_are_not_silently_cropped(tmp_path, frontend):
    first = np.linspace(-0.3, 0.3, 640, dtype=np.float32)
    second = np.ones_like(first)
    mix_path, target_path = tmp_path / "mixture.wav", tmp_path / "target.wav"
    sf.write(mix_path, np.column_stack((first, second)), 16000, subtype="FLOAT")
    sf.write(target_path, first, 16000, subtype="FLOAT")
    record = SampleRecord(17, mix_path, target_path)
    mixture, target = evaluation.read_waveforms(record, frontend)
    np.testing.assert_array_equal(mixture[:, 0], first)
    np.testing.assert_array_equal(target, first)
    assert mixture.shape == (640, 1)
    sf.write(target_path, first[:-1], 16000, subtype="FLOAT")
    with pytest.raises(ValueError, match="lengths must agree"):
        evaluation.read_waveforms(record, frontend)


def test_load_model_rejects_me_checkpoint(tmp_path, frontend):
    path = tmp_path / "me.pt"
    torch.save({"model_name": "CTSNet", "stage": "me", "args": asdict(frontend)}, path)
    with pytest.raises(ValueError, match="joint-stage checkpoint"):
        evaluation.load_model(path, torch.device("cpu"))


def test_checkpoint_source_change_is_rejected_before_loading_weights(tmp_path, frontend):
    path = tmp_path / "changed_code.pt"
    torch.save({
        "model_name": "CTSNet", "stage": "joint",
        "args": {**asdict(frontend), "norm_type": "IN", "is_causal": "yes"},
        "reproducibility_manifest": {"code_sha256": {"CTSNet.py": "different"}},
    }, path)
    with pytest.raises(ValueError, match="differs from the checkpoint source manifest"):
        evaluation.load_model(path, torch.device("cpu"))


def test_reference_never_rescales_estimate_and_failed_run_is_marked(tmp_path, frontend, monkeypatch):
    class Identity(torch.nn.Module):
        def forward(self, x):
            return x[..., 0, :].permute(0, 3, 1, 2)

    monkeypatch.setattr(evaluation, "load_model", lambda *args: (Identity(), frontend, {}))
    wave = np.sin(np.arange(1600) * 0.1).astype(np.float32) * 0.05
    sf.write(tmp_path / "mix.wav", np.column_stack((wave, wave * 7)), 16000, subtype="FLOAT")
    sf.write(tmp_path / "target.wav", wave * 4, 16000, subtype="FLOAT")
    with (tmp_path / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "mixture_path", "target_path"])
        writer.writerow([9, "mix.wav", "target.wav"])
    compared = []

    def scores(sr, ref, deg, *args):
        compared.append((ref.copy(), deg.copy()))
        return {"pesq": 2.0, "estoi_pct": 70.0, "sdr_db": 8.0}

    monkeypatch.setattr(evaluation, "compute_metrics", scores)
    output = tmp_path / "output"
    command = ["--val-dir", str(tmp_path), "--estimate-dir", str(output), "--device", "cpu"]
    evaluation.main(command)
    np.testing.assert_allclose(compared[0][1], wave, atol=1e-7)
    np.testing.assert_array_equal(compared[0][0], wave * 4)
    saved_estimate, _ = sf.read(output / "estimate" / "sample_00000009_estimate.wav")
    np.testing.assert_allclose(saved_estimate, wave, atol=1e-7)
    status = json.loads((output / "evaluation_status.json").read_text(encoding="utf-8"))
    assert status["completed"] and status["evaluated_samples"] == 1
    assert status["reference_level_matching"] is False
    assert status["pesq_mode"] == "nb"

    def fail(*args):
        raise RuntimeError("PESQ error")

    monkeypatch.setattr(evaluation, "compute_metrics", fail)
    with pytest.raises(RuntimeError, match="sample_id=9, enhanced"):
        evaluation.main(command)
    status = json.loads((output / "evaluation_status.json").read_text(encoding="utf-8"))
    assert not status["completed"] and status["status"] == "failed"
    assert status["evaluated_samples"] == 0 and status["metrics"] == {}


def test_real_checkpoint_network_audio_and_metrics_pipeline(tmp_path, frontend):
    """A random model checks plumbing only; these scores are not paper results."""
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        torch.manual_seed(810)
        model = evaluation.CTSNet(is_causal=True, norm_type="IN")
        checkpoint_path = tmp_path / "model.pt"
        torch.save({
            "model_name": "CTSNet", "stage": "joint", "epoch": 1, "stage_epoch": 1,
            "args": {**asdict(frontend), "norm_type": "IN", "is_causal": "yes"},
            "model_state_dict": model.state_dict(),
        }, checkpoint_path)
        sr = 16000
        time = np.arange(sr * 3) / sr
        phase = 2 * np.pi * (125 * time + 7 * time**2)
        envelope = 0.1 + 0.9 * np.sin(np.pi * 2 * time)**2
        target = 0.1 * envelope * sum(np.sin(k * phase) / k for k in range(1, 15))
        mixture = target + np.random.default_rng(32).normal(scale=0.02, size=target.shape)
        sf.write(tmp_path / "mixture.wav", mixture, sr, subtype="FLOAT")
        sf.write(tmp_path / "target.wav", target, sr, subtype="FLOAT")
        (tmp_path / "metadata.csv").write_text(
            "sample_id,mixture_path,target_path\n1,mixture.wav,target.wav\n", encoding="utf-8"
        )
        output = tmp_path / "evaluation"
        evaluation.main([
            "--val-dir", str(tmp_path), "--checkpoint", str(checkpoint_path),
            "--estimate-dir", str(output), "--save-samples", "no", "--device", "cpu",
        ])
        result = json.loads((output / "evaluation_status.json").read_text(encoding="utf-8"))
        assert result["completed"] and result["evaluated_samples"] == 1
        assert len(result["metrics"]) == 6
        assert all(np.isfinite(value) for value in result["metrics"].values())
        assert not (output / "estimate").exists()
    finally:
        torch.set_num_threads(previous_threads)

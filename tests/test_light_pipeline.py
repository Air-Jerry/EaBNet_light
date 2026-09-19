"""Small synthetic checks of the existing, unchanged light training interface."""

import csv
import math

import numpy as np
import pytest
import soundfile as sf
import torch

import evaluate_light
import train_light


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_dataset(directory, lengths=(1600, 1920), num_mics=8):
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(31)
    rows = []
    for sample_id, length in enumerate(lengths):
        target = rng.normal(0, 0.06, (length, 2)).astype(np.float32)
        mixture = target[:, :1] + rng.normal(0, 0.03, (length, num_mics)).astype(np.float32)
        mixture_path = directory / f"mixture_{sample_id}.wav"
        target_path = directory / f"target_{sample_id}.wav"
        sf.write(mixture_path, mixture, 16000, subtype="FLOAT")
        sf.write(target_path, target, 16000, subtype="FLOAT")
        rows.append({"sample_id": sample_id, "mixture_path": mixture_path.name, "target_path": target_path.name})
    with (directory / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "mixture_path", "target_path"))
        writer.writeheader()
        writer.writerows(rows)
    return directory


def test_metadata_loading_channel_selection_crop_and_collation(tmp_path):
    directory = make_dataset(tmp_path / "validation_set")
    dataset = train_light.EnhancementDataset(directory, 16000, target_ref_mic=1, num_mics=6)
    records = evaluate_light.load_records(directory)
    assert [record.sample_id for record in records] == [0, 1]
    item = dataset[0]
    raw_target, _ = sf.read(directory / "target_0.wav", dtype="float32", always_2d=True)
    torch.testing.assert_close(item["target"], torch.from_numpy(raw_target[:, 1]))
    assert item["mixture"].shape == (1600, 6)
    batch = train_light.collate_batch([dataset[0], dataset[1]])
    assert batch["mixture"].shape == (2, 1920, 6)
    assert batch["lengths"].tolist() == [1600, 1920]
    assert batch["sample_ids"].tolist() == [0, 1]
    assert not batch["mixture"][0, 1600:].count_nonzero()
    assert not batch["target"][0, 1600:].count_nonzero()
    cropped = train_light.EnhancementDataset(directory, 16000, target_ref_mic=1, num_mics=6,
                                            segment_samples=800, random_crop=False)
    assert cropped[0]["num_samples"] == 800
    torch.testing.assert_close(cropped[0]["target"], item["target"][:800])


@pytest.mark.parametrize("power", [0.5, 1.0])
def test_train_eval_stft_agree_and_reference_channel_roundtrips(power):
    generator = torch.Generator().manual_seed(33)
    mixture = torch.randn(2, 1923, 8, generator=generator) * 0.1
    target = mixture[:, :, 3]
    kwargs = dict(n_fft=512, hop_length=160, win_length=320, power=power, device=torch.device("cpu"))
    features, target_stft = train_light.build_stft_batch(mixture, target, **kwargs)
    evaluation_features = evaluate_light.build_stft_batch(mixture, **kwargs)
    torch.testing.assert_close(features, evaluation_features, atol=0, rtol=0)
    torch.testing.assert_close(features[:, :, :, 3, :].permute(0, 3, 1, 2), target_stft)
    recovered = evaluate_light.reconstruct_waveform(target_stft, length=target.shape[1], **kwargs)
    torch.testing.assert_close(recovered, target, atol=2e-6, rtol=2e-5)
    assert train_light.waveform_lengths_to_frames(torch.tensor([1923]), 160) == [features.shape[1]]


@pytest.mark.parametrize("power", [0.5, 1.0])
def test_evaluation_matches_training_compression_below_epsilon(power):
    mixture = torch.randn(1, 1600, 8, generator=torch.Generator().manual_seed(53)) * 1e-11
    target = mixture[:, :, 0]
    kwargs = dict(n_fft=512, hop_length=160, win_length=320, power=power, device=torch.device("cpu"))
    training_features, _ = train_light.build_stft_batch(mixture, target, **kwargs)
    evaluation_features = evaluate_light.build_stft_batch(mixture, **kwargs)
    assert training_features.count_nonzero() > 0
    torch.testing.assert_close(evaluation_features, training_features, rtol=0, atol=0)
    recovered = evaluate_light.reconstruct_waveform(
        evaluation_features[:, :, :, 0, :].permute(0, 3, 1, 2), length=target.shape[1], **kwargs)
    torch.testing.assert_close(recovered, target, rtol=2e-5, atol=1e-17)


def test_zero_waveform_has_finite_zero_stft():
    features, target = train_light.build_stft_batch(torch.zeros(1, 1600, 8), torch.zeros(1, 1600),
        n_fft=512, hop_length=160, win_length=320, power=0.5, device=torch.device("cpu"))
    assert torch.isfinite(features).all() and torch.isfinite(target).all()
    assert features.count_nonzero() == target.count_nonzero() == 0


def test_real_model_training_validation_and_checkpoint_roundtrip(tmp_path):
    directory = make_dataset(tmp_path / "validation_set", lengths=(1600, 1600))
    dataset = train_light.EnhancementDataset(directory, 16000)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=train_light.collate_batch)
    args = train_light.parse_args([])
    args.channels, args.embed_dim, args.cd1 = 8, 8, 8
    args.use_amp = args.model_amp = "no"
    args.grad_accum_steps = 2  # Complete window: do not mask the known final-window limitation.
    args.mem_log_every_batches = args.malloc_trim_every_batches = args.gc_every_batches = 0
    args.log_perf = "no"
    device = torch.device("cpu")
    torch.manual_seed(7)
    model = train_light.create_model(args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    before = {key: value.detach().clone() for key, value in model.named_parameters()}
    loss = train_light.run_epoch(model, loader, optimizer, scaler, device, [], args, training=True)
    assert math.isfinite(loss) and loss > 0
    assert any(not torch.equal(before[key], value) for key, value in model.named_parameters())
    assert all(torch.isfinite(value).all() for value in model.parameters())
    trained = {key: value.detach().clone() for key, value in model.named_parameters()}
    val_loss = train_light.run_epoch(model, loader, optimizer, scaler, device, [], args, training=False)
    assert math.isfinite(val_loss) and val_loss > 0
    assert all(torch.equal(trained[key], value) for key, value in model.named_parameters())
    checkpoint = tmp_path / "model.pt"
    train_light.save_checkpoint(checkpoint, {
        "model_state_dict": {"module." + key: value for key, value in model.state_dict().items()},
        "args": vars(args),
    })
    loaded, saved_args = evaluate_light.load_model(checkpoint, device)
    assert saved_args["power"] == args.power
    batch = next(iter(loader))
    features, _ = train_light.build_stft_batch(batch["mixture"], batch["target"], n_fft=512,
        hop_length=160, win_length=320, power=0.5, device=device)
    with torch.no_grad():
        torch.testing.assert_close(loaded(features), model(features), rtol=0, atol=0)


def test_checkpoint_loading_rejects_incomplete_state():
    args = train_light.parse_args([])
    args.channels, args.embed_dim, args.cd1 = 8, 8, 8
    model = train_light.create_model(args, torch.device("cpu"))
    incomplete = dict(model.state_dict())
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises(RuntimeError, match="Missing key"):
        train_light.load_model_state_flexible(model, incomplete)


@pytest.mark.xfail(strict=True, reason="Existing run_epoch divides an incomplete final accumulation window by grad_accum_steps")
def test_existing_final_accumulation_window_should_use_actual_batch_count():
    class ScalarModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, features):
            return self.weight

    args = train_light.parse_args([])
    args.use_amp = args.model_amp = "no"
    args.grad_accum_steps = 4
    args.grad_clip = 0
    args.mem_log_every_batches = args.malloc_trim_every_batches = args.gc_every_batches = 0
    args.log_perf = "no"
    model = ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    batch = {"mixture": torch.ones(1, 1600, 8), "target": torch.ones(1, 1600),
             "lengths": torch.tensor([1600]), "sample_ids": torch.tensor([0])}
    train_light.run_epoch(model, [batch], optimizer, torch.amp.GradScaler("cpu", enabled=False),
                          torch.device("cpu"), [], args, training=True,
                          loss_fn=lambda estimate, target, frames: (estimate - 1).square())
    # One item means d[(w-1)^2]/dw=-2, hence the correct SGD update is w=2.
    # Existing defaults produce w=0.5: a fourfold attenuation of this tail step.
    assert model.weight.item() == pytest.approx(2.0)

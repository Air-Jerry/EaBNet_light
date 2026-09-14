import pytest
import torch

from cts_features import build_mixture_stft, build_stft_batch, reconstruct_waveform


def test_frontend_matches_explicit_torch_stft_convention():
    torch.manual_seed(56)
    wave = torch.randn(2, 1027, 1)
    observed = build_mixture_stft(wave)
    reference = torch.stft(wave[..., 0], n_fft=320, hop_length=160, win_length=320,
                           window=torch.hann_window(320, periodic=True), center=True,
                           normalized=False, onesided=True, pad_mode="reflect", return_complex=True)
    expected = torch.view_as_real(reference).permute(0, 2, 1, 3).unsqueeze(3)
    assert torch.equal(observed, expected)


@pytest.mark.parametrize("length", [321, 1280, 1487])
@pytest.mark.parametrize("power", [1.0, 0.5])
def test_round_trip(length, power):
    torch.manual_seed(213)
    wave = torch.randn(1, length, 1) * 0.05
    stft = build_mixture_stft(wave, power=power)
    ri = stft.squeeze(3).permute(0, 3, 1, 2)
    reconstructed = reconstruct_waveform(ri, length=length, power=power)
    assert torch.allclose(reconstructed, wave[..., 0], atol=1e-6, rtol=1e-5)


def test_each_utterance_has_its_own_reflection_boundary():
    torch.manual_seed(1)
    wave = torch.randn(2, 1487, 1)
    lengths = torch.tensor([321, 1487])
    mixed, target = build_stft_batch(wave, wave[..., 0], 320, 160, 320, 1, torch.device("cpu"), lengths)
    short = build_mixture_stft(wave[:1, :321])
    count = 321 // 160 + 1
    assert torch.equal(mixed[:1, :count], short)
    assert torch.count_nonzero(mixed[0, count:]) == 0
    assert torch.equal(target, mixed.squeeze(3).permute(0, 3, 1, 2))


def test_frontend_rejects_invalid_data():
    with pytest.raises(ValueError, match="monaural"):
        build_mixture_stft(torch.randn(1, 800, 8))
    with pytest.raises(ValueError, match="more than"):
        build_mixture_stft(torch.randn(1, 100, 1))
    with pytest.raises(ValueError, match="NaN or Inf"):
        build_mixture_stft(torch.full((1, 800, 1), float("nan")))
    with pytest.raises(ValueError, match="161 bins"):
        build_mixture_stft(torch.randn(1, 800, 1), n_fft=512)


def test_training_fails_on_degenerate_mixture_but_accepts_silent_target():
    with pytest.raises(ValueError, match="constant/silent mixture"):
        build_stft_batch(torch.zeros(1, 800, 1), torch.zeros(1, 800),
                         320, 160, 320, 1, torch.device("cpu"))
    _, target = build_stft_batch(torch.randn(1, 800, 1), torch.zeros(1, 800),
                                  320, 160, 320, 1, torch.device("cpu"))
    assert torch.count_nonzero(target) == 0

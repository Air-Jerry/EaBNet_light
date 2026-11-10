"""Preserve reflection values and the gradients of the actual centered STFT objective."""
import pytest
import torch
import torch.nn.functional as F

from experiments.deterministic_reflect_pad import deterministic_reflection_padding


@pytest.mark.parametrize('padding',[(0,0),(0,4),(4,0),(3,5)])
def test_reflection_values_and_overlapping_gradients(padding):
    torch.manual_seed(1234)
    original=torch.randn(2,11,dtype=torch.double,requires_grad=True)
    candidate=original.detach().clone().requires_grad_()
    expected=F.pad(original,padding,'reflect')
    with deterministic_reflection_padding():
        actual=F.pad(candidate,padding,'reflect')
        assert torch.equal(expected,actual)
        assert torch.autograd.gradcheck(lambda x:F.pad(x,padding,'reflect'),(candidate,))
    weights=torch.linspace(.2,2,expected.numel()).reshape_as(expected)
    (expected*weights).sum().backward();(actual*weights).sum().backward()
    torch.testing.assert_close(original.grad,candidate.grad,rtol=0,atol=1e-14)


def test_centered_stft_spectra_and_waveform_gradients_are_preserved():
    torch.manual_seed(1234)
    original=torch.randn(2,1600,requires_grad=True);candidate=original.detach().clone().requires_grad_()
    window=torch.hann_window(320)
    settings=dict(n_fft=512,hop_length=160,win_length=320,window=window,
                  center=True,pad_mode='reflect',return_complex=True)
    expected=torch.stft(original,**settings)
    with deterministic_reflection_padding():actual=torch.stft(candidate,**settings)
    assert torch.equal(expected,actual)
    expected.abs().log1p().mean().backward();actual.abs().log1p().mean().backward()
    torch.testing.assert_close(original.grad,candidate.grad,rtol=1e-6,atol=1e-8)


def test_scope_restores_on_error_and_other_padding_stays_unchanged():
    original=F.pad;value=torch.arange(12.).reshape(2,6)
    expected=F.pad(value,(2,3),'constant',.4)
    with pytest.raises(RuntimeError):
        with deterministic_reflection_padding():
            assert torch.equal(F.pad(value,(2,3),'constant',.4),expected)
            assert torch.equal(F.pad(input=value,pad=(2,3),mode='constant',value=.4),expected)
            F.pad(value,(6,1),'reflect')
    assert F.pad is original

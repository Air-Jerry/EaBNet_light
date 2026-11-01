"""Explicit causal recurrence, future gradients and bounded streaming state."""

import pytest
import torch

from experiments.causal_wiener_projection import CausalWienerProjection


def test_projection_matches_explicit_frame_updates_and_reference_conjugation():
    torch.set_num_threads(2)
    torch.manual_seed(7341)
    mixture = torch.randn(2, 97, 9, 4, dtype=torch.complex64) * .1
    estimate = mixture[..., 0] + torch.randn(2, 97, 9, dtype=torch.complex64) * .02
    layer = CausalWienerProjection()
    with torch.inference_mode():
        actual = layer(mixture, estimate)
        covariance = torch.zeros(2, 9, 4, 4, dtype=torch.complex128)
        correlation = torch.zeros(2, 9, 4, dtype=torch.complex128)
        outputs = []
        for index in range(97):
            x, s = mixture[:, index].cdouble(), estimate[:, index].cdouble()
            covariance = .96 * covariance + .04 * x.unsqueeze(-1) * x.conj().unsqueeze(-2)
            correlation = .96 * correlation + .04 * x * s.conj().unsqueeze(-1)
            matrix = (covariance + covariance.mH) * .5
            load = (.01 * matrix.diagonal(dim1=-2, dim2=-1).real.mean(-1)).clamp_min(1e-8)
            weights = torch.linalg.solve(matrix + load[..., None, None] * torch.eye(4), correlation.unsqueeze(-1))
            outputs.append((weights.squeeze(-1).conj() * x).sum(-1))
        expected = torch.stack(outputs, 1).to(estimate.dtype)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(actual[:, 0], estimate[:, 0] / (1 + .01 / 4), rtol=1e-5, atol=1e-7)


def test_projection_has_no_future_dependency_and_finite_complex_gradients():
    torch.set_num_threads(2)
    torch.manual_seed(7342)
    mixture = torch.randn(1, 83, 7, 8, dtype=torch.complex64, requires_grad=True)
    estimate = torch.randn(1, 83, 7, dtype=torch.complex64, requires_grad=True)
    layer = CausalWienerProjection()
    expected = layer(mixture, estimate)
    changed_x, changed_s = mixture.detach().clone(), estimate.detach().clone()
    changed_x[:, 39:] *= 17
    changed_s[:, 39:] *= 23
    torch.testing.assert_close(expected[:, :39], layer(changed_x, changed_s)[:, :39], rtol=0, atol=0)
    expected[:, :39].abs().square().mean().backward()
    for tensor in (mixture, estimate):
        assert torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad[:, 39:]) == 0
        assert torch.count_nonzero(tensor.grad[:, :39]) > 0


def test_projection_matches_irregular_chunks_with_owned_bounded_state():
    torch.set_num_threads(2)
    torch.manual_seed(7343)
    mixture = torch.randn(1, 211, 7, 8, dtype=torch.complex64)
    estimate = torch.randn(1, 211, 7, dtype=torch.complex64)
    layer = CausalWienerProjection()
    with torch.inference_mode():
        expected = layer(mixture, estimate)
        pieces, state = [], None
        for start, end in ((0, 1), (1, 8), (8, 72), (72, 131), (131, 211)):
            part, state = layer.process(mixture[:, start:end], estimate[:, start:end], state, cache=True)
            pieces.append(part)
            assert state['covariance'].shape == (1, 7, 8, 8)
            assert state['covariance'].untyped_storage().nbytes() == 7 * 8 * 8 * 16
            assert state['correlation'].untyped_storage().nbytes() == 7 * 8 * 16
        torch.testing.assert_close(torch.cat(pieces, 1), expected, rtol=1e-5, atol=1e-7)
        with pytest.raises(ValueError, match='state'):
            layer.process(mixture.expand(2, -1, -1, -1), estimate.expand(2, -1, -1), state, cache=True)

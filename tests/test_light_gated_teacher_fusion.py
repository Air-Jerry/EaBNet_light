"""Nonzero causal paths, teacher preservation and explicit past-tap ownership."""
import copy

import pytest
import torch
from torch import nn

from experiments.causal_gated_teacher_fusion import CausalGatedTeacherFusion, TeacherSpatialInput
from light_causal_stream import CausalWaveformStream
from light_quality_loss import QualityObjective
from train_light import build_stft_batch


class CausalTeacher(nn.Module):
    is_causal = True
    M, context = 8, 32

    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(.7))

    def process(self, value, state=None):
        reference = value[..., 0, :]
        previous = reference.new_zeros(reference.shape[0], 1, 257, 2) if not state else state['previous']
        past = torch.cat((previous, reference), 1)
        output = self.gain * reference + .1 * past[:, :-1]
        return output.permute(0, 3, 1, 2), dict(previous=past[:, -1:].clone()) if state is not None else None


@pytest.fixture(params=[1, 4])
def trained(request):
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    model = CausalGatedTeacherFusion(CausalTeacher(), taps=request.param)
    value = torch.randn(2, 31, 257, 8, 2) * .02
    before = copy.deepcopy(model.teacher.state_dict())
    torch.testing.assert_close(model(value), model.teacher.process(value)[0], rtol=0, atol=0)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == (412626 if request.param == 1 else 428034)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001)
    for _ in range(3):
        model.train()
        assert not model.teacher.training
        optimizer.zero_grad(set_to_none=True)
        (model(value) - value[..., 7, :].permute(0, 3, 1, 2) * .6).square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.core.parameters())
        assert model.teacher.gain.grad is None
        torch.nn.utils.clip_grad_norm_(model.core.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
    assert torch.count_nonzero(model.core.down[0][0].weight.grad[:, 24:]) > 0
    assert all(torch.equal(before[key], value) for key, value in model.teacher.state_dict().items())
    assert all(torch.isfinite(value).all() for state in optimizer.state.values() for value in state.values())
    return model


def test_trained_future_is_unobservable_and_all_mics_reach_output(trained):
    model = trained
    value = (torch.randn(1, 53, 257, 8, 2) * .02).requires_grad_()
    changed = value.detach().clone()
    changed[:, 23:] *= 13
    for training in (True, False):
        model.train(training)
        expected = model(value)
        torch.testing.assert_close(expected[:, :, :23], model(changed)[:, :, :23], rtol=0, atol=0)
    expected[:, :, :23].square().mean().backward()
    assert torch.count_nonzero(value.grad[:, 23:]) == 0
    assert torch.count_nonzero(value.grad[:, :23, :, 7]) > 0


def state_bytes(value):
    if torch.is_tensor(value):
        assert value.untyped_storage().nbytes() == value.numel() * value.element_size()
        return value.untyped_storage().nbytes()
    if isinstance(value, dict):
        return sum(state_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(state_bytes(item) for item in value)
    return 0


def test_nonzero_partition_and_owned_state(trained):
    model = trained.eval()
    value = torch.randn(2, 53, 257, 8, 2) * .02
    with torch.inference_mode():
        expected = model(value)
        state, pieces = None, []
        for begin, end in ((0, 1), (1, 17), (17, 28), (28, 53)):
            result, state = model.forward_stream(value[:, begin:end], state)
            pieces.append(result)
        torch.testing.assert_close(torch.cat(pieces, 2), expected, rtol=1e-4, atol=2e-6)
        saved = copy.deepcopy(state)
        model.forward_stream(value[:, :1], state)
        def compare(one, two):
            if torch.is_tensor(one):
                torch.testing.assert_close(one, two, rtol=0, atol=0)
            elif isinstance(one, dict):
                for key in one:
                    compare(one[key], two[key])
            elif isinstance(one, (list, tuple)):
                for left, right in zip(one, two):
                    compare(left, right)
        compare(saved, state)
        assert state_bytes(state) == 2 * (600576 + 2056 + (49344 if model.taps == 4 else 0))
        with pytest.raises(ValueError, match='state differs'):
            model.forward_stream(value[:1, :1], state)


def test_actual_frontend_600_waveform_hops(trained):
    model = trained.eval()
    wave = torch.randn(1, 96000, 8) * .02
    with torch.inference_mode():
        features, _ = build_stft_batch(wave, wave[:, :, 0], 512, 160, 320, .5, torch.device('cpu'))
        full = QualityObjective().reconstruct(model(features), 96000)
        stream = CausalWaveformStream(model)
        stream.warmup()
        parts = [stream.push(wave[:, begin:begin + 160]) for begin in range(0, 96000, 160)]
        parts.append(stream.flush())
        torch.testing.assert_close(torch.cat(parts, 1), full, rtol=1e-4, atol=2e-6)


def test_teacher_and_spatial_input_contract():
    teacher = CausalTeacher()
    teacher.is_causal = False
    with pytest.raises(ValueError, match='causal teacher'):
        CausalGatedTeacherFusion(teacher)
    for taps in (True, 0, 1., 2, 5):
        with pytest.raises(ValueError, match='tap'):
            CausalGatedTeacherFusion(CausalTeacher(), taps=taps)
    features = torch.randn(1, 26, 3, 257)
    features[:, 0:24:3].abs_()
    augmented = TeacherSpatialInput.augment(features)
    assert augmented.shape == (1, 47, 3, 257)
    torch.testing.assert_close(augmented[:, :24], features[:, :24], rtol=0, atol=0)
    torch.testing.assert_close(augmented[:, -2:], features[:, -2:], rtol=0, atol=0)
    assert torch.isfinite(TeacherSpatialInput.augment(torch.zeros_like(features))).all()


def test_four_tap_filter_uses_exactly_the_three_frame_old_microphone_impulse():
    torch.set_num_threads(2)
    model = CausalGatedTeacherFusion(CausalTeacher(), taps=4).eval()
    with torch.no_grad():
        model.core.output.bias[14] = 1.
        value = torch.zeros(1, 5, 257, 8, 2)
        value[:, 0, :, 7, 0] = 1.
        expected = torch.zeros(1, 2, 5, 257)
        expected[:, 0, 3] = 1.
        torch.testing.assert_close(model(value), expected, rtol=0, atol=0)

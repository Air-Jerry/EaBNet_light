"""Past/current tap alignment, complete warm identity, actual learning and owned causal states."""
import copy

import pytest
import torch

from experiments.causal_frame_graph import tensor_leaves
from experiments.causal_past_filter_monaural import CausalPastFilterMonaural, past_filter_residual
from light_candidate_runtime import RECURRENT_ADAPT_CANDIDATE, build_model_candidate, candidate_metadata


def models():
    torch.set_num_threads(2); torch.manual_seed(1234)
    source = build_model_candidate(RECURRENT_ADAPT_CANDIDATE)
    with torch.no_grad():
        for module in (source.base.base.beamformer, source.base.base.mapping_output,
                       source.base.base.retention.output, source.base.output, source.output):
            module.weight.normal_(std=.01)
    model = CausalPastFilterMonaural(copy.deepcopy(source.base))
    model.load_warm(source.state_dict())
    return source, model


def test_complex_current_and_previous_taps_match_explicit_formula_and_chunk_history():
    torch.manual_seed(41)
    # Integer inputs keep the complex tap-alignment oracle exact in FP32.
    raw = torch.randint(-8, 9, (2, 13, 3, 2, 2)).float()
    coefficients = torch.zeros(2, 13, 3, 5, 2, 2)
    coefficients[..., -1, 0, 0] = 1
    coefficients[..., -2, 1, 0] = 2
    coefficients[..., -3, 0, 1] = .5
    actual, history = past_filter_residual(raw, coefficients)
    expected = raw[..., 0, :].clone()
    expected[:, 1:] += 2*raw[:, :-1, :, 1, :]
    expected[:, 2:, :, 0] -= .5*raw[:, :-2, :, 0, 1]
    expected[:, 2:, :, 1] += .5*raw[:, :-2, :, 0, 0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    pieces = []; state = None
    for start, end in ((0, 1), (1, 4), (4, 11), (11, 13)):
        output, state = past_filter_residual(raw[:, start:end], coefficients[:, start:end], state)
        pieces.append(output)
        assert state.shape == (2, 4, 3, 2, 2)
    assert torch.equal(torch.cat(pieces, 1), actual)
    assert torch.equal(state, history)
    assert state.untyped_storage().data_ptr() != raw.untyped_storage().data_ptr()


def test_complete_nonzero_warm_source_and_existing_metadata_are_preserved():
    metadata = candidate_metadata(RECURRENT_ADAPT_CANDIDATE)
    source, model = models(); source.eval(); model.eval()
    with torch.no_grad():
        value = torch.randn(1, 131, 257, 8, 2)*.1
        assert torch.equal(model(value), source(value))
    assert sum(p.numel() for p in model.parameters()) == 5687490
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 5102746
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in source.state_dict().items())
    assert metadata == candidate_metadata(RECURRENT_ADAPT_CANDIDATE)


def test_actual_adam_learns_filter_with_finite_gradients_and_unchanged_frozen_source():
    _, model = models(); model.train()
    before = model.deep_filter.weight.detach().clone()
    frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        output = model(torch.randn(1, 17, 257, 8, 2)*.1)
        loss = (output-torch.randn_like(output)*.05).square().mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        optimizer.step()
    assert not torch.equal(before, model.deep_filter.weight)
    assert all(torch.equal(value, dict(model.named_parameters())[name]) for name, value in frozen.items())
    assert {int(item['step']) for item in optimizer.state.values()} == {3}
    assert all(torch.isfinite(value).all() for item in optimizer.state.values() for value in item.values())


def test_nonzero_filter_has_no_future_gradient_and_fixed_independent_stream_states():
    _, model = models(); model.eval()
    with torch.no_grad(): model.deep_filter.weight.normal_(std=.01)
    value = (torch.randn(1, 131, 257, 8, 2)*.1).requires_grad_()
    full = model(value)
    derivative, = torch.autograd.grad(full[:, :, :65].square().mean(), value)
    assert torch.count_nonzero(derivative[:, 65:]) == 0
    with torch.inference_mode():
        changed = value.detach().clone(); changed[:, 65:] = torch.randn_like(changed[:, 65:])*10
        torch.testing.assert_close(model(changed)[:, :, :65], full[:, :, :65], rtol=0, atol=1e-6)
        pieces = []; state = None; sizes = []
        for start, end in ((0, 1), (1, 4), (4, 61), (61, 131)):
            incoming = [tensor.clone() for tensor in tensor_leaves(state)] if state is not None else None
            old = state
            output, state = model.forward_stream(value.detach()[:, start:end], state)
            pieces.append(output)
            sizes.append(sum(t.numel()*t.element_size() for t in tensor_leaves(state)))
            assert state['input_history'].shape == (1, 4, 257, 8, 2)
            if old is not None:
                assert all(torch.equal(a, b) for a, b in zip(incoming, tensor_leaves(old)))
                assert {t.untyped_storage().data_ptr() for t in tensor_leaves(old)}.isdisjoint(
                    {t.untyped_storage().data_ptr() for t in tensor_leaves(state)})
        torch.testing.assert_close(torch.cat(pieces, 2), full, rtol=0, atol=1e-5)
        assert sizes == [6284816]*4
        _, independent = model.forward_stream(value.detach()[:, :1])
        assert {t.untyped_storage().data_ptr() for t in tensor_leaves(state)}.isdisjoint(
            {t.untyped_storage().data_ptr() for t in tensor_leaves(independent)})


def test_warm_and_state_validation_reject_incomplete_or_incompatible_data():
    source, model = models()
    broken = dict(source.state_dict()); broken.pop('output.bias')
    with pytest.raises(ValueError, match='complete'): model.load_warm(broken)
    broken = dict(source.state_dict()); broken['output.weight'] = broken['output.weight'].double()
    with pytest.raises(ValueError, match='dtypes'): model.load_warm(broken)
    model.eval()
    with torch.inference_mode():
        value = torch.randn(1, 1, 257, 8, 2)*.1
        _, state = model.forward_stream(value)
        with pytest.raises(ValueError, match='complete'):
            model.forward_stream(value, {'base': state['base']})
        state['input_history'] = state['input_history'].double()
        with pytest.raises(ValueError, match='Past input'):
            model.forward_stream(value, state)
    with pytest.raises(ValueError): CausalPastFilterMonaural(copy.deepcopy(source.base), taps=True)


def test_full_six_second_waveform_matches_600_causal_hops_with_nonzero_past_filter():
    from light_causal_stream import CausalWaveformStream
    from train_light_causal_pmsqe import FrontendPMSQE
    _, model = models(); model.eval()
    with torch.no_grad(): model.deep_filter.weight.normal_(std=.002)
    waveform = torch.randn(1, 96000, 8)*.01
    objective = FrontendPMSQE(512, 160, 320, .5)
    with torch.inference_mode():
        features, _ = objective.build_features(waveform, waveform[..., 0], 512, 160, 320, .5, 'cpu', lengths=[96000])
        offline = objective.reconstruct(model(features), 96000)
        stream = CausalWaveformStream(model)
        chunks = [stream.push(waveform[:, offset:offset+160]) for offset in range(0, 96000, 160)]
        chunks.append(stream.flush())
        actual = torch.cat(chunks, 1)
        assert actual.shape == offline.shape == (1, 96000)
        torch.testing.assert_close(actual, offline, rtol=0, atol=1e-5)
        assert sum(t.numel()*t.element_size() for t in tensor_leaves(stream.state)) == 6284816

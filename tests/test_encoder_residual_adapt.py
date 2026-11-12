import torch

from experiments.causal_encoder_adapt_monaural import CausalEncoderAdaptMonaural
from experiments.causal_encoder_residual_adapt import CausalEncoderResidualAdapt
from light_candidate_runtime import WIDE_CONDITIONED_CANDIDATE, build_model_candidate


def test_residual_encoder_preserves_forward_and_updates_early_encoder():
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    original = CausalEncoderAdaptMonaural(build_model_candidate(WIDE_CONDITIONED_CANDIDATE))
    with torch.no_grad():
        original.base.base.beamformer.weight.normal_(std=.01)
        original.output.weight.normal_(std=.01)
    model = CausalEncoderResidualAdapt(build_model_candidate(WIDE_CONDITIONED_CANDIDATE))
    model.load_warm(original.state_dict())
    inputs = torch.randn(1, 20, 257, 8, 2) * .1
    train_original = original(inputs)
    before, after = [], []
    for layer in model.base.base.layers[:3]:
        layer.mhsa.register_forward_pre_hook(lambda _, args: before.append(args[0].requires_grad), prepend=True)
        layer.mhsa.register_forward_hook(lambda _, args, output: after.append(
            (args[0].requires_grad, output[0].requires_grad)))
    train_residual = model(inputs)
    assert torch.equal(train_original, train_residual)
    assert before == [True] * 3
    assert after == [(False, False)] * 3
    (train_residual - .1).square().mean().backward()
    names = ('base.base.stem.', 'base.base.encoders.')
    gradients = [p.grad for name, p in model.named_parameters() if name.startswith(names)]
    assert len(gradients) == 15 and all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert all(torch.count_nonzero(g) for g in gradients)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)

    model.eval()
    with torch.inference_mode():
        expected = model(inputs)
        left, state = model.forward_stream(inputs[:, :7])
        right, _ = model.forward_stream(inputs[:, 7:], state)
        torch.testing.assert_close(torch.cat((left, right), 2), expected, rtol=0, atol=1e-5)

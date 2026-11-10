"""The whole backbone can learn without changing inherited causal inference."""
import copy

import torch

from experiments.causal_unfrozen_root_monaural import CausalUnfrozenRootMonaural
from light_candidate_runtime import RECURRENT_ADAPT_CANDIDATE, build_model_candidate


def test_whole_root_receives_real_gradients_and_keeps_exact_warm_inference():
    torch.set_num_threads(2); torch.manual_seed(1234)
    source = build_model_candidate(RECURRENT_ADAPT_CANDIDATE)
    with torch.no_grad():
        for module in (source.base.base.beamformer, source.base.base.mapping_output,
                       source.base.base.retention.output, source.base.output, source.output):
            module.weight.normal_(std=.01)
    model = CausalUnfrozenRootMonaural(copy.deepcopy(source.base))
    model.load_state_dict(source.state_dict(), strict=True)
    assert all(p.requires_grad for p in model.parameters())
    assert sum(p.numel() for p in model.parameters()) == 5686130
    assert set(model.state_dict()) == set(source.state_dict())
    assert model.base.base.layers[0].full is model.base.base.layers[3].full
    features = torch.randn(1, 17, 257, 8, 2)*.1
    model.eval(); source.eval()
    with torch.inference_mode():
        torch.testing.assert_close(model(features), source(features), rtol=0, atol=0)
    model.train()
    assert all(layer.mhsa.training for layer in model.base.base.layers[:4])
    original = model.base.base.stem.conv.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(features)-.1).square().mean(); loss.backward()
        assert model.base.base.stem.conv.weight.grad.abs().sum() > 0
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
    assert not torch.equal(original, model.base.base.stem.conv.weight)
    assert torch.equal(original, source.base.base.stem.conv.weight)
    assert all(torch.isfinite(v).all() for item in optimizer.state.values() for v in item.values())

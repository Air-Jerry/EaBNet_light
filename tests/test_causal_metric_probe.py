"""The existing training-only surrogate also differentiates a causal generator."""

import torch

from light_causal_recurrent65_model import CausalRecurrentUNet65
from light_metric_loss import MetricObjective


def test_surrogate_trains_causal_generator_and_validation_does_not_update_it():
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    model = CausalRecurrentUNet65(hidden=32, ffn=64, layers=2, global_hidden=16, context=5)
    objective = MetricObjective('cpu', accelerated=True, score_fn=lambda *args: 2.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    clean = torch.randn(2, 8000) * .02
    mixture = clean[..., None].expand(-1, -1, 8) + torch.randn(2, 8000, 8) * .04
    source = model.beamformer.weight.detach().clone()
    for _ in range(2):
        model.train(); objective.training = True
        features, target = objective.build_features(mixture, clean, 512, 160, 320, .5, 'cpu', [8000] * 2)
        estimate = model(features)
        optimizer.zero_grad(set_to_none=True)
        loss = objective(estimate, target, [51] * 2)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    assert not torch.equal(model.beamformer.weight, source)
    assert objective.updates == 4
    assert objective.metric_stats()['valid_pesq_labels'] == 4
    objective.training = False; model.eval()
    with torch.inference_mode():
        features, target = objective.build_features(mixture, clean, 512, 160, 320, .5, 'cpu', [8000] * 2)
        assert torch.isfinite(objective(model(features), target, [51] * 2))
    assert objective.updates == 4
    objective.close()

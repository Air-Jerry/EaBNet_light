"""Branch ablation bypasses the mapping while retaining trained spatial weights."""

import torch

from experiments.causal_tcn_mapping import CausalTCNMapping
from experiments.diagnose_causal_branches import spatial_only


def test_spatial_ablation_retains_full_source_and_bypasses_nonzero_mapping():
    torch.set_num_threads(2)
    torch.manual_seed(314)
    model = CausalTCNMapping(hidden=32, ffn=64, layers=2, global_hidden=16, context=5, mapping_channels=8).eval()
    model.beamformer.weight.data.normal_(std=.01)
    value = torch.randn(1, 17, 257, 8, 2) * .02
    with torch.inference_mode():
        expected = model(value)
        torch.testing.assert_close(spatial_only(model, value), expected, rtol=0, atol=0)
        model.mapping_output.weight.normal_(std=.1)
        torch.testing.assert_close(spatial_only(model, value), expected, rtol=0, atol=0)
        assert (model(value) - expected).abs().max() > 1e-5

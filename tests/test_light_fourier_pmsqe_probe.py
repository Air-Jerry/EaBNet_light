"""The combined probe may omit only newly initialized Fourier weights."""

import pytest
import torch

from experiments.causal_tcn_mapping import CausalTCNMapping
from experiments.probe_causal_fourier_pmsqe import SourceCompatibleFourier


def test_combined_probe_strictly_preserves_old_weights_and_zero_output():
    torch.set_num_threads(2)
    settings = dict(hidden=32, ffn=64, layers=2, global_hidden=16, context=5, mapping_channels=8)
    source = CausalTCNMapping(**settings)
    source.mapping_output.weight.data.normal_(std=.01)
    source.beamformer.weight.data.normal_(std=.01)
    changed = SourceCompatibleFourier(**settings)
    result = changed.load_state_dict(source.state_dict())
    assert len(result.missing_keys) == 6
    with torch.inference_mode():
        source.eval(); changed.eval(); value = torch.randn(1, 7, 257, 8, 2) * .02
        torch.testing.assert_close(changed(value), source(value), rtol=0, atol=0)
    damaged = dict(source.state_dict()); damaged.pop('stem.conv.weight')
    with pytest.raises(AssertionError): changed.load_state_dict(damaged)
    damaged = dict(source.state_dict()); damaged['unexpected'] = torch.zeros(1)
    with pytest.raises(AssertionError): changed.load_state_dict(damaged)

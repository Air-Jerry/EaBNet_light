"""Accounting and decision checks for the diagnostic screen, not paper scores."""

import csv
import json

import pytest
import torch

import analyze_light_structure as analysis
from light_structure_variants import StructureConfig, build_candidate


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("value,accepted,distance", [
    (734999, False, 1), (735000, True, 0),
    (744999, True, 0), (745000, False, 1),
])
def test_rounding_interval_uses_integer_endpoints(value, accepted, distance):
    assert analysis.in_interval(value, analysis.FULL_RANGE) is accepted
    assert analysis.interval_distance(value, analysis.FULL_RANGE) == distance


def test_layer_accounting_deduplicates_shared_skip_tensors():
    model = build_candidate(StructureConfig(share_skip_attention=True))
    layers = analysis.count_parameters_by_module(model)
    assert sum(row["parameters"] for row in layers) == 767258
    aliases = analysis.shared_parameter_aliases(model)
    assert len(aliases) == 4  # CA weight/bias and two SA weights.
    assert all(len(names) == 5 for names in aliases)
    assert analysis.component_counts(model)["skip_attention"] == 8354


def test_matching_total_cannot_hide_a_wrong_ablation_count():
    row, _ = analysis.analyze_config(StructureConfig(ca_groups=2), torch.ones(1, 3, 257, 8, 2), 17)
    assert row["status"] == "complete"
    assert row["full_parameter_match"] is True
    assert row["ablation_parameter_match"] is False
    assert row["passes_both_parameter_constraints"] is False
    assert row["distance_to_both_intervals"] == 7056
    assert row["rejection_reason"] == "no_skip_model_outside_rounded_0.73M"


def test_default_matrix_mac_subtotals_match_independent_module_accounting():
    observed = analysis.observe_model(build_candidate(StructureConfig()), torch.ones(1, 100, 257, 8, 2))
    # Independently computed from actual convolution grids and LSTM dimensions;
    # omitted operators are explicitly excluded, so neither is a full profiler.
    assert observed["matrix_macs"]["total_transpose_input"] == 4679433280
    assert observed["matrix_macs"]["total_transpose_output"] == 6366870592
    assert observed["dfsmn_calls"] == 3
    assert observed["dfsmn_unique_called_modules"] == 1


def test_cli_screen_writes_consistent_reports_and_rejects_empty_shortlist(tmp_path):
    exit_code = analysis.main(["--output-dir", str(tmp_path), "--frames", "1", "--require-match"])
    assert exit_code == 2
    report = json.loads((tmp_path / "structure_candidates.json").read_text(encoding="utf-8"))
    with (tmp_path / "structure_candidates.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(report["candidates"]) == len(report["details"]) == 24
    assert report["execution_errors"] == 0
    assert report["shortlist"] == []
    assert report["core_sources_unchanged"] is True
    assert report["exact_reproduction_verified"] is False
    assert report["candidate_training_started"] is False
    for csv_row, row, detail in zip(rows, report["candidates"], report["details"]):
        assert csv_row["candidate_id"] == row["candidate_id"] == detail["candidate_id"]
        assert int(csv_row["parameters"]) == row["parameters"]
        assert not row["ablation_parameter_match"]
        assert detail["full_forward"]["output_shape"] == [1, 2, 1, 257]
        assert detail["ablated_forward"]["output_shape"] == [1, 2, 1, 257]
    assert "没有可进入候选训练" in (tmp_path / "structure_screen_summary.md").read_text(encoding="utf-8")

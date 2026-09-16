"""Execute a bounded 24-candidate architecture screen without training.

The candidates are diagnostic hypotheses, not verified author architectures.
Exit 0 means analysis completed; --require-match exits 2 if none satisfies both
rounded parameter counts. Execution errors always exit 1. Core source files,
training, datasets and the default model are never changed by this script.
"""

import argparse
import csv
import hashlib
import json
import math
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

from light_structure_variants import all_configs, build_candidate


ROOT = Path(__file__).resolve().parent
CORE_FILES = ("EaBNet_light.py", "train_light.py", "evaluate_light.py")
FULL_RANGE = (735000, 745000)
ABLATION_RANGE = (725000, 735000)


def source_hashes(names):
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def count_parameters_by_module(model):
    """Count each tensor once, without counting children at every ancestor."""
    seen = set()
    result = []
    for name, module in model.named_modules():
        own = []
        for local_name, parameter in module.named_parameters(recurse=False):
            if parameter.requires_grad and id(parameter) not in seen:
                own.append({"name": local_name, "shape": list(parameter.shape), "count": parameter.numel()})
                seen.add(id(parameter))
        if own:
            result.append({"module": name, "type": type(module).__name__,
                           "parameters": sum(item["count"] for item in own), "tensors": own})
    if sum(item["parameters"] for item in result) != parameter_count(model):
        raise RuntimeError("Unique per-module parameter counts do not sum to the model total")
    return result


def shared_parameter_aliases(model):
    aliases = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    return [names for names in aliases.values() if len(names) > 1]


def component_counts(model):
    parts = {name: parameter_count(getattr(model.cred, name))
             for name in ("encoder", "dfsmn", "skip_attention", "decoder", "out_conv")}
    parts["beamforming_head"] = parameter_count(model.bf_map)
    if sum(parts.values()) != parameter_count(model):
        raise RuntimeError("Component counts include an unexpected cross-component shared tensor")
    return parts


def analytic_counts(config):
    """Default-width count deltas, independent of inspecting model tensors."""
    # Ten backbone CAs; 64*128/g weights each. Biases remain unchanged.
    backbone = 758904 - (81920 - 81920 // config.ca_groups)
    if config.skip_fusion == "add":
        backbone -= 64 * 64 * (4 * 2 * 3 + 2 * 5)
    if config.dfsmn_layout == "flattened":
        # Two projections: 64 -> 64 -> 64 becomes 448 -> 64 -> 448.
        backbone += 2 * (448 - 64) * 64 + (448 - 64)
    one_skip = 8192 // config.ca_groups + 64 + 2 * 7 * 7
    skip = one_skip * (1 if config.share_skip_attention else 5)
    return {"full": backbone + skip, "without_skip": backbone, "skip": skip}


def in_interval(value, bounds):
    return bounds[0] <= value < bounds[1]


def interval_distance(value, bounds):
    """Integer distance to the nearest allowed count, not to rounded center."""
    return max(bounds[0] - value, value - (bounds[1] - 1), 0)


def observe_model(model, sample):
    """Collect real forward shapes and two explicitly scoped MAC estimates.

    One multiply-accumulate counts as one MAC. We count Conv/Linear/LSTM
    matrix products only. Both deconvolution conventions are emitted because
    the target paper does not identify its profiler; neither is author-matched.
    """
    stages, calls, handles = [], [], []
    counts = {"conv": 0, "transpose_input": 0, "transpose_output": 0, "linear": 0, "lstm": 0}

    def record_stage(name):
        def hook(_module, inputs, output):
            stages.append({"stage": name, "input": list(inputs[0].shape), "output": list(output.shape)})
        return hook

    def count(module, inputs, output):
        if isinstance(module, nn.ConvTranspose2d):
            area = math.prod(module.kernel_size)
            counts["transpose_input"] += inputs[0].numel() * (module.out_channels // module.groups) * area
            counts["transpose_output"] += output.numel() * (module.in_channels // module.groups) * area
        elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
            counts["conv"] += output.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)
        elif isinstance(module, nn.Linear):
            counts["linear"] += output.numel() * module.in_features
        elif isinstance(module, nn.LSTM):
            if module.num_layers != 1 or module.bidirectional or not module.batch_first:
                raise ValueError("This bounded profiler only supports the paper's single-layer unidirectional LSTMs")
            batch, time, _ = inputs[0].shape
            counts["lstm"] += batch * time * 4 * module.hidden_size * (module.input_size + module.hidden_size)

    try:
        for group in ("encoder", "decoder"):
            for index, module in enumerate(getattr(model.cred, group)):
                handles.append(module.register_forward_hook(record_stage(f"{group}.{index}")))
        handles.append(model.cred.dfsmn.register_forward_hook(
            lambda module, _inputs, _output: calls.append(id(module))))
        for module in model.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.LSTM)):
                handles.append(module.register_forward_hook(count))
        model.eval()
        with torch.inference_mode():
            output = model(sample)
        expected_shape = (sample.shape[0], 2, sample.shape[1], sample.shape[2])
        if tuple(output.shape) != expected_shape or not bool(torch.isfinite(output).all()):
            raise RuntimeError(f"Non-finite or wrong-shaped output: {tuple(output.shape)}")
        expected_frequencies = [127, 63, 31, 15, 7, 15, 31, 63, 127, 257]
        if [stage["output"][-1] for stage in stages] != expected_frequencies:
            raise RuntimeError("Encoder/decoder frequency geometry differs from the documented path")
        common = sum(counts[name] for name in ("conv", "linear", "lstm"))
        return {
            "input_shape": list(sample.shape), "output_shape": list(output.shape), "finite": True,
            "stages": stages, "dfsmn_calls": len(calls), "dfsmn_unique_called_modules": len(set(calls)),
            "matrix_macs": {**counts,
                            "total_transpose_input": common + counts["transpose_input"],
                            "total_transpose_output": common + counts["transpose_output"]},
        }
    finally:
        for handle in handles:
            handle.remove()


def hypotheses(config):
    notes = ["Diagnostic candidate; parameter matching cannot confirm author equivalence."]
    if config.ca_groups > 1:
        notes.append("Grouped CA acts on literal [all averages, all maxima] concatenation; groups can isolate pooling branches. No author evidence establishes this connectivity.")
    if config.share_skip_attention:
        notes.append("One skip-attention module is called at all five scales; the target paper does not state this sharing.")
    if config.skip_fusion == "add":
        notes.append("Skip fusion is addition with 64-channel decoder inputs; Figure 2 does not identify concat versus add.")
    if config.dfsmn_layout == "flattened":
        notes.append("Channel-major flattening uses 448 -> 64 projection -> 448; 64 denotes projection width here. This does not establish agreement with the paper's 64 hidden units.")
    return notes


def analyze_config(config, sample, seed):
    detail = {"candidate_id": config.candidate_id, "config": asdict(config),
              "author_confirmed": False, "hypotheses": hypotheses(config)}
    row = {"candidate_id": config.candidate_id, **asdict(config), "status": "error",
           "passes_both_parameter_constraints": False, "rejection_reason": ""}
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            full = build_candidate(config)
            full_count = parameter_count(full)
            full_parts = component_counts(full)
            full_layers = count_parameters_by_module(full)
            aliases = shared_parameter_aliases(full)
            full_observation = observe_model(full, sample)
            del full
            torch.manual_seed(seed)
            ablated = build_candidate(config, without_skip_attention=True)
            ablated_count = parameter_count(ablated)
            if parameter_count(ablated.cred.skip_attention) != 0:
                raise RuntimeError("Skip ablation still has trainable attention parameters")
            ablated_observation = observe_model(ablated, sample)
            del ablated
        expected = analytic_counts(config)
        actual = {"full": full_count, "without_skip": ablated_count, "skip": full_count - ablated_count}
        if actual != expected or actual["skip"] != full_parts["skip_attention"]:
            raise RuntimeError(f"Measured counts {actual} disagree with independent algebra {expected}")
        full_match, ablation_match = in_interval(full_count, FULL_RANGE), in_interval(ablated_count, ABLATION_RANGE)
        reasons = []
        if not full_match:
            reasons.append("full_model_outside_rounded_0.74M")
        if not ablation_match:
            reasons.append("no_skip_model_outside_rounded_0.73M")
        row.update(status="complete", parameters=full_count, without_skip_parameters=ablated_count,
                   skip_parameters=full_count - ablated_count, full_parameter_match=full_match,
                   ablation_parameter_match=ablation_match,
                   passes_both_parameter_constraints=full_match and ablation_match,
                   distance_to_both_intervals=interval_distance(full_count, FULL_RANGE) + interval_distance(ablated_count, ABLATION_RANGE),
                   finite_full=True, finite_without_skip=True, shape_match=True,
                   matrix_macs_transpose_input=full_observation["matrix_macs"]["total_transpose_input"],
                   matrix_macs_transpose_output=full_observation["matrix_macs"]["total_transpose_output"],
                   rejection_reason=";".join(reasons))
        detail.update(status="complete", parameter_counts=actual, analytic_counts=expected,
                      parameter_components=full_parts, parameters_by_module=full_layers,
                      shared_parameter_aliases=aliases, full_forward=full_observation,
                      ablated_forward=ablated_observation)
    except Exception as exc:
        row["rejection_reason"] = f"execution_error: {type(exc).__name__}: {exc}"
        detail.update(status="error", error=row["rejection_reason"])
    return row, detail


def write_summary(path, report):
    completed = [row for row in report["candidates"] if row["status"] == "complete"]
    nearest = sorted(completed, key=lambda row: row["distance_to_both_intervals"])[:3]
    lines = ["# 结构候选筛选结果", "", "这些是诊断性结构假设，不是作者已确认配置。", "",
             f"实际构建 {len(report['candidates'])} 种配置；完成 {len(completed)} 种，执行失败 {report['execution_errors']} 种；同时通过两个参数约束 {len(report['shortlist'])} 种。", "",
             "约束：完整模型 735,000–744,999 参数；无跳接注意力模型 725,000–734,999 参数（按表中百万参数保留两位小数的通常四舍五入）。", "",
             "| 最接近的配置 | 完整参数 | 无跳接注意力参数 | 距两个合法区间的距离之和 |", "|---|---:|---:|---:|"]
    for row in nearest:
        lines.append(f"| {row['candidate_id']} | {row['parameters']:,} | {row['without_skip_parameters']:,} | {row['distance_to_both_intervals']:,} |")
    lines += ["", "计数按唯一 Parameter 去重；消融模型实际移除跳接注意力。完整和消融模型均执行前向，核对输出、有限值及五级频率尺寸。", "",
              f"本次输入为 {report['input']['frames']} 个谱帧、8 麦、257 频点；10 ms 帧移下对应名义 {report['input']['frames'] * 0.01:g} 秒。真实 16,000 点波形在现有 center=True STFT 下产生 101 帧，因此 100 帧不能直接等同于该波形的完整前处理。", "",
              "MAC 仅统计 Conv/Linear/LSTM 矩阵乘加。转置卷积分别按输入贡献、输出网格上的稠密卷积代理计数；后者包含上采样零位的假想乘法，不代表实际算子执行量。FFT、DFSMN 手写记忆运算、归一化、池化、激活和其他逐元素操作不在内，不能当作作者 6.42 G/s 的同口径测量。", "",
              "分组 CA 保留 [全部平均池化, 全部最大池化] 的通道顺序，因此可能隔离两种池化分支；共享跳接没有目标论文依据；flattened 使用 448→64→448，64 是投影维，未确认是否符合原文 hidden units。", ""]
    if not report["shortlist"]:
        lines += ["**本轮没有可进入候选训练的结构。默认模型保持原样，没有启动候选训练。**", "",
                  "这排除了本轮 24 种具体组合，不能证明不存在其他实现，也不能证明论文参数表错误。下一步需要补充未披露结构的证据，不应无依据扩大参数搜索。"]
    else:
        lines += ["通过参数筛选的候选仍需要公式、梯度和受控训练验证。参数吻合不代表作者结构已经确认。"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=417)
    parser.add_argument("--require-match", action="store_true")
    args = parser.parse_args(argv)
    if args.frames < 1 or args.cpu_threads < 1:
        parser.error("frames and cpu-threads must be positive")
    before = source_hashes(CORE_FILES)
    previous_threads = torch.get_num_threads()
    rows, details = [], []
    try:
        torch.set_num_threads(args.cpu_threads)
        generator = torch.Generator().manual_seed(args.seed)
        sample = torch.randn(1, args.frames, 257, 8, 2, generator=generator)
        for index, config in enumerate(all_configs(), 1):
            row, detail = analyze_config(config, sample, args.seed)
            rows.append(row)
            details.append(detail)
            print(f"[{index}/24] {config.candidate_id}: {row.get('parameters', '?')} / "
                  f"{row.get('without_skip_parameters', '?')} -- {row['rejection_reason'] or 'count match'}", flush=True)
    finally:
        torch.set_num_threads(previous_threads)
    after = source_hashes(CORE_FILES)
    shortlist = [row["candidate_id"] for row in rows if row["passes_both_parameter_constraints"]]
    errors = sum(row["status"] == "error" for row in rows)
    report = {
        "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "24 diagnostic hypotheses; no model or training recipe selected by parameter matching",
        "exact_reproduction_verified": False, "candidate_training_started": False,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
        "input": {"batch": 1, "frames": args.frames, "frequency_bins": 257, "microphones": 8, "seed": args.seed},
        "matrix_mac_protocol": {
            "unit": "one multiply-accumulate is one MAC; raw total for the specified input tensor",
            "nominal_duration_seconds": args.frames * 0.01,
            "duration_note": "100 spectral frames correspond to a nominal 1 second at 10 ms hop, not an actual centered waveform-STFT call (which gives 101 frames for 16000 samples)",
            "included": ["Conv1d", "Conv2d", "ConvTranspose2d under two conventions", "Linear", "LSTM matrix products"],
            "transpose_input": "input-grid kernel contributions, including computations later removed by Chomp",
            "transpose_output": "output-grid dense-convolution proxy including hypothetical multiplies at upsampling zeros; not actual operator MACs",
            "excluded": ["FFT/iFFT", "manual DFSMN memory arithmetic", "normalization", "pooling", "activations", "bias adds", "other elementwise operations"],
            "author_profiler_known": False,
        },
        "constraints": {"full_parameters_half_open": FULL_RANGE, "without_skip_parameters_half_open": ABLATION_RANGE,
                        "basis": "target paper Table 2; ordinary rounding of parameters in millions to two decimals"},
        "core_sources_before": before, "core_sources_after": after, "core_sources_unchanged": before == after,
        "analysis_sources": source_hashes(("light_structure_variants.py", "analyze_light_structure.py")),
        "execution_errors": errors, "shortlist": shortlist, "candidates": rows, "details": details,
    }
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(name for row in rows for name in row))
    with (destination / "structure_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (destination / "structure_candidates.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    write_summary(destination / "structure_screen_summary.md", report)
    print(f"Completed: {len(rows)} candidates, {len(shortlist)} count matches, {errors} execution errors. Outputs: {destination}")
    if before != after or errors:
        return 1
    return 2 if args.require_match and not shortlist else 0


if __name__ == "__main__":
    raise SystemExit(main())

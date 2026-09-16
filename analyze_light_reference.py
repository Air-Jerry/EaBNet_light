"""Check source-derived hypotheses against three paper counts and learning.

This is a diagnostic experiment, not model selection on the paper's test set.
TCN replacement counts are derived from the cited original EaBNet source; that
ablation is not trained or evaluated here. A count match never certifies Eq.11.
"""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from EaBNet_light import com_mag_mse_loss
from analyze_light_structure import (CORE_FILES, FULL_RANGE, ABLATION_RANGE, component_counts,
                                     in_interval, observe_model, parameter_count, source_hashes)
from light_reference_variants import (all_reference_candidates, build_reference_candidate,
                                      reference_candidate_metadata)


TCN_RANGE = (2455000, 2465000)


def original_tcn_parameters(features=448, width=64, kernel=5, blocks=6, repeats=3):
    """Two pointwise weights, two dense temporal weights, 3 BN/PReLU pairs.

    Original EaBNet's default d_feat=256 is not its 512-FFT counterpart. The
    target geometry is 64*7=448; keeping that author's 6*3 blocks is an explicit
    inference. Temporal convolutions are dense, not depthwise.
    """
    return repeats * blocks * (2 * features * width + 2 * width * width * kernel + 9 * width)


def count_constraints(full, without_skip, dfsmn):
    tcn = full - dfsmn + original_tcn_parameters()
    return {"full_match": in_interval(full, FULL_RANGE),
            "without_skip_match": in_interval(without_skip, ABLATION_RANGE),
            "tcn_replacement_match": in_interval(tcn, TCN_RANGE),
            "tcn_replacement_parameters": tcn}


def learning_probe(candidate_id, seed, steps):
    """A fixed synthetic batch diagnoses trainability, never enhancement quality."""
    torch.manual_seed(seed)
    model = build_reference_candidate(candidate_id).train()
    generator = torch.Generator().manual_seed(seed + 1)
    mixture = torch.randn(1, 5, 257, 8, 2, generator=generator)
    target = (0.7 * mixture[:, :, :, 0, :]).permute(0, 3, 1, 2).contiguous()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    losses = []
    finite_gradients = True
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = com_mag_mse_loss(model(mixture), target, [5])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Synthetic optimization loss became non-finite")
        loss.backward()
        finite_gradients = finite_gradients and all(
            p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        if not finite_gradients:
            raise RuntimeError("Missing or non-finite parameter gradient")
        losses.append(float(loss.detach()))
        optimizer.step()
    with torch.no_grad():
        final_loss = float(com_mag_mse_loss(model(mixture), target, [5]))
    if not torch.isfinite(torch.tensor(final_loss)):
        raise RuntimeError("Post-update loss became non-finite")
    return {"steps": steps, "losses_before_updates": losses, "loss_after_last_update": final_loss,
            "loss_decreased": final_loss < losses[0], "all_parameter_gradients_finite": finite_gradients,
            "data": "one fixed synthetic STFT tensor, target=0.7*reference microphone",
            "quality_metric_or_generalization_evidence": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=417)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args(argv)
    if min(args.frames, args.steps, args.cpu_threads) < 1:
        parser.error("frames, steps, and cpu-threads must be positive")
    before = source_hashes(CORE_FILES)
    old_threads = torch.get_num_threads()
    rows, details = [], []
    try:
        torch.set_num_threads(args.cpu_threads)
        with torch.random.fork_rng(devices=[]):
            sample = torch.randn(1, args.frames, 257, 8, 2,
                                 generator=torch.Generator().manual_seed(args.seed))
            for candidate_id in all_reference_candidates():
                torch.manual_seed(args.seed)
                model = build_reference_candidate(candidate_id)
                full, parts = parameter_count(model), component_counts(model)
                observation = observe_model(model, sample)
                del model
                torch.manual_seed(args.seed)
                ablated = build_reference_candidate(candidate_id, without_skip_attention=True)
                no_skip = parameter_count(ablated)
                ablation_observation = observe_model(ablated, sample)
                if parameter_count(ablated.cred.skip_attention):
                    raise RuntimeError("Skip ablation retained trainable parameters")
                del ablated
                checks = count_constraints(full, no_skip, parts["dfsmn"])
                count_match = all(checks[key] for key in ("full_match", "without_skip_match", "tcn_replacement_match"))
                probe = learning_probe(candidate_id, args.seed, args.steps)
                row = {"candidate_id": candidate_id, "parameters": full, "without_skip_parameters": no_skip,
                       "dfsmn_parameters": parts["dfsmn"], **checks, "matches_three_rounded_counts": count_match,
                       "literal_equation_11": candidate_id.startswith("literal_"),
                       "author_confirmed": False, "initial_loss": probe["losses_before_updates"][0],
                       "final_loss": probe["loss_after_last_update"], "loss_decreased": probe["loss_decreased"]}
                rows.append(row)
                details.append({"candidate_id": candidate_id, "evidence": reference_candidate_metadata(candidate_id),
                                "parameter_components": parts,
                                "full_forward": observation, "without_skip_forward": ablation_observation,
                                "synthetic_learning": probe})
                print(f"{candidate_id}: {full:,}/{no_skip:,}/{checks['tcn_replacement_parameters']:,}; "
                      f"three counts={count_match}, loss={row['initial_loss']:.6f}->{row['final_loss']:.6f}", flush=True)
    finally:
        torch.set_num_threads(old_threads)
    after = source_hashes(CORE_FILES)
    if before != after:
        raise RuntimeError("Core source files changed during the diagnostic run")
    report = {
        "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "exact_reproduction_verified": False, "default_model_replaced": False,
        "full_benchmark_training_performed": False, "synthetic_optimization_performed": True,
        "environment": {"torch": torch.__version__, "device": "cpu"},
        "input_frames": args.frames, "seed": args.seed,
        "core_sources_before": before, "core_sources_after": after, "core_sources_unchanged": True,
        "analysis_sources": source_hashes(("light_reference_variants.py", "light_structure_variants.py",
                                           "analyze_light_reference.py", "analyze_light_structure.py")),
        "constraints": {"full_half_open": FULL_RANGE, "without_skip_half_open": ABLATION_RANGE,
                        "tcn_replacement_half_open": TCN_RANGE},
        "tcn_protocol": {"source": "https://github.com/Andong-Li-speech/EaBNet/blob/main/EaBNet.py",
                         "source_sha256": "fcca4f828765a82e2c4605b1ef1fcc630cfb30f97c70e45449eac6f62fe7e0e8",
                         "parameters": original_tcn_parameters(), "features": 448, "width": 64,
                         "kernel": 5, "blocks": 6, "repeats": 3,
                         "basis": "Conditional inference: original dense SqueezedTCM with feature width64*7",
                         "actual_tcn_ablation_forward_or_training_performed": False},
        "reference_attention_source": "https://github.com/Jongchan/attention-module/blob/459efad0e05ee7dde50c41ca10a3d0800bc3792a/MODELS/cbam.py",
        "limitations": ["CBAM candidates replace target Eq.11 with the different reference[13] shared two-layer MLP.",
                        "Only the ChannelGate is borrowed; spatial attention remains target Eq.5 interpretation.",
                        "Flat projection64 and flat hidden64 have different meanings; target paper does not resolve this.",
                        "Three rounded counts and synthetic loss cannot establish author-code equivalence.",
                        "Matrix MACs reuse the explicitly incomplete protocol in analyze_light_structure.py."],
        "count_compatible_candidates": [row["candidate_id"] for row in rows if row["matches_three_rounded_counts"]],
        "equation_and_count_compatible_candidates": [row["candidate_id"] for row in rows
                                                      if row["matches_three_rounded_counts"] and row["literal_equation_11"]],
        "candidates": rows, "details": details,
    }
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "reference_candidates.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    with (destination / "reference_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 引用文献结构排查", "", "这是结构诊断与合成 batch 学习检查，不是论文性能复现。", "",
             "| 候选 | 完整参数 | 无 skip 参数 | TCN 替换推算参数 | 三组计数符合 | 式(11)原算子 | 合成 loss |",
             "|---|---:|---:|---:|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['candidate_id']} | {row['parameters']:,} | {row['without_skip_parameters']:,} | "
                     f"{row['tcn_replacement_parameters']:,} | {row['matches_three_rounded_counts']} | "
                     f"{row['literal_equation_11']} | {row['initial_loss']:.6f} → {row['final_loss']:.6f} |")
    lines += ["", "TCN 列按原 EaBNet 的 6×3 个稠密 SqueezedTCM、64 宽、核5及当前 64×7=448 输入维计算；这是有条件推算，没有实际训练这个消融。",
              "", "CBAM 候选使用引用[13]官方实现的共享 MLP、reduction=16 和两个有 bias 的 Linear；它与目标论文式(11)的单 Conv 并不等价。",
              "", "因此，即使三个参数量同时吻合，也只能得到待验证候选，不能据此替换默认模型或归责论文。训练适配入口保留原训练主函数、数据和 loss，并将候选名及源码哈希写入 checkpoint。",
              "", "默认模型和原训练、旧评估文件 SHA256 均未改变。合成训练结果只证明该小批数据上的基本可训练性。"]
    (destination / "reference_screen_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 2 if args.require_exact else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Record a bounded, data-free audit; completion is not reproduction certification.

Run ``python audit_light.py`` to write output/light_audit.json, or add
``--require-exact`` to exit 2 while author-equivalence remains unresolved.
No dataset is opened, checkpoint is loaded, or training is started.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

from EaBNet_light import EaBNet


ROOT = Path(__file__).resolve().parent
PRESERVED_HASHES = {
    "train_light.py": "608A8D7AC3DD6E615C502613E525F764FC67E9E9C4C9C3D0E167AB2A0211F391",
    "evaluate_light.py": "1F86987E39C011293CA774866712E12E37368DB31352AF030D1F2FF920D4F002",
}
DISCLOSED_DEFAULTS = {
    "sample_rate": 16000,
    "win_length": 320,
    "hop_length": 160,
    "n_fft": 512,
    "num_mics": 8,
    "channels": 64,
    "norm_type": "BN",
    "dfsmn_layers": 3,
    "learning_rate": 0.001,
    "train_lr_reduce_on_plateau": "yes",
    "lr_reduce_metric": "val",
    "train_lr_patience": 2,
    "train_lr_factor": 0.5,
    "train_lr_min_delta": 0.0,
}


def count_parameters(module):
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def check(name, observed, expected, source):
    return {
        "item": name,
        "observed": observed,
        "expected": expected,
        "status": "matched" if observed == expected else "mismatch",
        "source": source,
    }


def file_fingerprints():
    result = {}
    for name in ("EaBNet_light.py", "train_light.py", "evaluate_light.py", "evaluate_light_paper.py", "audit_light.py"):
        digest = hashlib.sha256((ROOT / name).read_bytes()).hexdigest().upper()
        result[name] = {"path": str(ROOT / name), "sha256": digest}
        if name in PRESERVED_HASHES:
            result[name].update(
                original_sha256=PRESERVED_HASHES[name],
                unchanged=digest == PRESERVED_HASHES[name],
            )
    return result


def dependency_versions():
    result = {"python": platform.python_version(), "torch": torch.__version__}
    for name in ("numpy", "soundfile", "tensorboard", "tqdm", "pesq", "pystoi", "mir_eval"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def observe_model():
    """Exercise a CPU inference with hooks; restore caller RNG/thread settings."""
    previous_threads = torch.get_num_threads()
    handles = []
    stages = []
    dfsmn_calls = []

    def record_stage(name):
        def hook(_module, inputs, output):
            stages.append({"stage": name, "input": list(inputs[0].shape), "output": list(output.shape)})
        return hook

    try:
        torch.set_num_threads(2)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(417)
            model = EaBNet().cpu().eval()
            for group in ("encoder", "decoder"):
                for index, stage in enumerate(getattr(model.cred, group)):
                    handles.append(stage.register_forward_hook(record_stage(f"{group}.{index}")))
            handles.append(model.cred.dfsmn.register_forward_hook(
                lambda module, _inputs, _output: dfsmn_calls.append(id(module))
            ))
            sample = torch.randn(1, 9, 257, 8, 2)
            with torch.inference_mode():
                output = model(sample)

            parts = {
                "encoder": count_parameters(model.cred.encoder),
                "dfsmn": count_parameters(model.cred.dfsmn),
                "skip_attention": count_parameters(model.cred.skip_attention),
                "decoder": count_parameters(model.cred.decoder),
                "cred_output_projection": count_parameters(model.cred.out_conv),
                "beamforming_head": count_parameters(model.bf_map),
            }
            lstms = [
                {"input_size": layer.input_size, "hidden_size": layer.hidden_size,
                 "num_layers": layer.num_layers, "bidirectional": layer.bidirectional}
                for layer in model.bf_map.modules() if isinstance(layer, nn.LSTM)
            ]
            linears = [
                {"in_features": layer.in_features, "out_features": layer.out_features}
                for layer in model.bf_map.modules() if isinstance(layer, nn.Linear)
            ]
            kernels, strides = [], []
            for stage in model.cred.encoder:
                convolution = next(layer for layer in stage.in_conv.modules() if isinstance(layer, nn.Conv2d))
                kernels.append(list(convolution.kernel_size))
                strides.append(list(convolution.stride))
            return {
                "device": "cpu", "mode": "eval", "seed": 417,
                "input_shape": list(sample.shape), "output_shape": list(output.shape),
                "output_finite": bool(torch.isfinite(output).all()),
                "stages": stages,
                "parameters": {"total": count_parameters(model), "by_component": parts},
                "dfsmn": {
                    "configured_layers": model.cred.dfsmn_layers,
                    "observed_calls": len(dfsmn_calls),
                    "unique_called_modules": len(set(dfsmn_calls)),
                    "hidden_width": model.cred.dfsmn.out_conv.out_channels,
                    "projection_width": model.cred.dfsmn.in_conv.out_channels,
                    "left_memory_order": model.cred.dfsmn.memory_size,
                    "right_memory_order": model.cred.dfsmn.right_memory_size,
                },
                "lstms": lstms, "head_linear_layers": linears,
                "encoder_kernels_time_frequency": kernels,
                "encoder_strides_time_frequency": strides,
            }
    finally:
        for handle in handles:
            handle.remove()
        torch.set_num_threads(previous_threads)


def build_report():
    # Importing the training module defines its parser; main() is not invoked.
    import train_light

    defaults = vars(train_light.parse_args([]))
    observation = observe_model()
    fingerprints = file_fingerprints()
    parameters = observation["parameters"]
    total = parameters["total"]
    skip_parameters = parameters["by_component"]["skip_attention"]
    checks = [check(name, defaults.get(name), expected, "paper section 4.2; microphones: section 4.1")
              for name, expected in DISCLOSED_DEFAULTS.items()]
    checks.extend([
        check("encoder_stage_count", len([s for s in observation["stages"] if s["stage"].startswith("encoder")]), 5, "section 4.2"),
        check("decoder_stage_count", len([s for s in observation["stages"] if s["stage"].startswith("decoder")]), 5, "section 4.2"),
        check("encoder_kernels_time_frequency", observation["encoder_kernels_time_frequency"], [[2, 5]] + [[2, 3]] * 4, "section 4.2, axes transposed to code convention"),
        check("encoder_strides_time_frequency", observation["encoder_strides_time_frequency"], [[1, 2]] * 5, "section 4.2, axes transposed to code convention"),
        check("dfsmn_call_count", observation["dfsmn"]["observed_calls"], 3, "section 4.2"),
        check("dfsmn_unique_called_modules", observation["dfsmn"]["unique_called_modules"], 1, "section 4.2: shared weights"),
        check("dfsmn_hidden_width", observation["dfsmn"]["hidden_width"], 64, "section 4.2"),
        check("lstm_count", len(observation["lstms"]), 2, "figure 1 and section 4.2"),
        check("lstm_hidden_widths", [layer["hidden_size"] for layer in observation["lstms"]], [64, 64], "section 4.2"),
        check("head_linear_count", len(observation["head_linear_layers"]), 1, "figure 1 and section 3.1"),
    ])
    unresolved = [
        {"category": "architecture", "item": "author_implementation",
         "detail": "No verified author source/checkpoint/config was provided; equation checks do not establish author-code equivalence."},
        {"category": "architecture", "item": "dfsmn_unspecified_settings",
         "detail": "Reference [16] supplies the cell equation; target-paper memory order, stride, projection width, right context and per-frequency versus flattened C*F layout remain unspecified. Current choices: order 20, stride 1, projection 64, no right context, per-frequency sequences."},
        {"category": "architecture", "item": "attention_and_fusion",
         "detail": "SA kernel/padding/weight sharing, CA exact pooling and convolution details, FFT-convolution kernel/bias, skip concatenation versus addition and decoder boundary treatment are not fully specified."},
        {"category": "architecture", "item": "parameter_total",
         "detail": f"Observed {total:,} parameters; paper reports 0.74M. Current revision is expected to have 800,674, not the paper total."},
        {"category": "architecture", "item": "skip_attention_ablation",
         "detail": f"Observed {skip_parameters:,} skip-attention parameters. Table 2 changes 0.73M to 0.74M; ordinary rounding permits an increment below 20,000, so 41,770 cannot explain it."},
        {"category": "training", "item": "recipe_and_loss",
         "detail": "Loss, power compression (current default 0.5), batch/accumulation, crop, epochs, initialization, seeds, AMP and exact optimizer details are not all disclosed. Original training logic is preserved; this audit only compares declared defaults, not a training run or command-line overrides."},
        {"category": "training", "item": "preserved_training_implementation_issues",
         "detail": "Original run_epoch attenuates incomplete final accumulation windows; Python random crops are not seeded by main; batched waveform padding changes short-utterance STFT boundaries. These remain unchanged under the user's training-preservation constraint. A strict xfail test reproduces the accumulation issue."},
        {"category": "training", "item": "end_to_end_causality",
         "detail": "Global temporal pooling and symmetric SA use future context. A causal convolution/DFSMN flag does not certify streaming causality, and the target paper does not specify a fully causal proposed model."},
        {"category": "data", "item": "dataset_and_targets",
         "detail": "No dataset is read. Exact utterance selection, split, RIR/SNR draws, array geometry, clean/early target construction and author evaluation subset are unverified."},
        {"category": "metrics", "item": "paper_benchmark",
         "detail": "No trained checkpoint or benchmark evaluation is run. PESQ mode, alignment/length/level handling and exact Si-SNR conventions need author confirmation; unchanged evaluate_light.py is not a certificate of the paper's protocol."},
        {"category": "metrics", "item": "macs",
         "detail": "The paper's 6.42 G MAC/s was not profiled; a parameter count or forward pass does not establish this compute figure."},
    ]
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_completed": True,
        "exact_reproduction_verified": False,
        "meaning": "Exit 0 means this data-free audit completed, not that reproduction is exact or paper metrics are reached.",
        "environment": {"python_executable": sys.executable, "platform": platform.platform(),
                        "dependencies": dependency_versions(), "torch_cuda_build": torch.version.cuda},
        "file_fingerprints": fingerprints,
        "original_training_and_evaluation_preserved": all(fingerprints[name]["unchanged"] for name in PRESERVED_HASHES),
        "training_defaults": defaults,
        "disclosed_config_checks": checks,
        "model_observation": observation,
        "implementation_checks": [
            check("output_shape", observation["output_shape"], [1, 2, 9, 257], "preserved model interface"),
            check("finite_output", observation["output_finite"], True, "bounded random-input CPU smoke check"),
            check("component_parameter_sum", sum(parameters["by_component"].values()), total, "unique parameter counts"),
        ],
        "paper_parameter_comparison": {
            "reported_total_millions": 0.74,
            "observed_total": total,
            "matches_two_decimal_rounding": 735000 <= total < 745000,
            "reported_without_skip_attention_millions": 0.73,
            "observed_skip_attention_parameters": skip_parameters,
            "compatible_with_reported_ablation_rounding": 0 < skip_parameters < 20000,
            "note": "Rounding comparisons are necessary consistency checks, not proof of architecture identity.",
        },
        "unresolved": unresolved,
        "paper_reference": {
            "title": "A Lightweight Fourier Convolutional Attention Encoder for Multi-Channel Speech Enhancement",
            "doi": "10.1109/ICASSP49357.2023.10095716",
            "evaluation_set": "ConferencingSpeech2021 development test set",
            "table_1_proposed": {"PESQ": 2.359, "STOI": 0.926, "E_STOI": 0.847, "Si_SNR": 11.10,
                                 "parameters_millions": 0.74, "MACs_G_per_second": 6.42},
            "results_are": "paper reference values, not measurements from this audit",
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/light_audit.json"))
    parser.add_argument("--require-exact", action="store_true", help="Exit 2 if exact reproduction is not verified")
    args = parser.parse_args(argv)
    report = build_report()
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Audit written: {destination}")
    print(f"Parameters: {report['model_observation']['parameters']['total']:,}; exact reproduction verified: false")
    return 2 if args.require_exact and not report["exact_reproduction_verified"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

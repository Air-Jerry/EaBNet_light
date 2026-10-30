"""Fine-tune a completed quality model with training-only real PESQ supervision."""

import argparse
import json
import math
from pathlib import Path
import sys

import torch

import train_light as trainer
import train_light_quality as quality
from light_candidate_runtime import candidate_metadata, read_candidate_checkpoint, select_model_factory
from light_metric_loss import MetricObjective
from light_quality_model import QUALITY_CANDIDATE


METRIC_FIELDS = ("metric_warmstart_checkpoint", "metric_warmstart_sha256", "metric_warmstart_epoch",
                 "metric_weight", "metric_discriminator_lr", "metric_implementation_sha256",
                 "metric_discriminator_step_unit")

LEGACY_RUNTIME = {
    "light_quality_loss.py": "d39581a21efd17823049385f09399e07bf63660d688192467bdd4c5c99ab88db",
    "light_metric_loss.py": "3ade5cbefd5cd5dfae0737dc2907ea331a583b9b6519ab81b9d3eccebc08d7cd",
    "train_light_metric.py": "66ac737342cf7947c734ce7e3aeaf6070869fa3311b31ccb52bac4289bea3d6e",
}


def parse_args(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    selector = argparse.ArgumentParser(description=__doc__, add_help=False)
    selector.add_argument("--warmstart")
    selector.add_argument("--metric-weight", type=float)
    selector.add_argument("--metric-lr", type=float)
    selector.add_argument("--metric-performance", choices=("standard", "overlap"))
    selector.add_argument("--upgrade-runtime", action="store_true")
    selected, remaining = selector.parse_known_args(arguments)
    args = trainer.parse_args(remaining, defaults={
        "checkpoint_dir": "./checkpoints_cbam_metric_v1", "best_dir": "./bestmodels_cbam_metric_v1",
        "log_dir": "./logs_cbam_metric_v1", "num_epochs": 6, "learning_rate": 1e-4,
        "batch_size": 4, "grad_accum_steps": 1, "segment_seconds": 6, "save_every": 1,
        "train_lr_patience": 3, "use_amp": "no", "model_amp": "no", "allow_tf32": "no",
        "stop_on_non_finite": "yes", "skip_non_finite_batches": "no",
    })
    if args.use_amp != "no" or args.model_amp != "no" or args.allow_tf32 != "no":
        raise ValueError("The metric experiment requires FP32 with AMP and TF32 disabled")
    if args.parallel_mode == "dp":
        raise ValueError("Use none or ddp for metric training")
    latest = Path(args.checkpoint_dir).expanduser() / "checkpoint_latest.pt"
    existing = None
    if latest.exists():
        if args.resume != "yes":
            raise FileExistsError("Existing metric run requires --resume yes")
        existing, _ = read_candidate_checkpoint(latest, QUALITY_CANDIDATE)
        source_name = existing["args"].get("metric_warmstart_checkpoint")
        if not source_name:
            raise ValueError("Existing checkpoint is not a metric-training checkpoint")
        if selected.warmstart and Path(selected.warmstart).expanduser().resolve() != Path(source_name).resolve():
            raise ValueError("Metric warm-start differs from the saved run")
    else:
        if not selected.warmstart:
            raise ValueError("A new metric experiment requires --warmstart with a completed quality checkpoint")
        source_name = selected.warmstart
    source_path = Path(source_name).expanduser().resolve()
    source, _ = read_candidate_checkpoint(source_path, QUALITY_CANDIDATE)
    saved = source["args"]
    if saved.get("quality_objective") != "spectral_mrstft_waveform_v1":
        raise ValueError("Warm-start must be a quality experiment, not another metric run")
    if saved["quality_loss_implementation_sha256"] != quality.sha256_file(Path(__file__).with_name("light_quality_loss.py")):
        raise ValueError("Warm-start quality loss differs from the current implementation")
    explicit = quality._explicit_options(remaining)
    for name in quality.SOURCE_FIELDS:
        if f"--{name.replace('_', '-')}" in explicit and getattr(args, name) != saved[name]:
            raise ValueError(f"Explicit {name} differs from warm-start")
        setattr(args, name, saved[name])
    if (args.sample_rate, args.n_fft, args.hop_length, args.win_length, args.power) != (16000, 512, 160, 320, 0.5):
        raise ValueError("Metric training must retain the current 16kHz frontend")
    for name in quality.QUALITY_FIELDS:
        setattr(args, name, saved[name])
    args.quality_objective = "spectral_mrstft_waveform_metricgan_v1"
    args.quality_loss_implementation_sha256 = quality.sha256_file(Path(__file__).with_name("light_metric_loss.py"))
    args.metric_warmstart_checkpoint = str(source_path)
    args.metric_warmstart_sha256 = quality.sha256_file(source_path)
    args.metric_warmstart_epoch = int(source["epoch"])
    previous = {} if existing is None else existing["args"]
    args.metric_performance = selected.metric_performance or previous.get("metric_performance", "overlap")
    if args.metric_performance == "standard" and args.batch_size != 1:
        raise ValueError("Standard metric runtime requires batch size 1")
    args.metric_weight = selected.metric_weight if selected.metric_weight is not None else previous.get("metric_weight", 0.05)
    args.metric_discriminator_lr = selected.metric_lr if selected.metric_lr is not None else previous.get("metric_discriminator_lr", 1e-4)
    if any(not math.isfinite(value) or value <= 0 for value in (args.metric_weight, args.metric_discriminator_lr)):
        raise ValueError("Metric weight and discriminator LR must be finite and positive")
    args.metric_implementation_sha256 = {
        name: quality.sha256_file(Path(__file__).with_name(name))
        for name in ("light_quality_loss.py", "light_metric_loss.py", "train_light_metric.py", "train_light.py")
    }
    args.metric_discriminator_step_unit = "one globally averaged D update per example per rank; G accumulation unchanged"
    for name, value in candidate_metadata(QUALITY_CANDIDATE).items():
        setattr(args, name, value)
    if existing is not None:
        if "metric_state" not in existing:
            raise ValueError("Metric checkpoint lacks discriminator and optimizer state")
        upgrading = (selected.upgrade_runtime and previous.get("metric_implementation_sha256") == LEGACY_RUNTIME
                     and previous.get("quality_loss_implementation_sha256") == LEGACY_RUNTIME["light_metric_loss.py"]
                     and previous.get("metric_discriminator_step_unit") ==
                     "one globally averaged D update per microbatch; G accumulation unchanged"
                     and previous.get("batch_size") == 1)
        if selected.upgrade_runtime and not upgrading:
            raise ValueError("Runtime upgrade only accepts the known original metric implementation")
        if not upgrading and previous.get("metric_performance", "standard") != args.metric_performance:
            raise ValueError("Metric performance mode differs from checkpoint")
        if upgrading and args.batch_size * args.grad_accum_steps != previous["batch_size"] * previous["grad_accum_steps"]:
            raise ValueError("Runtime upgrade must preserve examples per generator update")
        for name in (*quality.SOURCE_FIELDS, *quality.QUALITY_FIELDS, *METRIC_FIELDS):
            if upgrading and name in ("metric_implementation_sha256", "quality_loss_implementation_sha256",
                                      "metric_discriminator_step_unit"):
                continue
            if previous.get(name) != getattr(args, name):
                raise ValueError(f"Metric resume {name} differs from checkpoint")
        args.metric_runtime_upgrade_from = (dict(epoch=existing["epoch"], implementation=LEGACY_RUNTIME,
            generator_adam_preserved=True, discriminator_adam_preserved=True,
            examples_per_rank_per_generator_update=previous["batch_size"] * previous["grad_accum_steps"],
            discriminator_updates_per_generator_update=previous["batch_size"] * previous["grad_accum_steps"])
            if upgrading else previous.get("metric_runtime_upgrade_from"))
    quality._validate_output_paths(args, source_path)
    return args, source


def initialize_checkpoint(args, source):
    directory, best = Path(args.checkpoint_dir), Path(args.best_dir)
    if (directory / "checkpoint_latest.pt").exists():
        return
    for path in (directory, best):
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"New metric run requires an empty output directory: {path}")
    torch.manual_seed(args.seed)
    with select_model_factory(trainer, QUALITY_CANDIDATE):
        model = trainer.create_model(args, torch.device("cpu"))
    model.load_state_dict(source["model_state_dict"], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    objective = MetricObjective("cpu", weight=args.metric_weight,
                                discriminator_lr=args.metric_discriminator_lr)
    state = dict(epoch=0, best_val_loss=float("inf"), model_state_dict=model.state_dict(),
                 optimizer_state_dict=optimizer.state_dict(), scaler_state_dict={}, lr_reducer_state={},
                 metric_state=objective.state_dict(), args=vars(args).copy())
    trainer.save_checkpoint(directory / "model_epoch_0.pt", state)
    trainer.save_checkpoint(directory / "checkpoint_latest.pt", state)
    print(f"Metric warm-start: {args.metric_warmstart_checkpoint}; fresh G/D Adam; frozen backbone", flush=True)


def main(argv=None):
    args, source = parse_args(argv)
    # Reuse the existing distributed initialization and rank-zero error broadcast.
    original_initializer = quality._initialize_checkpoint
    quality._initialize_checkpoint = initialize_checkpoint
    try:
        quality.prepare_checkpoint(args, source)
    finally:
        quality._initialize_checkpoint = original_initializer
    saved, _ = read_candidate_checkpoint(Path(args.checkpoint_dir) / "checkpoint_latest.pt", QUALITY_CANDIDATE)
    args.resume = "yes"
    objective = None
    original_parser, original_epoch, original_save = trainer.parse_args, trainer.run_epoch, trainer.save_checkpoint

    def metric_epoch(*epoch_args, **epoch_kwargs):
        nonlocal objective
        if objective is None:
            device = epoch_kwargs.get("device", epoch_args[4] if len(epoch_args) > 4 else None)
            objective = MetricObjective(device, weight=args.metric_weight,
                                        discriminator_lr=args.metric_discriminator_lr,
                                        accelerated=args.metric_performance == "overlap")
            objective.load_state_dict(saved["metric_state"])
        objective.training = epoch_kwargs["training"]
        objective.reset_stats()
        objective.reset_metric_stats()
        epoch_kwargs.update(loss_fn=objective, feature_builder=objective.build_features,
                            correct_accumulation=True, loss_weighting="frames",
                            fail_nonfinite_grad=True, forward_with_lengths=True)
        epoch_kwargs["defer_ddp_sync"] = args.metric_performance == "overlap"
        value = original_epoch(*epoch_args, **epoch_kwargs)
        stats = objective.metric_stats()
        if objective.training and stats["valid_pesq_labels"] == 0:
            raise RuntimeError("No valid real PESQ labels in the entire training epoch")
        if trainer.is_main_process():
            phase = "train" if objective.training else "val"
            print(f"[METRIC][{phase}] " + json.dumps(stats, sort_keys=True), flush=True)
        return value

    def save_metric(path, state):
        state = dict(state)
        state["metric_state"] = objective.state_dict() if objective is not None else saved["metric_state"]
        original_save(path, state)

    trainer.parse_args = lambda: args
    trainer.run_epoch = metric_epoch
    trainer.save_checkpoint = save_metric
    try:
        with select_model_factory(trainer, QUALITY_CANDIDATE):
            trainer.main()
    finally:
        trainer.parse_args, trainer.run_epoch, trainer.save_checkpoint = original_parser, original_epoch, original_save
        if objective is not None:
            objective.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

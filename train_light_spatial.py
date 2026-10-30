"""Train the independent SpatialNet residual candidate through the original loop."""

import argparse
from pathlib import Path
import sys

import torch

import train_light as trainer
import train_light_quality as quality
from light_candidate_runtime import candidate_metadata, read_candidate_checkpoint, select_model_factory
from light_quality_loss import QualityObjective
from light_quality_model import QUALITY_CANDIDATE
from light_spatial_model import SPATIAL_CANDIDATE


SPATIAL_FIELDS = ("spatial_training_precision", "spatial_training_implementation_sha256")


def parse_args(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    selector = argparse.ArgumentParser(description=__doc__, add_help=False)
    selector.add_argument("--warmstart")
    selected, remaining = selector.parse_known_args(arguments)
    args = trainer.parse_args(remaining, defaults={
        "checkpoint_dir": f"./checkpoints_{SPATIAL_CANDIDATE}",
        "best_dir": f"./bestmodels_{SPATIAL_CANDIDATE}", "log_dir": f"./logs_{SPATIAL_CANDIDATE}",
        "num_epochs": 10, "learning_rate": 0.001, "batch_size": 1,
        "grad_accum_steps": 4, "segment_seconds": 6, "save_every": 1,
        "train_lr_patience": 3, "grad_clip": 5.0, "use_amp": "no", "model_amp": "no",
        "allow_tf32": "no", "stop_on_non_finite": "yes", "skip_non_finite_batches": "no",
    })
    if args.use_amp != "no" or args.model_amp != "no" or args.allow_tf32 != "no":
        raise ValueError("Use the model's CUDA bfloat16 path with FP32 frontend/loss; generic AMP/TF32 must be off")
    if args.parallel_mode == "dp":
        raise ValueError("Spatial training supports none or ddp")
    latest = Path(args.checkpoint_dir).expanduser() / "checkpoint_latest.pt"
    existing = None
    if latest.exists():
        if args.resume != "yes":
            raise FileExistsError("Existing spatial run requires --resume yes")
        existing, _ = read_candidate_checkpoint(latest, SPATIAL_CANDIDATE)
        source_name = existing["args"]["quality_source_checkpoint"]
        if selected.warmstart and Path(selected.warmstart).expanduser().resolve() != Path(source_name).resolve():
            raise ValueError("Spatial warm-start differs from checkpoint")
    else:
        if not selected.warmstart:
            raise ValueError("A new spatial run requires --warmstart with a completed quality checkpoint")
        source_name = selected.warmstart
    source_path = Path(source_name).expanduser().resolve()
    source, _ = read_candidate_checkpoint(source_path, QUALITY_CANDIDATE)
    saved = source["args"]
    explicit = quality._explicit_options(remaining)
    for name in quality.SOURCE_FIELDS:
        if f"--{name.replace('_', '-')}" in explicit and getattr(args, name) != saved[name]:
            raise ValueError(f"Explicit {name} differs from source")
        setattr(args, name, saved[name])
    if (args.sample_rate, args.n_fft, args.hop_length, args.win_length, args.power) != (16000, 512, 160, 320, 0.5):
        raise ValueError("Spatial candidate requires the existing 16kHz frontend")
    args.quality_source_checkpoint = str(source_path)
    args.quality_source_sha256 = quality.sha256_file(source_path)
    args.quality_source_epoch = int(source["epoch"])
    args.quality_base_config = {name: saved[name] for name in quality.SOURCE_FIELDS}
    args.quality_objective = "spectral_mrstft_waveform_v1"
    args.quality_loss_implementation_sha256 = quality.sha256_file(Path(__file__).with_name("light_quality_loss.py"))
    args.quality_refiner_causal = False
    args.spatial_training_precision = "CUDA bfloat16 refiner activations; FP32 frozen base, frontend, loss, parameters and inference; CPU FP32"
    args.spatial_training_implementation_sha256 = {name: quality.sha256_file(Path(__file__).with_name(name))
        for name in ("train_light_spatial.py", "train_light.py", "light_quality_loss.py")}
    for name, value in candidate_metadata(SPATIAL_CANDIDATE).items():
        setattr(args, name, value)
    if existing is not None:
        for name in (*quality.SOURCE_FIELDS, *quality.QUALITY_FIELDS, *SPATIAL_FIELDS):
            if existing["args"].get(name) != getattr(args, name):
                raise ValueError(f"Spatial resume {name} differs from checkpoint")
    quality._validate_output_paths(args, source_path)
    return args, source


def initialize_checkpoint(args, source):
    directory, best = Path(args.checkpoint_dir), Path(args.best_dir)
    if (directory / "checkpoint_latest.pt").exists():
        return
    for path in (directory, best):
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"New spatial run requires empty outputs: {path}")
    torch.manual_seed(args.seed)
    with select_model_factory(trainer, SPATIAL_CANDIDATE):
        model = trainer.create_model(args, torch.device("cpu"))
    model.base.load_state_dict(source["model_state_dict"], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    state = dict(epoch=0, best_val_loss=float("inf"), model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(), scaler_state_dict={}, lr_reducer_state={}, args=vars(args).copy())
    trainer.save_checkpoint(directory / "model_epoch_0.pt", state)
    trainer.save_checkpoint(directory / "checkpoint_latest.pt", state)
    print(f"Spatial warm-start: {args.quality_source_checkpoint}; fresh Adam; frozen source", flush=True)


def main(argv=None):
    args, source = parse_args(argv)
    if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This spatial training candidate requires CUDA bfloat16 support")
    initializer = quality._initialize_checkpoint
    quality._initialize_checkpoint = initialize_checkpoint
    try:
        quality.prepare_checkpoint(args, source)
    finally:
        quality._initialize_checkpoint = initializer
    if trainer.is_main_process():
        print(f"Spatial precision: {args.spatial_training_precision}", flush=True)
    args.resume = "yes"
    objective = QualityObjective(n_fft=args.n_fft, hop_length=args.hop_length,
                                 win_length=args.win_length, power=args.power)
    parser, epoch = trainer.parse_args, trainer.run_epoch

    def spatial_epoch(*epoch_args, **epoch_kwargs):
        objective.reset_stats()
        epoch_kwargs.update(loss_fn=objective, feature_builder=objective.build_features,
            correct_accumulation=True, loss_weighting="frames", fail_nonfinite_grad=True,
            forward_with_lengths=True, defer_ddp_sync=True)
        return epoch(*epoch_args, **epoch_kwargs)

    trainer.parse_args = lambda: args
    trainer.run_epoch = spatial_epoch
    try:
        with select_model_factory(trainer, SPATIAL_CANDIDATE):
            trainer.main()
    finally:
        trainer.parse_args, trainer.run_epoch = parser, epoch
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

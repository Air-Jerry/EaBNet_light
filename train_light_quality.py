"""Train a frozen epoch-40 backbone and a quality refiner through the original loop."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

import train_light as trainer
from light_candidate_runtime import candidate_metadata, read_candidate_checkpoint, select_model_factory
from light_quality_loss import QualityObjective
from light_quality_model import BASE_CANDIDATE, QUALITY_CANDIDATE


SOURCE_FIELDS = (
    "sample_rate", "n_fft", "hop_length", "win_length", "power", "target_ref_mic",
    "channels", "num_mics", "embed_dim", "kd1", "cd1", "d_feat", "p", "q", "bf_type",
    "topo_type", "intra_connect", "norm_type", "dfsmn_layers", "dfsmn_memory_size",
    "is_causal", "is_u2",
)
QUALITY_FIELDS = (
    "quality_source_checkpoint", "quality_source_sha256", "quality_source_epoch",
    "quality_base_config", "quality_objective", "quality_loss_implementation_sha256",
    "quality_refiner_causal",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _explicit_options(arguments):
    return {argument.split("=", 1)[0] for argument in arguments if argument.startswith("--")}


def _validate_output_paths(args, source_path):
    protected = (source_path.parent, Path(args.train_dir).expanduser().resolve(),
                 Path(args.val_dir).expanduser().resolve())
    outputs = tuple(Path(getattr(args, name)).expanduser().resolve()
                    for name in ("checkpoint_dir", "best_dir", "log_dir"))
    for output in outputs:
        for original in protected:
            if output == original or output in original.parents or original in output.parents:
                raise ValueError(f"Output directory overlaps source checkpoint or dataset: {output}")
    for index, output in enumerate(outputs):
        for other in outputs[index + 1:]:
            if output == other or output in other.parents or other in output.parents:
                raise ValueError("Quality checkpoint, best and log directories must be separate")


def parse_args(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    selector = argparse.ArgumentParser(description=__doc__, add_help=False)
    selector.add_argument("--warmstart", help="Original epoch-40 checkpoint; required for a new run")
    selected, remaining = selector.parse_known_args(arguments)
    args = trainer.parse_args(remaining, defaults={
        "checkpoint_dir": f"./checkpoints_{QUALITY_CANDIDATE}",
        "best_dir": f"./bestmodels_{QUALITY_CANDIDATE}", "log_dir": f"./logs_{QUALITY_CANDIDATE}",
        "num_epochs": 10, "learning_rate": 0.0003, "batch_size": 1,
        "grad_accum_steps": 4, "segment_seconds": 6, "save_every": 1,
        "train_lr_patience": 3, "use_amp": "no", "model_amp": "no", "allow_tf32": "no",
    })
    latest = Path(args.checkpoint_dir).expanduser() / "checkpoint_latest.pt"
    existing = None
    if latest.exists():
        if args.resume != "yes":
            raise FileExistsError("A quality checkpoint already exists; use --resume yes or new output directories")
        existing, _ = read_candidate_checkpoint(latest, QUALITY_CANDIDATE)
        source_name = existing["args"].get("quality_source_checkpoint")
        if not source_name:
            raise ValueError("Quality checkpoint lacks its source checkpoint path")
        if selected.warmstart and Path(selected.warmstart).expanduser().resolve() != Path(source_name).resolve():
            raise ValueError("--warmstart differs from the saved quality source checkpoint")
    else:
        if not selected.warmstart:
            raise ValueError("A new quality run requires --warmstart with the original epoch-40 checkpoint")
        source_name = selected.warmstart
    source_path = Path(source_name).expanduser().resolve()
    source, _ = read_candidate_checkpoint(source_path, BASE_CANDIDATE)
    if source.get("epoch") != 40:
        raise ValueError("Quality warm-start requires the original epoch-40 checkpoint")
    source_config = source["args"]
    explicit = _explicit_options(remaining)
    for name in SOURCE_FIELDS:
        if name not in source_config:
            raise ValueError(f"Source checkpoint lacks required model/frontend field: {name}")
        if f"--{name.replace('_', '-')}" in explicit and getattr(args, name) != source_config[name]:
            raise ValueError(f"Explicit {name} differs from the source checkpoint")
        setattr(args, name, source_config[name])
    args.quality_source_checkpoint = str(source_path)
    args.quality_source_sha256 = sha256_file(source_path)
    args.quality_source_epoch = 40
    args.quality_base_config = {name: source_config[name] for name in SOURCE_FIELDS}
    args.quality_objective = "spectral_mrstft_waveform_v1"
    args.quality_loss_implementation_sha256 = sha256_file(Path(__file__).with_name("light_quality_loss.py"))
    args.quality_refiner_causal = False
    for name, value in candidate_metadata(QUALITY_CANDIDATE).items():
        setattr(args, name, value)
    if existing is not None:
        saved_checkpoints = [existing]
        for saved_path in (latest.parent / "model_epoch_0.pt", latest.parent / "best_model.pt",
                           Path(args.best_dir).expanduser() / "best_model.pt"):
            if saved_path.exists():
                saved_checkpoints.append(read_candidate_checkpoint(saved_path, QUALITY_CANDIDATE)[0])
        for checkpoint in saved_checkpoints:
            for name in (*SOURCE_FIELDS, *QUALITY_FIELDS):
                if checkpoint["args"].get(name) != getattr(args, name):
                    raise ValueError(f"Quality resume {name} differs from checkpoint")
    _validate_output_paths(args, source_path)
    return args, source


def _initialize_checkpoint(args, source):
    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    best_dir = Path(args.best_dir).expanduser()
    if (checkpoint_dir / "checkpoint_latest.pt").exists():
        return
    for directory in (checkpoint_dir, best_dir):
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"A new quality run requires an empty output directory: {directory}")
    torch.manual_seed(args.seed)
    with select_model_factory(trainer, QUALITY_CANDIDATE):
        model = trainer.create_model(args, torch.device("cpu"))
    stripped = {}
    for name, value in source["model_state_dict"].items():
        key = name[len("module."):] if name.startswith("module.") else name
        if key in stripped:
            raise ValueError(f"Duplicate base parameter after removing module prefix: {key}")
        stripped[key] = value
    model.base.load_state_dict(stripped, strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    state = {
        "epoch": 0, "best_val_loss": float("inf"), "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": {}, "lr_reducer_state": {},
        "args": vars(args).copy(),
    }
    trainer.save_checkpoint(checkpoint_dir / "model_epoch_0.pt", state)
    trainer.save_checkpoint(checkpoint_dir / "checkpoint_latest.pt", state)
    print(f"Quality warm-start: {args.quality_source_checkpoint}; fresh Adam; frozen backbone", flush=True)


def prepare_checkpoint(args, source):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1 and args.parallel_mode in ("auto", "ddp")
    if use_ddp and not dist.is_initialized():
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", init_method="env://")
    if not trainer.is_distributed():
        _initialize_checkpoint(args, source)
        return
    outcome = [None]
    if trainer.is_main_process():
        try:
            _initialize_checkpoint(args, source)
        except Exception as error:
            outcome[0] = f"{type(error).__name__}: {error}"
    dist.broadcast_object_list(outcome, src=0)
    if outcome[0] is not None:
        raise RuntimeError(outcome[0])


def main(argv=None):
    args, source = parse_args(argv)
    prepare_checkpoint(args, source)
    args.resume = "yes"
    objective = QualityObjective(n_fft=args.n_fft, hop_length=args.hop_length,
                                 win_length=args.win_length, power=args.power)
    original_parser, original_epoch = trainer.parse_args, trainer.run_epoch

    def quality_epoch(*epoch_args, **epoch_kwargs):
        objective.reset_stats()
        epoch_kwargs.update(loss_fn=objective, feature_builder=objective.build_features,
                            correct_accumulation=True, loss_weighting="frames",
                            fail_nonfinite_grad=True, forward_with_lengths=True)
        value = original_epoch(*epoch_args, **epoch_kwargs)
        if trainer.is_main_process():
            phase = "train" if epoch_kwargs["training"] else "val"
            print(f"[QUALITY][{phase}][rank0] " + json.dumps(objective.stats(), sort_keys=True), flush=True)
        return value

    trainer.parse_args = lambda: args
    trainer.run_epoch = quality_epoch
    try:
        with select_model_factory(trainer, QUALITY_CANDIDATE):
            trainer.main()
    finally:
        trainer.parse_args, trainer.run_epoch = original_parser, original_epoch
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

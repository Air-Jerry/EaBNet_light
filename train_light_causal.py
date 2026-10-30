"""Train the causal spatial network end to end through the existing data/loop."""

from pathlib import Path
import sys

import torch

import train_light as trainer
import train_light_quality as quality
from light_candidate_runtime import candidate_metadata, read_candidate_checkpoint, select_model_factory
from light_causal_spatial_model import CAUSAL_CANDIDATE
from light_quality_loss import QualityObjective


def parse_args(argv=None, candidate_id=CAUSAL_CANDIDATE, defaults=None, entry_file=None):
    settings = {
        "checkpoint_dir": f"./checkpoints_{candidate_id}",
        "best_dir": f"./bestmodels_{candidate_id}", "log_dir": f"./logs_{candidate_id}",
        "num_epochs": 30, "learning_rate": 0.001, "batch_size": 1, "grad_accum_steps": 4,
        "segment_seconds": 6, "save_every": 1, "train_lr_patience": 3, "grad_clip": 5.0,
        "norm_type": "LN", "use_amp": "no", "model_amp": "no", "allow_tf32": "no",
        "stop_on_non_finite": "yes", "skip_non_finite_batches": "no",
    }
    settings.update(defaults or {})
    args = trainer.parse_args(sys.argv[1:] if argv is None else argv, defaults=settings)
    if (args.sample_rate, args.n_fft, args.hop_length, args.win_length, args.power, args.num_mics) != (
            16000, 512, 160, 320, 0.5, 8):
        raise ValueError("Keep the existing 16kHz eight-microphone frontend")
    if args.is_causal != "yes" or args.norm_type != "LN" or args.parallel_mode == "dp":
        raise ValueError("Use the causal LayerNorm network with none or ddp")
    if any(value != "no" for value in (args.use_amp, args.model_amp, args.allow_tf32)):
        raise ValueError("Use internal CUDA bfloat16 activations with FP32 frontend/loss/inference")
    args.causal_training_objective = "spectral_mrstft_waveform_v1"
    args.causal_training_precision = "CUDA bfloat16 activations; FP32 parameters, frontend, loss and inference; CPU FP32"
    args.causal_training_implementation_sha256 = {name: quality.sha256_file(Path(__file__).with_name(name))
        for name in ("train_light_causal.py", "train_light.py", "light_quality_loss.py")}
    if entry_file is not None:
        args.causal_training_implementation_sha256[Path(entry_file).name] = quality.sha256_file(entry_file)
    for name, value in candidate_metadata(candidate_id).items():
        setattr(args, name, value)
    latest = Path(args.checkpoint_dir).expanduser() / "checkpoint_latest.pt"
    if latest.exists():
        if args.resume != "yes":
            raise FileExistsError("Use --resume yes or a new output directory")
        state, _ = read_candidate_checkpoint(latest, candidate_id, args)
        for name in ("target_ref_mic", "causal_training_objective", "causal_training_precision",
                     "causal_training_implementation_sha256"):
            if state["args"].get(name) != getattr(args, name):
                raise ValueError(f"Causal resume {name} differs from checkpoint")
    outputs = [Path(getattr(args, name)).expanduser().resolve() for name in ("checkpoint_dir", "best_dir", "log_dir")]
    datasets = [Path(value).expanduser().resolve() for value in (args.train_dir, args.val_dir)]
    for index, output in enumerate(outputs):
        for other in [*datasets, *outputs[index + 1:]]:
            if output == other or output in other.parents or other in output.parents:
                raise ValueError("Output directories must be separate and cannot overlap datasets")
    return args


def initialize_checkpoint(args, unused=None):
    directory, best = Path(args.checkpoint_dir), Path(args.best_dir)
    if (directory / "checkpoint_latest.pt").exists():
        return
    for path in (directory, best):
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"New causal run requires empty outputs: {path}")
    torch.manual_seed(args.seed)
    with select_model_factory(trainer, args.structure_candidate):
        model = trainer.create_model(args, torch.device("cpu"))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    state = dict(epoch=0, best_val_loss=float("inf"), model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(), scaler_state_dict={}, lr_reducer_state={}, args=vars(args).copy())
    trainer.save_checkpoint(directory / "model_epoch_0.pt", state)
    trainer.save_checkpoint(directory / "checkpoint_latest.pt", state)


def main(argv=None, candidate_id=CAUSAL_CANDIDATE, defaults=None, entry_file=None):
    args = parse_args(argv, candidate_id, defaults, entry_file)
    if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA bfloat16 support is required for this training configuration")
    initializer = quality._initialize_checkpoint
    quality._initialize_checkpoint = initialize_checkpoint
    try:
        quality.prepare_checkpoint(args, None)
    finally:
        quality._initialize_checkpoint = initializer
    args.resume = "yes"
    objective = QualityObjective(n_fft=args.n_fft, hop_length=args.hop_length,
                                 win_length=args.win_length, power=args.power)
    parser, epoch = trainer.parse_args, trainer.run_epoch

    def causal_epoch(*epoch_args, **epoch_kwargs):
        objective.reset_stats()
        epoch_kwargs.update(loss_fn=objective, feature_builder=objective.build_features,
            correct_accumulation=True, loss_weighting="frames", fail_nonfinite_grad=True,
            forward_with_lengths=True, defer_ddp_sync=True)
        return epoch(*epoch_args, **epoch_kwargs)

    trainer.parse_args = lambda: args
    trainer.run_epoch = causal_epoch
    try:
        with select_model_factory(trainer, args.structure_candidate):
            trainer.main()
    finally:
        trainer.parse_args, trainer.run_epoch = parser, epoch
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

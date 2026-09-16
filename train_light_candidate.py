"""Train an explicitly named hypothesis through unchanged train_light.main."""

import argparse
from pathlib import Path

import train_light as trainer
from light_candidate_runtime import candidate_metadata, read_candidate_checkpoint, select_model_factory
from light_reference_variants import all_reference_candidates


def main(argv=None):
    selector = argparse.ArgumentParser(description=__doc__, add_help=False)
    selector.add_argument("--candidate", choices=all_reference_candidates(), required=True)
    selected, remaining = selector.parse_known_args(argv)
    candidate_id = selected.candidate
    args = trainer.parse_args(remaining, defaults={
        "checkpoint_dir": f"./checkpoints_{candidate_id}",
        "best_dir": f"./bestmodels_{candidate_id}", "log_dir": f"./logs_{candidate_id}",
    })
    for name, value in candidate_metadata(candidate_id).items():
        setattr(args, name, value)
    latest = Path(args.checkpoint_dir).expanduser() / "checkpoint_latest.pt"
    if args.resume == "yes" and latest.exists():
        read_candidate_checkpoint(latest, candidate_id, args)
    # Enforce separate architecture identity even when the user disables resume.
    for existing in (latest, Path(args.best_dir).expanduser() / "best_model.pt",
                     Path(args.checkpoint_dir).expanduser() / "best_model.pt"):
        if existing.exists():
            read_candidate_checkpoint(existing, candidate_id)
    original_parser = trainer.parse_args
    trainer.parse_args = lambda: args
    print(f"Structure hypothesis: {candidate_id}; author equivalence is unverified.", flush=True)
    try:
        with select_model_factory(trainer, candidate_id):
            trainer.main()
    finally:
        trainer.parse_args = original_parser
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

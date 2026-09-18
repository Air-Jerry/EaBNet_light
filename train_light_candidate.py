"""Train an explicitly named hypothesis through unchanged train_light.main."""

'''
查看日志
tail -f ./logs_cbam_flat_projection64/full_train_2gpu.log
'''

''''
TRAIN_PY="$HOME/miniconda3/envs/EaBNet/bin/python"
TRAIN_DIR="/data/ssd1/jinrui.yang/training_set"
VAL_DIR="/data/ssd1/jinrui.yang/validation_set"

mkdir -p ./logs_cbam_flat_projection64

CUDA_VISIBLE_DEVICES=0,1 nohup "$HOME/miniconda3/envs/EaBNet/bin/python" -u \
  -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  train_light_candidate.py \
  --candidate cbam_flat_projection64 \
  --train-dir /data/ssd1/jinrui.yang/training_set \
  --val-dir /data/ssd1/jinrui.yang/validation_set \
  --checkpoint-dir ./checkpoints_cbam_flat_projection64 \
  --best-dir ./bestmodels_cbam_flat_projection64 \
  --log-dir ./logs_cbam_flat_projection64 \
  --num-epochs 30 \
  --resume yes \
  --resume-reset-lr no \
  --parallel-mode ddp \
  --batch-size 4 \
  --grad-accum-steps 1 \
  --num-workers 4 \
  --segment-seconds 6 \
  --learning-rate 0.001 \
  --lr-reduce-metric val \
  --train-lr-patience 2 \
  --train-lr-factor 0.5 \
  --train-lr-min-delta 0 \
  --use-amp yes \
  --model-amp yes \
  --allow-tf32 yes \
  --stop-on-non-finite yes \
  --save-every 5 \
  --strict-memory no \
  > ./logs_cbam_flat_projection64/full_train_2gpu.log 2>&1 &

'''


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

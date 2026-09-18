"""Evaluate a candidate checkpoint with the existing strict four-metric path."""

'''
CUDA_VISIBLE_DEVICES=0 nohup "$HOME/miniconda3/envs/EaBNet/bin/python" -u \
  evaluate_light_candidate.py \
  --candidate cbam_flat_projection64 \
  --val-dir /data/ssd1/jinrui.yang/validation_set \
  --checkpoint ./bestmodels_cbam_flat_projection64/best_model.pt \
  --device cuda \
  --max-samples 0 \
  --save-csv ./logs_cbam_flat_projection64/val_best_epoch30.csv \
  --save-json ./logs_cbam_flat_projection64/val_best_epoch30.json \
  > ./logs_cbam_flat_projection64/eval_best_epoch30.log 2>&1 &
'''

import argparse

import evaluate_light as legacy
import evaluate_light_paper as paper
from light_candidate_runtime import read_candidate_checkpoint, select_model_factory
from light_reference_variants import all_reference_candidates


def load_candidate_model(checkpoint_path, device, expected_candidate=None):
    _, candidate_id = read_candidate_checkpoint(checkpoint_path, expected_candidate)
    with select_model_factory(legacy, candidate_id):
        return legacy.load_model(checkpoint_path, device)


def main(argv=None):
    selector = argparse.ArgumentParser(description=__doc__, add_help=False)
    selector.add_argument("--candidate", choices=all_reference_candidates())
    selected, remaining = selector.parse_known_args(argv)
    args = paper.parse_args(remaining)
    original_loader = paper.load_model
    paper.load_model = lambda path, device: load_candidate_model(path, device, selected.candidate)
    try:
        return paper.evaluate(args)
    finally:
        paper.load_model = original_loader


if __name__ == "__main__":
    main()

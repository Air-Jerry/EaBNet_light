"""Train full-frequency causal matrix memory with compiled training LayerNorm."""

from experiments.compiled_training_norm import compiled_training_norms
from experiments.compiled_training_retention import compiled_training_retention
from light_candidate_runtime import RETENTION_CANDIDATE
from train_light_causal_frequency_attention import main


if __name__ == '__main__':
    with compiled_training_norms(), compiled_training_retention():
        raise SystemExit(main(candidate_id=RETENTION_CANDIDATE, entry_file=__file__,
            training_files=('experiments/compiled_training_norm.py', 'experiments/compiled_training_retention.py')))

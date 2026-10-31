"""Full257-frequency attention with compiled training LayerNorm; original FP32 inference."""

from experiments.compiled_training_norm import compiled_training_norms
from train_light_causal_frequency_attention import main


if __name__ == '__main__':
    with compiled_training_norms():
        raise SystemExit(main(entry_file=__file__, training_files=('experiments/compiled_training_norm.py',)))

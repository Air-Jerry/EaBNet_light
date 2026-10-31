"""Train the257-query/129-key causal frequency-attention candidate."""

from light_candidate_runtime import POOLED_FREQUENCY_ATTENTION_CANDIDATE
from train_light_causal_frequency_attention import main


if __name__ == '__main__':
    raise SystemExit(main(candidate_id=POOLED_FREQUENCY_ATTENTION_CANDIDATE, entry_file=__file__))

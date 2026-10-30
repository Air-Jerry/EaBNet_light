"""Train the multi-resolution causal model with the original full dataset/loop."""

from light_causal_fast_model import FAST_CANDIDATE
from train_light_causal import main as train_main


def main(argv=None):
    return train_main(argv, candidate_id=FAST_CANDIDATE,
        defaults={"batch_size": 4, "grad_accum_steps": 1}, entry_file=__file__)


if __name__ == "__main__":
    raise SystemExit(main())

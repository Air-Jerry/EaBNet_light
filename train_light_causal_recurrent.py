"""Train the full-frequency recurrent model through the unchanged causal workflow."""

from light_causal_recurrent_model import RECURRENT_CANDIDATE
from train_light_causal import main as causal_main


def main(argv=None):
    return causal_main(argv, candidate_id=RECURRENT_CANDIDATE,
        defaults={'batch_size': 4, 'grad_accum_steps': 1}, entry_file=__file__)


if __name__ == '__main__':
    raise SystemExit(main())

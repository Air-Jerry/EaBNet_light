"""Train the 65-group recurrent spatial model through the unchanged causal workflow."""

from light_causal_recurrent65_model import RECURRENT65_CANDIDATE
from train_light_causal import main as causal_main


def main(argv=None):
    return causal_main(argv, candidate_id=RECURRENT65_CANDIDATE,
        defaults={'batch_size': 4, 'grad_accum_steps': 1}, entry_file=__file__)


if __name__ == '__main__':
    raise SystemExit(main())

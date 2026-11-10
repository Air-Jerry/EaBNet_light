#!/usr/bin/env python3
"""Forward launcher Python calls; select the declared scaled entry for recurrent DDP."""
import os
from pathlib import Path
import sys


def training_arguments(arguments, entry):
    result = list(arguments)
    if '-m' in result and result[result.index('-m') + 1:][:1] == ['torch.distributed.run']:
        positions = [i for i, value in enumerate(result)
                     if Path(value).name == 'train_light_causal_recurrent_adapt.py']
        if len(positions) != 1:
            raise ValueError('Use exactly one original recurrent training entry')
        result[positions[0]] = str(entry)
    return result


if __name__ == '__main__':
    interpreter = os.environ['CAUSAL_REAL_PYTHON']
    arguments = training_arguments(sys.argv[1:], Path(__file__).with_name('scaled_recurrent_entry.py'))
    os.execv(interpreter, [interpreter, *arguments])


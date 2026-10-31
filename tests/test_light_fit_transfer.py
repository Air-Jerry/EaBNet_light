"""Hold-out selection is fixed before scoring and excludes exact fitted targets."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

import experiments.diagnose_fit_transfer as diagnostic


def test_fixed_snr_strata_and_decoded_target_exclusion(monkeypatch):
    rows = [dict(sample_id=str(index), snr_db=str((-5, -2, 0, 2)[index % 4]),
                 target_path=f'{index}.wav', mixture_path=f'mix_{index}.wav') for index in range(48)]
    monkeypatch.setattr(diagnostic, 'resolve_record_path', lambda directory, name: directory / name)
    monkeypatch.setattr(diagnostic, 'target_data', lambda path: np.array([int(path.stem)], dtype='float32'))
    excluded = {hashlib.sha256(np.array([index], dtype='float32').tobytes()).hexdigest() for index in range(4)}
    first = diagnostic.select_records(rows, Path('/fake'), excluded)
    second = diagnostic.select_records(rows, Path('/fake'), excluded)
    assert first == second
    assert len(first) == len({record[0] for record in first}) == 32
    assert not {record[0] for record in first}.intersection({'0','1','2','3'})
    assert all(sum(record[1] == snr for record in first) == 8 for snr in (-5, -2, 0, 2))
    with pytest.raises(ValueError):diagnostic.select_records(rows[:12], Path('/fake'), excluded)

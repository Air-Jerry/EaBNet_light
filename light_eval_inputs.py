"""Resolve evaluation-only audio pairs without changing the training data interface.

The original ``--val-dir``/``metadata.csv`` path remains available.  Direct
inputs can instead name one explicit mixture/reference pair or two directory
trees with matching relative stems.  Resolution never modifies audio files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from evaluate_light import SampleRecord, load_records


_AUDIO_EXTENSIONS = frozenset({".wav", ".flac"})


def _examples(values: list[str] | set[str]) -> str:
    """Keep malformed-dataset diagnostics short even for a large input tree."""
    ordered = sorted(values)
    result = ", ".join(repr(value) for value in ordered[:5])
    if len(ordered) > 5:
        result += f", ... ({len(ordered)} total)"
    return result


def _audio_index(root: Path, suffix: str, label: str) -> dict[str, Path]:
    """Index WAV/FLAC files by relative path after removing a stem suffix.

    Extensions are case insensitive; directory names and stems are case
    sensitive. Case-fold collisions are rejected so pairing does not change
    when the same dataset is moved between Windows and Linux.
    """
    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in _AUDIO_EXTENSIONS
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not files:
        raise ValueError(f"No WAV or FLAC files found in {label} directory: {root}")

    mismatch: list[str] = []
    empty_stems: list[str] = []
    duplicates: list[str] = []
    index: dict[str, Path] = {}
    folded: dict[str, str] = {}
    for path in files:
        relative = path.relative_to(root)
        stem = path.stem
        if suffix:
            if not stem.endswith(suffix):
                mismatch.append(relative.as_posix())
                continue
            stem = stem[: -len(suffix)]
        if not stem:
            empty_stems.append(relative.as_posix())
            continue
        key = (relative.parent / stem).as_posix()
        folded_key = key.casefold()
        if key in index:
            duplicates.append(key)
            continue
        if folded_key in folded:
            duplicates.append(f"{folded[folded_key]} / {key} (case collision)")
            continue
        index[key] = path.resolve()
        folded[folded_key] = key

    if mismatch:
        raise ValueError(
            f"Every WAV/FLAC filename stem in {label} must end with suffix "
            f"{suffix!r}; mismatches: {_examples(mismatch)}"
        )
    if empty_stems:
        raise ValueError(
            f"Removing the {label} suffix leaves an empty filename stem: "
            f"{_examples(empty_stems)}"
        )
    if duplicates:
        raise ValueError(
            f"Duplicate or case-colliding pairing keys in {label}: "
            f"{_examples(duplicates)}"
        )
    return index


def pairs_sha256(records: list[SampleRecord]) -> str:
    """Hash the resolved pair manifest, not the audio contents."""
    manifest = [
        {
            "sample_id": record.sample_id,
            "mixture_path": str(Path(record.mixture_path).resolve()),
            "target_path": str(Path(record.target_path).resolve()),
        }
        for record in records
    ]
    encoded = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_evaluation_inputs(
    args: Any,
) -> tuple[list[SampleRecord], dict[str, Any], set[Path]]:
    """Resolve metadata, one file pair, or strictly paired directory trees.

    ``args`` accepts ``val_dir``, ``mixture_path``, ``target_path``,
    ``mixture_suffix`` and ``target_suffix``. Missing optional attributes have
    their CLI defaults so older callers using a Namespace remain compatible.
    The returned protected paths cover metadata; callers must also protect
    the audio paths present in the records from output-file overwrites.
    """
    val_dir = getattr(args, "val_dir", None)
    mixture_arg = getattr(args, "mixture_path", None)
    target_arg = getattr(args, "target_path", None)
    mixture_suffix = getattr(args, "mixture_suffix", "") or ""
    target_suffix = getattr(args, "target_suffix", "") or ""
    if val_dir:
        if mixture_arg or target_arg or mixture_suffix or target_suffix:
            raise ValueError(
                "Use either --val-dir or --mixture-path/--target-path; "
                "direct-input suffixes require directory inputs."
            )
        dataset = Path(val_dir).expanduser().resolve()
        records = load_records(dataset)
        metadata = (dataset / "metadata.csv").resolve()
        return records, {
            "mode": "metadata",
            "dataset": str(dataset),
            "metadata_path": str(metadata),
            "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
            "pairing": "metadata.csv mixture_path/target_path rows in original order",
            "resolved_pairs_sha256": pairs_sha256(records),
            "resolved_pairs_sha256_scope": "ordered sample IDs and resolved paths; not audio contents",
        }, {metadata}

    if not mixture_arg or not target_arg:
        raise ValueError(
            "Specify --val-dir or provide both --mixture-path and --target-path."
        )
    mixture_path = Path(mixture_arg).expanduser().resolve()
    target_path = Path(target_arg).expanduser().resolve()
    for label, path in (("mixture", mixture_path), ("target", target_path)):
        if not path.exists():
            raise FileNotFoundError(f"The {label} path does not exist: {path}")

    if mixture_path.is_file() and target_path.is_file():
        if mixture_suffix or target_suffix:
            raise ValueError("--mixture-suffix/--target-suffix require directory inputs.")
        for label, path in (("mixture", mixture_path), ("target", target_path)):
            if path.suffix.lower() not in _AUDIO_EXTENSIONS:
                raise ValueError(f"The {label} file must be WAV or FLAC: {path}")
        records = [SampleRecord(1, mixture_path, target_path)]
        mode = "files"
        pairing = "One explicitly selected mixture and target file; names may differ"
    elif mixture_path.is_dir() and target_path.is_dir():
        mixtures = _audio_index(mixture_path, mixture_suffix, "mixture")
        targets = _audio_index(target_path, target_suffix, "target")
        missing_targets = set(mixtures).difference(targets)
        missing_mixtures = set(targets).difference(mixtures)
        if missing_targets or missing_mixtures:
            diagnostics: list[str] = []
            if missing_targets:
                diagnostics.append(f"missing targets for: {_examples(missing_targets)}")
            if missing_mixtures:
                diagnostics.append(f"missing mixtures for: {_examples(missing_mixtures)}")
            raise ValueError("Unmatched audio pairs; " + "; ".join(diagnostics))
        records = [
            SampleRecord(sample_id, mixtures[key], targets[key])
            for sample_id, key in enumerate(sorted(mixtures), start=1)
        ]
        mode = "directories"
        pairing = (
            "Recursive case-sensitive relative path plus filename stem after removing "
            "the configured suffix; WAV/FLAC extensions ignored; sorted by pairing key"
        )
    else:
        raise ValueError(
            "--mixture-path and --target-path must both be files or both be directories."
        )

    return records, {
        "mode": mode,
        "mixture_path": str(mixture_path),
        "target_path": str(target_path),
        "mixture_suffix": mixture_suffix,
        "target_suffix": target_suffix,
        "pairing": pairing,
        "metadata_path": None,
        "metadata_sha256": None,
        "resolved_pairs_sha256": pairs_sha256(records),
        "resolved_pairs_sha256_scope": "ordered sample IDs and resolved paths; not audio contents",
    }, set()

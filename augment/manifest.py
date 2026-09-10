"""Reading and writing the augmentation manifest.

The manifest is the record that lets any ``aug(N)`` file be traced back to *exactly*
which transforms (and parameters) produced it - the deliverable the user asked for.
It is a single ``.csv``:

* one row per file the run accounts for: ``aug_id, source_file, output_file, plot, n_methods``
  and one column per augmentation method holding the drawn parameters (blank if that method was
  not applied). The originals appear as ``aug_id = 0`` with no methods (no file is written for
  them — the canonical ``plot(N).laz`` already exists); each plot's augmented copies are
  ``aug_id = 1..k`` (per-plot numbering), so ``(source_file, aug_id)`` identifies a row.
* the run's **provenance** (selected methods, their ranges, the methods-per-sample and
  copies-per-plot counts, the seed and the height channel) is written as leading ``#`` comment
  lines, so the single CSV both lists the files *and* records how it was made. CSV readers that
  pass ``comment="#"`` skip those lines.

It also provides :func:`load_previous_output`, used when reusing a previous output folder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from common.naming import LAS_SUFFIXES
from .plan import AppliedAug, AugmentConfig
from .transforms import AUGMENTATIONS

MANIFEST_NAME = "Data augmentation.csv"


@dataclass
class ManifestRow:
    """One written file's provenance (an original copy ``aug_id=0`` or an augmented copy)."""

    aug_id: int                # 0 for the verbatim original copy; 1..k per plot for augmented
    source_file: str           # the input plot file this row was derived from
    output_file: str           # the file actually written to the output folder
    plot: int | None
    methods: list[AppliedAug]  # the (key, params) transforms applied (empty for originals)


def _format_params(params: dict) -> str:
    """Compact, human-readable rendering of drawn parameters, e.g. ``angle_deg=37.21``."""
    parts = []
    for name, value in params.items():
        if isinstance(value, float):
            parts.append(f"{name}={value:.4g}")
        else:
            parts.append(f"{name}={value}")
    return "; ".join(parts)


def _provenance_lines(config: AugmentConfig) -> list[str]:
    """The run's provenance as ``#`` comment lines prepended to the manifest CSV."""
    lines = [
        "# Augmentation run provenance",
        f"# selected = {', '.join(config.selected)}",
        f"# min_methods = {config.min_methods}",
        f"# max_methods = {config.max_methods}",
        f"# n_per_sample = {config.n_per_sample}",
        f"# seed = {config.seed}",
        f"# height_channel = {config.height_channel}",
        f"# out_suffix = {config.out_suffix}",
    ]
    for key in config.selected:
        for pname, (lo, hi) in config.ranges.get(key, {}).items():
            lines.append(f"# range:{key}.{pname} = {lo} .. {hi}")
    return lines


def write_manifest(rows: list[ManifestRow], config: AugmentConfig, path: str | Path) -> Path:
    """Write the manifest CSV (provenance as leading ``#`` comment lines) and return its path."""
    path = Path(path)
    method_keys = [a.key for a in AUGMENTATIONS]

    records = []
    for row in rows:
        applied = {key: params for key, params in row.methods}
        record = {
            "aug_id": row.aug_id,
            "source_file": row.source_file,
            "output_file": row.output_file,
            "plot": row.plot,
            "n_methods": len(row.methods),
        }
        for key in method_keys:
            record[key] = _format_params(applied[key]) if key in applied else ""
        records.append(record)

    frame = pd.DataFrame.from_records(
        records,
        columns=["aug_id", "source_file", "output_file", "plot", "n_methods", *method_keys],
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    header = "\n".join(_provenance_lines(config)) + "\n"
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(header)
        frame.to_csv(fh, index=False)
    return path


@dataclass
class PreviousOutput:
    """A previous augmentation output folder, ready to reuse for feature generation."""

    folder: Path
    files: list[Path]            # every .las/.laz in the folder (originals + augmented)
    manifest_path: Path | None   # the manifest, if present


def load_previous_output(folder: str | Path) -> PreviousOutput:
    """List the point-cloud files (and locate the manifest) in a previous output folder.

    Raises
    ------
    FileNotFoundError
        If ``folder`` does not exist or contains no ``.las/.laz`` files.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a folder: {folder}")
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in LAS_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"No .las/.laz files found in {folder}")
    manifest = folder / MANIFEST_NAME
    return PreviousOutput(
        folder=folder, files=files, manifest_path=manifest if manifest.exists() else None
    )

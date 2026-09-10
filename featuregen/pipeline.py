"""Batch processing: a list of files -> a DataFrame -> a .csv table.

Kept free of any GUI dependency so it can be unit-tested and reused headless. The
GUI's worker thread calls :func:`run_batch` and forwards the ``progress`` callback
to update its progress bar and log.

The features and targets tables are written as ``.csv``. Each column's description (the metric
tooltip / reference-column note) is written as leading ``#`` comment lines, so the single file both
holds the data and documents it; readers that pass ``comment="#"`` skip those lines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from .external import ExternalData
from .features import FEATURES_BY_KEY, compute_features
from .io_las import Config, load_plot

# Called once per file with (index, total, file_name, error_or_None).
ProgressCallback = Callable[[int, int, str, str | None], None]

# Safety cap: a coded column with too many distinct values would explode the feature table into
# hundreds of near-empty binary columns, so one-hot encoding is refused past this many.
MAX_ONEHOT_COLUMNS = 100


@dataclass
class BatchResult:
    """Outcome of a batch run."""

    output_path: Path
    frame: pd.DataFrame                       # rows = files, columns = features (X)
    failed: list[tuple[str, str]] = field(default_factory=list)  # (file_name, reason)
    targets_path: Path | None = None          # separate targets (y) file, if a target was set
    targets_frame: pd.DataFrame | None = None  # rows = files, single target column

    @property
    def n_succeeded(self) -> int:
        return len(self.frame)

    @property
    def n_failed(self) -> int:
        return len(self.failed)


def run_batch(
    files: list[str | Path],
    output_path: str | Path,
    feature_keys: list[str] | None = None,
    config: Config | None = None,
    progress: ProgressCallback | None = None,
    external: ExternalData | None = None,
    external_columns: list[str] | None = None,
    encode_columns: list[str] | None = None,
    target_column: str | None = None,
    targets_path: str | Path | None = None,
) -> BatchResult:
    """Compute features for ``files`` and write them to ``output_path`` (.csv).

    A file that cannot be read (e.g. missing height channel) is recorded in
    ``failed`` and skipped; the batch always completes. The output index is the
    file name (one row per file) and the columns are the selected features in
    canonical order, followed by any selected imported reference columns.

    Imported reference features
    ---------------------------
    If ``external`` is given and ``external_columns`` is non-empty, each file's
    plot number (the ``plot(...)`` part of its name) is matched against the
    reference sheet and the selected columns are appended to that file's row.
    Unmatched files get ``NaN`` for the reference columns.

    Ground-truth target (y)
    -----------------------
    If ``target_column`` is given (a reference column designated as the ML label),
    it is matched per file like the other reference columns but is **kept out of the
    features table** and written to a **separate** ``targets_path`` workbook (one
    target column, same file-name index). This keeps ``y`` out of ``X``.
    """
    config = config or Config()
    files = [Path(f) for f in files]
    if feature_keys is None:
        feature_keys = list(FEATURES_BY_KEY.keys())
    external_columns = external_columns or []
    # The target never doubles as a feature column.
    if target_column is not None:
        external_columns = [c for c in external_columns if c != target_column]
    use_external = external is not None and bool(external_columns)
    use_target = external is not None and target_column is not None
    # One-hot encoding applies only to selected reference *feature* columns (the target, being a
    # regression label, is written as-is). Pre-flight the combined column cap before any work.
    encode_columns = [c for c in (encode_columns or []) if c in external_columns]
    if encode_columns and external is not None:
        total_binary = sum(len(external.unique_values(c)) for c in encode_columns)
        if total_binary > MAX_ONEHOT_COLUMNS:
            raise ValueError(
                f"Splitting {len(encode_columns)} coded column(s) would create {total_binary} "
                f"binary features, over the limit of {MAX_ONEHOT_COLUMNS}. Untick some columns "
                "or pick lower-cardinality ones."
            )

    rows: dict[str, dict[str, float]] = {}
    extra_rows: dict[str, dict[str, object]] = {}
    target_rows: dict[str, object] = {}
    failed: list[tuple[str, str]] = []
    total = len(files)

    for i, path in enumerate(files):
        try:
            plot = load_plot(path, config)
            rows[path.name] = compute_features(plot, feature_keys)
            if use_external:
                extra_rows[path.name] = external.lookup(path.name, external_columns)
            if use_target:
                target_rows[path.name] = external.lookup(path.name, [target_column])[target_column]
            error: str | None = None
        except Exception as exc:  # noqa: BLE001 - surface any read/parse failure per file
            failed.append((path.name, str(exc)))
            error = str(exc)
        if progress is not None:
            progress(i + 1, total, path.name, error)

    # Build the frame with explicit column order even if every file failed.
    frame = pd.DataFrame.from_dict(rows, orient="index", columns=feature_keys)
    frame.index.name = "filename"
    if use_external:
        extra_frame = pd.DataFrame.from_dict(
            extra_rows, orient="index", columns=external_columns
        )
        frame = frame.join(extra_frame)

    # Expand any coded reference columns into one binary column per value ("<col>=<value>"),
    # preserving column order; a missing/NaN source value yields NaN across that column's dummies.
    binary_columns: list[str] = []
    for col in encode_columns:
        frame, made = _one_hot_expand(frame, col, external.unique_values(col))
        binary_columns += made

    output_path = Path(output_path)
    _write_csv(frame, output_path, external_columns, external, binary_columns)

    # Separate targets (y) table.
    targets_frame: pd.DataFrame | None = None
    written_targets_path: Path | None = None
    if use_target:
        targets_frame = pd.DataFrame.from_dict(
            {name: {target_column: value} for name, value in target_rows.items()},
            orient="index",
            columns=[target_column],
        )
        targets_frame.index.name = "filename"
        written_targets_path = (
            Path(targets_path) if targets_path is not None
            else output_path.with_name(f"targets_{output_path.stem}.csv")
        )
        _write_targets_csv(targets_frame, written_targets_path, target_column, external)

    return BatchResult(
        output_path=output_path,
        frame=frame,
        failed=failed,
        targets_path=written_targets_path,
        targets_frame=targets_frame,
    )


def _one_hot_expand(
    frame: pd.DataFrame, column: str, values: list
) -> tuple[pd.DataFrame, list[str]]:
    """Replace ``column`` in ``frame`` with one binary column per value in ``values``.

    Each new column is ``"<column>=<value>"`` holding 1.0 where the row equals that value, 0.0
    otherwise, and NaN where the source value is missing (so unmatched plots stay unmatched). The
    binary columns take the original column's position; an empty ``values`` leaves the frame as-is.
    """
    if column not in frame.columns or not values:
        return frame, []
    src = frame[column]
    insert_at = frame.columns.get_loc(column)
    frame = frame.drop(columns=[column])
    made: list[str] = []
    for offset, value in enumerate(values):
        name = f"{column}={value}"
        col = src.eq(value).astype(float)
        col[src.isna()] = float("nan")  # missing source -> NaN, not a spurious 0
        frame.insert(insert_at + offset, name, col)
        made.append(name)
    return frame, made


def _write_with_comments(frame: pd.DataFrame, path: Path, descriptions: dict[str, str]) -> None:
    """Write ``frame`` to ``path`` as CSV, prepending ``descriptions`` as ``# col: desc`` lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {col}: {desc}" for col, desc in descriptions.items() if desc]
    header = ("\n".join(lines) + "\n") if lines else ""
    with path.open("w", encoding="utf-8", newline="") as fh:
        if header:
            fh.write(header)
        frame.to_csv(fh, index=True)


def _write_targets_csv(
    frame: pd.DataFrame, path: Path, target_column: str, external: ExternalData | None
) -> None:
    """Write the single-column targets (y) table to ``.csv``, indexed by file name."""
    source = external.source_path.name if external is not None else "imported sheet"
    desc = {target_column: f"Ground-truth target from {source}, matched by plot number."}
    _write_with_comments(frame, path, desc)


def _write_csv(
    frame: pd.DataFrame,
    output_path: Path,
    external_columns: list[str] | None = None,
    external: ExternalData | None = None,
    binary_columns: list[str] | None = None,
) -> None:
    """Write the features frame to ``.csv``, documenting each column in leading ``#`` comment lines."""
    external_columns = set(external_columns or [])
    binary_columns = set(binary_columns or [])
    source = external.source_path.name if external is not None else "imported sheet"
    descriptions: dict[str, str] = {}
    for name in frame.columns:
        feature = FEATURES_BY_KEY.get(name)
        if feature is not None:
            descriptions[name] = feature.tooltip
        elif name in binary_columns:
            descriptions[name] = f"One-hot binary feature (split from a coded column in {source})."
        elif name in external_columns:
            descriptions[name] = f"Imported reference feature from {source}."
    _write_with_comments(frame, output_path, descriptions)

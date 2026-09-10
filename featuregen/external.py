"""Optional reference data imported from a spreadsheet and matched to plot files.

The LiDAR metrics in :mod:`featuregen.features` are computed from the point cloud
alone. A user often also has *measured* values for each plot (yields, weights,
heading date, species, ...) in a separate spreadsheet and wants to fold them into
the same export so the result is one ready-to-train feature table.

This module reads that spreadsheet into an :class:`ExternalData`:

* a **chosen key column** holds the plot number (e.g. ``429`` or ``plot(429)``) and becomes
  the index; the user picks it via the "Label" selector (defaults to the first column);
* every other column is an optional, user-toggleable extra feature.

Matching to a LiDAR file is by plot number: a file named ``test_plot(429).laz``
carries the number ``429`` inside ``plot(...)``, which is looked up in the index.
A file whose number is absent from the sheet simply yields ``NaN`` for the extra
columns, so a missing measurement never aborts a batch.

The key column's cells are read through the configurable label format (:data:`common.naming
.LABEL_FORMAT`), so a sheet keyed as ``plot(429)`` or a bare ``429`` both resolve to ``429``.
Either a ``.csv`` or a ``.xlsx`` reference file is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from math import nan
from pathlib import Path

import pandas as pd

# The plot-number conventions live in common.naming (shared by the augmenter and the ML module
# too); re-exported here so existing imports of featuregen.external.plot_number_from_name keep
# working. plot_number_from_label reads the *external* key column in the user's label format.
from common.naming import (
    LabelFormat,
    PlotKey,
    key_stride,
    plot_number_from_label,
    plot_number_from_name,
)

def default_reference_dir() -> str:
    """The configured default folder for the reference table (``reference_dir`` in the active preset).

    Used to seed the feature tab's "Browse…" dialog when a project has no pinned reference yet.
    Empty when no default is configured. Read lazily so the GUI picks up the current preset.
    """
    from common.config import CONFIG  # local import to avoid a config import at package load

    return CONFIG.start_dir("reference")


@dataclass
class ExternalData:
    """Reference spreadsheet, indexed by plot number.

    Attributes
    ----------
    frame:
        The sheet with the plot-number column promoted to the (integer) index;
        the remaining columns are the selectable extra features.
    columns:
        The extra-feature column names, in sheet order.
    source_path:
        Where the data was read from (used for the Excel header comments).
    """

    frame: pd.DataFrame
    columns: list[str]
    source_path: Path
    # Plot numbers carried by more than one row. The sheet cannot say which of them is meant, and
    # :meth:`lookup` silently takes the first, so this is surfaced to the user rather than left to
    # be discovered as quietly wrong feature values. Typically a sheet covering several scan dates
    # or growth stages, where the same plot appears once per date.
    duplicate_keys: list[int] = dc_field(default_factory=list)

    def unique_values(self, column: str) -> list:
        """Sorted distinct non-null values of ``column`` (the categories one-hot would create)."""
        if column not in self.frame.columns:
            return []
        vals = self.frame[column].dropna().unique().tolist()
        try:
            return sorted(vals)
        except TypeError:  # mixed/unorderable types -> stable string sort
            return sorted(vals, key=str)

    def lookup(self, filename: str, columns: list[str]) -> dict[str, object]:
        """Return the requested ``columns`` for the plot matching ``filename``.

        Missing plot numbers (no ``plot(...)`` in the name, or a number absent
        from the sheet) yield ``NaN`` for every requested column so the row still
        lines up with the computed features.
        """
        number = plot_number_from_name(filename)
        if number is None or number not in self.frame.index:
            return {col: nan for col in columns}
        row = self.frame.loc[number]
        if isinstance(row, pd.DataFrame):  # duplicate plot numbers -> take the first
            row = row.iloc[0]
        return {col: (row[col] if col in row.index else nan) for col in columns}


def list_reference_columns(path: str | Path) -> list[str]:
    """Return the column headers of the reference file at ``path`` (label/key candidates).

    Reads only the header (plus one data row) so the GUI can offer every column as the "Label"
    (key) selector before the full table is loaded. Accepts ``.csv`` or ``.xlsx``. Trailing
    all-blank columns are dropped.

    One data row is read because ``dropna(how="all")`` on a header-only (zero-row) frame treats
    *every* column as empty and drops them all, returning ``[]``; reading a row lets real columns
    survive while a genuinely-blank trailing column is still removed (matching ``load_external``).
    """
    frame = _read_table(path, nrows=1)
    frame = frame.dropna(axis=1, how="all")
    return [str(c) for c in frame.columns]


def _read_table(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Read a reference table from ``path`` (``.csv`` or ``.xlsx``) into a DataFrame."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, nrows=nrows)
    return pd.read_excel(path, nrows=nrows)


def column_values(path: str | Path, column: str) -> list[str]:
    """The distinct values of ``column`` in the sheet at ``path``, as strings, sorted.

    Offered to the GUI so the row filter can be picked from what the sheet actually contains
    rather than typed from memory. Returns ``[]`` for an unreadable file or a missing column.
    """
    try:
        frame = _read_table(path)
    except Exception:  # noqa: BLE001 - the caller reports load errors properly
        return []
    if column not in frame.columns:
        return []
    values = {str(v).strip() for v in frame[column].dropna()}
    return sorted(values)


def _apply_filter(
    frame: pd.DataFrame, column: str | None, value: str | None, path: Path
) -> pd.DataFrame:
    """Keep only the rows whose ``column`` equals ``value`` (compared as trimmed strings).

    A reference sheet often covers more than one scan date or growth stage, listing every plot
    once per date. That makes the plot number ambiguous, and the lookup would silently take
    whichever row came first. Filtering to a single date is what makes the key unique again.

    Comparison is on the string form so a numeric code typed as ``2`` still matches a cell read
    as ``2.0``. A filter that matches nothing raises rather than yielding an empty join.
    """
    if not column or value is None or value == "" or column not in frame.columns:
        return frame
    wanted = str(value).strip()
    as_text = frame[column].map(lambda v: str(v).strip())
    kept = frame[as_text == wanted]
    if kept.empty:
        available = ", ".join(sorted({str(v).strip() for v in frame[column].dropna()})[:10])
        raise ValueError(
            f"No rows in {path.name} have {column} = '{wanted}'. Values present: {available}"
        )
    return kept


def load_external(
    path: str | Path,
    key_column: str | None = None,
    label_format: LabelFormat | None = None,
    row_column: str | None = None,
    stride: int | None = None,
    filter_column: str | None = None,
    filter_value: str | None = None,
) -> ExternalData:
    """Read the reference table at ``path`` (``.csv``/``.xlsx``) into an :class:`ExternalData`.

    ``key_column`` is the column whose cells hold the plot's column number; if ``None`` the first
    column is used (back-compat). Its cells are parsed through ``label_format`` (the active
    :data:`common.naming.LABEL_FORMAT` by default) so a key like ``plot(429)`` or a bare ``429``
    both resolve to ``429``.

    ``row_column`` names a second key column for sheets whose plots are identified by a *pair* -
    SGCBP keys its ground truth on ``(runNo, rangeNo)``, where neither number is unique alone. The
    pair is packed into the single canonical plot number by :meth:`~common.naming.PlotKey.number`,
    the exact rule :func:`clip.importer.plan_import` uses on the file names, with the same
    ``stride`` (which the import records in its manifest). Using one rule on both sides is what
    guarantees the sheet and the plots stay joined - the number is derived from the source data on
    each side, never assigned independently.

    Rows whose key does not parse are dropped and the resulting integer becomes the index.
    Fully-empty columns (trailing blanks common in exported sheets) are discarded.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the table has no usable key column or no rows with a valid key.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Reference data file not found: {path}")

    frame = _read_table(path)
    frame = frame.dropna(axis=1, how="all")  # drop trailing/blank columns
    if frame.shape[1] < 2:
        raise ValueError(
            f"{path.name} needs a plot-number column plus at least one data column."
        )
    frame = _apply_filter(frame, filter_column, filter_value, path)

    key_col = key_column if key_column in frame.columns else frame.columns[0]
    cols = frame[key_col].map(lambda v: plot_number_from_label(v, label_format))
    use_row = row_column is not None and row_column in frame.columns and row_column != key_col
    if use_row:
        rows = frame[row_column].map(lambda v: plot_number_from_label(v, label_format))
        keys = [
            None if (c is None or r is None) else PlotKey(col=int(c), row=int(r))
            for c, r in zip(cols, rows)
        ]
    else:
        keys = [None if c is None else PlotKey(col=int(c)) for c in cols]

    # The stride must match the import's, so a sheet loaded before any plots exist (or against a
    # differently-sized import) still packs the pair the same way. Falling back to the sheet's own
    # widest column keeps a standalone load sensible.
    packing = stride if stride is not None else key_stride([k for k in keys if k is not None])
    numbers = [None if k is None else k.number(packing) for k in keys]

    frame = frame.assign(**{str(key_col): numbers}).dropna(subset=[key_col])
    if frame.empty:
        raise ValueError(
            f"No rows in {path.name} have a readable plot number in the key "
            f"column ('{key_col}')."
        )
    frame[key_col] = frame[key_col].astype(int)
    frame = frame.set_index(key_col)
    if use_row:
        # The row column has been folded into the index; leaving it as a feature would hand the
        # model a bare grid coordinate.
        frame = frame.drop(columns=[row_column], errors="ignore")

    duplicates = frame.index[frame.index.duplicated()].unique().tolist()

    columns = [str(c) for c in frame.columns]
    frame.columns = columns  # normalise headers to plain strings
    return ExternalData(
        frame=frame, columns=columns, source_path=path,
        duplicate_keys=[int(k) for k in duplicates],
    )

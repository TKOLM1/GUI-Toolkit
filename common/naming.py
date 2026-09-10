"""The filename conventions that hold the whole pipeline together.

Two numbers can be embedded in a plot file's name:

* ``plot(N)`` - the **plot number**. It identifies the physical wheat plot and is the
  key that matches a file to its reference/ground-truth row *and* that groups every
  augmented copy of a plot together so the ML train/test split never leaks a plot
  across both sides. It is preserved verbatim through augmentation.
* ``aug(N)`` - the **augmentation id**, appended by the augmenter. ``N`` numbers a plot's
  augmented copies ``1..k`` (restarting at 1 for each plot), so together with the
  ``source_file`` it keys into the augmentation manifest, which records exactly which
  transforms (and parameters) produced that file. Numbering per plot lets the results map
  show "all aug(1) copies", "all aug(2) copies", and so on.

So ``field_plot(429).laz`` augmented as its 2nd copy becomes ``field_plot(429)_aug(2).laz``:
the plot number survives, the per-plot aug id is added.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# The point-cloud file extensions the whole app accepts.
LAS_SUFFIXES = {".las", ".laz"}

# Matches the number inside "plot(<number>)" of a file name, e.g.
# "test_plot(429).laz" -> 429. Case-insensitive and tolerant of inner spaces.
#
# This is the *canonical* plot pattern the app always writes and reads internally — every clipped /
# imported / augmented file is named with it, so the anti-leakage grouping (ml) keys off it. It is
# deliberately fixed; the user-configurable label format below only governs how labels are *read in*
# from external sources (mask attribute values, imported filenames, reference-table keys).
_PLOT_RE = re.compile(r"plot\s*\(\s*(\d+)\s*\)", re.IGNORECASE)

# Matches the number inside "aug(<number>)", e.g. "x_aug(7).laz" -> 7.
_AUG_RE = re.compile(r"aug\s*\(\s*(\d+)\s*\)", re.IGNORECASE)


@dataclass(frozen=True)
class LabelFormat:
    """How a plot number is spelled in *external* sources (not in the app's own filenames).

    The number sits between a fixed ``start`` and ``end`` string (defaults ``"plot("`` / ``")"``),
    so ``plot_number_from_label("plot(429)")`` is ``429``. Used to read labels out of imported
    filenames and reference-table key cells, letting users whose data uses a different convention
    (e.g. ``"P_" … ""``) still match. The app then *writes* every file with the canonical
    ``plot(N)`` (see :func:`build_plot_name`), so the deep pipeline never sees the custom format.
    """

    start: str = "plot("
    end: str = ")"

    def pattern(self) -> re.Pattern[str]:
        """A compiled regex capturing the integer between ``start`` and ``end`` (start may be empty)."""
        return re.compile(re.escape(self.start) + r"\s*(-?\d+)\s*" + re.escape(self.end), re.IGNORECASE)


# The process-wide active label format. Set once at startup from the active preset (and again when
# the user switches preset on the Setup tab) via :func:`set_label_format`. Mutable module state on
# purpose: it is read-only configuration shared by every page, like CONFIG.
LABEL_FORMAT = LabelFormat()


def set_label_format(start: str, end: str) -> None:
    """Replace the process-wide :data:`LABEL_FORMAT` (applied from the active preset)."""
    global LABEL_FORMAT
    LABEL_FORMAT = LabelFormat(start=start, end=end)


def plot_number_from_label(text: object, fmt: LabelFormat | None = None) -> int | None:
    """Extract the plot number from ``text`` spelled in the user's label format, or ``None``.

    Tries the configured ``fmt`` (``LABEL_FORMAT`` by default) first; if that does not match, falls
    back to a bare integer so a plain ``429`` (or a numeric reference-table cell) still reads. Used
    for *external* inputs — imported filenames and reference-table key cells — never for the app's
    own canonical ``plot(N)`` names (use :func:`plot_number_from_name` for those).

    >>> plot_number_from_label("plot(429).laz")
    429
    >>> plot_number_from_label("429")
    429
    """
    fmt = fmt or LABEL_FORMAT
    s = str(text)
    match = fmt.pattern().search(s)
    if match:
        return int(match.group(1))
    bare = re.fullmatch(r"\s*(-?\d+)(?:\.0+)?\s*", s)  # tolerate "429" and "429.0"
    return int(bare.group(1)) if bare else None


@dataclass(frozen=True)
class PlotKey:
    """A plot's identity in the *source* data: a column number, optionally paired with a row.

    Some datasets identify a plot with one number (``plot(429)``); others need two, because the
    plot sits at a position in a field grid and neither coordinate is unique on its own. The SGCBP
    wheat set is the second kind: its files are named ``<run>-<range>-1-b.pcd`` and its ground
    truth is keyed on ``(runNo, rangeNo)`` — there are only 18 distinct range numbers across 234
    files, so ``range`` alone identifies nothing.

    Holding both lets one rule serve every stage: the pair is unique, it re-keys the reference
    sheet the same way it names the files (so the two can never drift apart), and ``row``/``col``
    *are* the field layout, which is what the grid option reconstructs.
    """

    col: int
    row: int | None = None

    def number(self, stride: int) -> int:
        """The single canonical ``plot(N)`` integer for this key.

        With no row the column number is used unchanged, so a dataset that already has plain plot
        numbers keeps them (429 stays 429). With a row the two are packed as ``row*stride + col``,
        which is reversible, stable, and stays readable: row 1 / column 12 with ``stride`` 100
        reads as 112.
        """
        return self.col if self.row is None else self.row * stride + self.col


def key_stride(keys: list[PlotKey]) -> int:
    """The packing multiplier for :meth:`PlotKey.number` — the power of ten clear of every column.

    Derived from the data rather than hardcoded so the packed number is as short as the dataset
    allows, and so no column value can ever overflow into the row's digits.
    """
    widest = max((k.col for k in keys if k.col is not None), default=0)
    stride = 10
    while stride <= widest:
        stride *= 10
    return stride


@dataclass(frozen=True)
class KeyFormat:
    """How a :class:`PlotKey` is spelled in a source file name.

    ``col`` is the required format for the column number; ``row`` is the optional second one. Both
    are read from the same string, so ``1-12-1-b.pcd`` with ``row`` = ``""``/``"-"`` and ``col`` =
    ``"-"``/``"-1-b"`` yields ``PlotKey(row=1, col=12)``.
    """

    col: LabelFormat = field(default_factory=LabelFormat)
    row: LabelFormat | None = None

    def read(self, text: object) -> PlotKey | None:
        """Parse ``text`` into a :class:`PlotKey`, or ``None`` when the column number is absent.

        A missing *row* is not a failure — it simply means this source is single-numbered — but a
        missing column is, since there would be nothing to identify the plot by.
        """
        col = plot_number_from_label(text, self.col)
        if col is None:
            return None
        row = plot_number_from_label(text, self.row) if self.row is not None else None
        return PlotKey(col=col, row=row)


def plot_number_from_name(name: str) -> int | None:
    """Extract the integer inside ``plot(...)`` of a file name, or ``None``.

    >>> plot_number_from_name("test_plot(429).laz")
    429
    """
    match = _PLOT_RE.search(str(name))
    return int(match.group(1)) if match else None


def aug_number_from_name(name: str) -> int | None:
    """Extract the integer inside ``aug(...)`` of a file name, or ``None``.

    >>> aug_number_from_name("test_plot(429)_aug(7).laz")
    7
    """
    match = _AUG_RE.search(str(name))
    return int(match.group(1)) if match else None


def build_aug_name(original_name: str | Path, aug_id: int, suffix: str = ".laz") -> str:
    """Name for augmented copy ``aug_id`` of ``original_name``.

    The original stem (including its ``plot(N)`` part) is kept and ``_aug(<id>)`` is
    appended before the extension, which is swapped to ``suffix``.

    >>> build_aug_name("field_plot(429).laz", 7)
    'field_plot(429)_aug(7).laz'
    """
    stem = Path(original_name).stem
    return f"{stem}_aug({aug_id}){suffix}"


def build_copy_name(original_name: str | Path, suffix: str = ".laz") -> str:
    """Name for the verbatim copy of an original file, with its extension set to ``suffix``.

    >>> build_copy_name("field_plot(429).las")
    'field_plot(429).laz'
    """
    return f"{Path(original_name).stem}{suffix}"


def build_plot_name(label: object, universal_string: str = "", suffix: str = ".laz") -> str:
    """Name for a freshly clipped plot: ``plot(<label>)<universal_string><suffix>``.

    Used by the clipping tab. ``label`` is the plot number carried by the mask polygon;
    ``universal_string`` is inserted **verbatim right after** ``plot(N)`` (and before any
    ``_aug(k)`` the augmenter later appends), so it never disturbs ``plot``/``aug`` parsing.

    >>> build_plot_name(429, "_siteA")
    'plot(429)_siteA.laz'
    >>> build_plot_name(429)
    'plot(429).laz'
    """
    return f"plot({label}){universal_string}{suffix}"

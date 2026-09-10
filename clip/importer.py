"""Importing a folder of already-separated plot clouds into a project's ``plots/`` folder.

This is the second of the import tab's two routes (the first is :mod:`clip.clipper`, which cuts
one whole-field cloud with a ``.gpkg`` of masks). Here the plots are *already* separate files -
one per plot, from a previous crop, a collaborator, or a published dataset - and the job is to
give them the canonical ``plot(N)`` names the rest of the pipeline keys off, converting the file
format where needed.

**How a plot is identified.** Each file name is read into a :class:`~common.naming.PlotKey` - a
column number, optionally paired with a row - using the user's :class:`~common.naming.KeyFormat`.
One number is enough for a dataset that already has plot ids (``plot(429).laz`` -> 429). Two are
needed when the id is a position in a field grid: SGCBP names its files ``<run>-<range>-1-b.pcd``
and keys its ground truth on ``(runNo, rangeNo)``, where neither number is unique alone.

That single choice settles three things that would otherwise each need their own mechanism:

* **uniqueness** - the pair distinguishes plots that share a row or a column, so there is no
  collision to patch up and nothing has to be renumbered;
* **the join** - :func:`featuregen.external.load_external` re-keys the reference sheet through the
  *same* :meth:`~common.naming.PlotKey.number` rule, so the files and the sheet cannot drift apart;
* **the layout** - row and column *are* the field grid, so :class:`GridLayout` can rebuild the real
  arrangement of a dataset whose plots were all stored on the same origin.

The rest is mechanical: convert to ``.laz`` (``.pcd`` via :mod:`clip.pcd`, ``.las``/``.laz`` via
:mod:`common.las_io`, so every point dimension survives), optionally scale the coordinates for a
dataset recorded in millimetres, and write an :data:`MANIFEST_NAME` recording which source file
became which ``plot(N)``.

Everything here is GUI-free; :class:`gui.workers.ImportWorker` runs it off the UI thread.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from common import las_io
from common.naming import (
    LAS_SUFFIXES,
    KeyFormat,
    PlotKey,
    build_plot_name,
    key_stride,
)

from .pcd import read_pcd

# Every point-cloud extension the import route accepts (LAS/LAZ plus PCD).
PCD_SUFFIX = ".pcd"
IMPORT_SUFFIXES = LAS_SUFFIXES | {PCD_SUFFIX}

# The manifest written beside the imported plots: the audit trail from the source dataset's own
# identifiers to the plot(N) the pipeline uses. Row/column are recorded even when the layout
# option is off, because they are what a reference sheet is re-keyed on.
MANIFEST_NAME = "import_manifest.csv"
MANIFEST_COLUMNS = ["plot", "file", "row", "column", "source_file", "source_path"]

# Called once per file with (done, total, source_name, error_or_None).
ProgressCallback = Callable[[int, int, str, str | None], None]

# Common unit choices for source data, as (label, metres-per-source-unit). The whole feature
# stack (entropy bin sizes, height thresholds, viewer radii) assumes metres, so a dataset
# recorded in millimetres has to be scaled on the way in or every metric silently comes out
# 1000x too large.
UNIT_SCALES: dict[str, float] = {
    "Already in metres": 1.0,
    "Millimetres → metres": 0.001,
    "Centimetres → metres": 0.01,
    "Feet → metres": 0.3048,
}
DEFAULT_UNIT = "Already in metres"


def unit_name(text: str) -> str:
    """Resolve ``text`` to a key of :data:`UNIT_SCALES`, or :data:`DEFAULT_UNIT` if it matches none.

    Matching ignores case and treats an ASCII ``->`` as the ``→`` the dropdown displays, so a
    preset file (which is plain text, often edited in an editor that will not produce the arrow)
    can spell the unit either way.
    """
    wanted = text.strip().lower().replace("->", "→").replace(" ", "")
    for name in UNIT_SCALES:
        if name.lower().replace(" ", "") == wanted:
            return name
    return DEFAULT_UNIT


def unit_scale(text: str) -> float:
    """The metres-per-source-unit factor for ``text`` (see :func:`unit_name`)."""
    return UNIT_SCALES[unit_name(text)]


# Splits a name into digit / non-digit runs so file names sort "2" before "10".
_NATURAL_RE = re.compile(r"(\d+)")

# How many files to inspect when looking for a declared coordinate system. The answer is the same
# for every file in a dataset, so a handful settles it.
_DETECT_SAMPLE = 8


def detect_units(sources: list[Path]) -> tuple[str, str] | None:
    """The units ``sources`` *declare*, as ``(unit_name, reason)``, or ``None`` if they declare none.

    Only one source of truth is consulted: the coordinate reference system recorded in a
    ``.las``/``.laz`` header, which names its own axis unit. That is a statement by the file, not
    an inference about it - which also makes it the only way to recognise feet, since 0.3048 is
    far too close to 1.0 to be separated from metres by any property of the data.

    A ``.pcd`` carries no coordinate system at all, and plenty of ``.las`` files are written
    without one. Those return ``None``: the units are genuinely unknown, and the caller must ask
    rather than guess. Inferring them from how large the plots come out is possible but rests on
    an assumption about what is being scanned, so it is deliberately not done here.
    """
    import laspy

    for src in sources[:_DETECT_SAMPLE]:
        if src.suffix.lower() == PCD_SUFFIX:
            continue  # PCD has no coordinate-system header at all
        try:
            with laspy.open(src) as reader:
                crs = reader.header.parse_crs()
            if crs is None:
                continue
            factor = float(crs.axis_info[0].unit_conversion_factor)
        except Exception:  # noqa: BLE001 - an unreadable or CRS-less file is simply not evidence
            continue
        for name, scale in UNIT_SCALES.items():
            # These factors are decades apart, so a loose tolerance cannot mis-hit.
            if abs(factor - scale) <= scale * 0.01:
                return name, f"{src.name} declares its coordinates in {crs.axis_info[0].unit_name}"
    return None


@dataclass
class GridLayout:
    """Spread the imported plots out so they no longer sit on top of each other.

    Some datasets store every plot re-centred on its own origin, which is fine per file but makes
    the whole set overlap into one blob when loaded together - the 2-D layout view and any viewer
    become useless. Switching this on discards the source X/Y *positions* (never the shapes) and
    gives each plot its own cell.

    Where the cells go depends on what the file names carried. With a **row and column** the real
    field layout is rebuilt - column across X, row down Y - which is what the plot numbers meant in
    the first place. With only a single number the plots are simply laid out in reading order,
    ``columns`` per row; that is an arbitrary arrangement, but it still separates them.

    Cell size is the largest plot's span across the whole import, plus ``padding`` as a fraction of
    it, so no two plots can touch however uneven their sizes. Z is never touched: height is real
    data that features depend on.

    This is a *presentation* fix. It changes where plots sit relative to one another, not their
    internal geometry, so every per-plot feature is identical with it on or off.
    """

    columns: int = 10
    padding: float = 0.10  # extra gap as a fraction of the largest plot span

    def spacing(self, span_x: float, span_y: float) -> tuple[float, float]:
        """Cell pitch in X and Y for the given largest-plot spans (always strictly positive)."""
        factor = 1.0 + max(self.padding, 0.0)
        # A degenerate span (one point, or every plot identical) would collapse the grid onto a
        # single spot; fall back to a unit cell so the plots still separate.
        return (max(span_x, 1e-9) * factor, max(span_y, 1e-9) * factor)


@dataclass
class ImportPlan:
    """What :func:`plan_import` worked out before anything is written.

    ``entries`` is the import in file order as ``(source_path, key, plot_number)``. ``unreadable``
    lists files whose name gave no column number, and ``duplicates`` maps a plot number to the
    files that both claim it - which now means two files genuinely share an identity (the same
    row *and* column), not merely that the format was too coarse.
    """

    entries: list[tuple[Path, PlotKey, int]] = field(default_factory=list)
    duplicates: dict[int, list[Path]] = field(default_factory=dict)
    unreadable: list[Path] = field(default_factory=list)
    stride: int = 10

    @property
    def is_clean(self) -> bool:
        """True when every source file yielded a distinct, readable plot identity."""
        return not self.duplicates and not self.unreadable


@dataclass
class ImportResult:
    """Outcome of an import run."""

    output_dir: Path
    written: list[Path] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (source name, reason)
    manifest_path: Path | None = None

    @property
    def n_written(self) -> int:
        return len(self.written)


def natural_key(path: Path) -> tuple:
    """Sort key that orders ``plot2`` before ``plot10`` (digit runs compared as numbers)."""
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in _NATURAL_RE.split(str(path))
    )


def list_sources(folder: str | Path) -> list[Path]:
    """Every importable point-cloud file at or below ``folder``, in natural path order.

    The search recurses because datasets routinely split their plots across sub-folders (SGCBP
    keeps one folder per driving run), and importing those one at a time would give each batch its
    own numbering. Reading the whole tree at once means a plot's identity comes from its name, not
    from which folder it happened to be in.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(
        (p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMPORT_SUFFIXES),
        key=natural_key,
    )


def plan_import(sources: list[Path], fmt: KeyFormat) -> ImportPlan:
    """Work out every source file's plot key and canonical number, without writing anything.

    The numbers are *derived* from the file names, never invented: re-running an import, or
    importing the same dataset into a new project, always produces the same ``plot(N)``. That is
    what lets the reference sheet be re-keyed through the identical rule and still line up.

    Files whose name yields no column number are recorded in ``unreadable``; two files that resolve
    to the same identity are recorded in ``duplicates``. Both mean the format is wrong for this
    dataset (or the data really does contain a duplicate), and both are reported rather than
    silently patched.
    """
    plan = ImportPlan()
    keyed: list[tuple[Path, PlotKey]] = []
    for src in sources:
        key = fmt.read(src.stem)
        if key is None:
            plan.unreadable.append(src)
            continue
        keyed.append((src, key))

    # The stride is a property of the whole import (the widest column seen), so it is computed
    # once here and reused for the reference sheet - both sides must pack the pair identically.
    plan.stride = key_stride([k for _, k in keyed])

    claimed: dict[int, list[Path]] = {}
    for src, key in keyed:
        number = key.number(plan.stride)
        plan.entries.append((src, key, number))
        claimed.setdefault(number, []).append(src)
    plan.duplicates = {n: paths for n, paths in claimed.items() if len(paths) > 1}
    return plan


def _read_any(path: Path):
    """Read a ``.pcd``/``.las``/``.laz`` into a ``LasData`` (format decided by the extension)."""
    if path.suffix.lower() == PCD_SUFFIX:
        return read_pcd(path)
    return las_io.read_cloud(path)


def _set_coords(las, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    """Replace the cloud's coordinates, re-anchoring the header offsets on the new minimum corner.

    LAS stores each coordinate as a 32-bit integer, ``(value - offset) / scale``. That caps the
    representable span at roughly ``2.1e9 * scale`` *measured from the offset* - only ~214 metres
    at this pipeline's 0.1 mm precision. Real scan coordinates sit far outside that (SGCBP's later
    runs are at y ~ 218,000 mm), and so does a cloud that has just been moved onto a grid cell, so
    any write would fail with "Values given do not fit after applying offset and scale".

    Anchoring the offsets to the data's own minimum means only the plot's *extent* has to fit,
    which it always does. Every coordinate change goes through here so the invariant cannot be
    broken by a later step: the offsets are always correct for the values currently held.
    """
    las.header.offsets = [float(x.min()), float(y.min()), float(z.min())]
    las.x, las.y, las.z = x, y, z


def _coords(las) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The cloud's x/y/z as independent float arrays (safe to transform and hand back)."""
    return tuple(np.asarray(a, dtype=np.float64).copy() for a in (las.x, las.y, las.z))  # type: ignore[return-value]


def _apply_scale(las, scale: float) -> None:
    """Multiply the cloud's coordinates by ``scale`` in place (unit conversion)."""
    if scale == 1.0:
        return
    x, y, z = _coords(las)
    _set_coords(las, x * scale, y * scale, z * scale)


def _place_on_grid(las, cell: tuple[int, int], pitch: tuple[float, float]) -> None:
    """Re-centre the cloud on its own X/Y midpoint and move it to grid cell ``(col, row)``.

    The bounding-box midpoint is used rather than the mean or median: plots are often lopsided,
    and it is the box that a viewer draws, so aligning midpoints is what makes the cells look
    evenly spaced. Z is deliberately untouched.
    """
    col, row = cell
    pitch_x, pitch_y = pitch
    x, y, z = _coords(las)
    _set_coords(
        las,
        x - (x.min() + x.max()) / 2.0 + col * pitch_x,
        # Rows run "downwards" so the first row sits at the top, matching how the 2-D view reads.
        y - (y.min() + y.max()) / 2.0 - row * pitch_y,
        z,
    )


def _grid_cells(plan: ImportPlan, grid: GridLayout) -> dict[Path, tuple[int, int]]:
    """Assign each source file its ``(col, row)`` cell.

    When the file names carried a row, the real field layout is rebuilt from it: both axes are
    normalised so the smallest row/column observed sits at cell 0, which keeps the grid compact
    whatever the dataset numbers from. Otherwise the plots are laid out in plot-number order,
    ``grid.columns`` per row.
    """
    keyed = [(src, key) for src, key, _ in plan.entries]
    if keyed and all(key.row is not None for _, key in keyed):
        min_row = min(key.row for _, key in keyed)
        min_col = min(key.col for _, key in keyed)
        return {src: (key.col - min_col, key.row - min_row) for src, key in keyed}

    columns = max(grid.columns, 1)
    ordered = sorted(plan.entries, key=lambda e: e[2])  # by plot number
    return {src: (i % columns, i // columns) for i, (src, _, _) in enumerate(ordered)}


def _largest_spans(sources: list[Path], scale: float) -> tuple[float, float]:
    """The biggest X and Y extent across every source, in output units (after ``scale``).

    LAS/LAZ spans come from the header's mins/maxs, so no points are read; a ``.pcd`` has no
    such header and is parsed. Files that fail to open are ignored here - the write pass reports
    them properly.
    """
    span_x = span_y = 0.0
    for src in sources:
        try:
            if src.suffix.lower() == PCD_SUFFIX:
                las = read_pcd(src)
                dx = float(np.ptp(np.asarray(las.x, dtype=np.float64)))
                dy = float(np.ptp(np.asarray(las.y, dtype=np.float64)))
            else:
                import laspy

                with laspy.open(src) as reader:
                    mins, maxs = reader.header.mins, reader.header.maxs
                dx, dy = float(maxs[0] - mins[0]), float(maxs[1] - mins[1])
        except Exception:  # noqa: BLE001 - a bad file must not stop the measuring pass
            continue
        span_x, span_y = max(span_x, dx), max(span_y, dy)
    return span_x * scale, span_y * scale


def import_plots(
    sources: list[Path],
    output_dir: str | Path,
    fmt: KeyFormat,
    universal_string: str = "",
    scale: float = 1.0,
    grid: GridLayout | None = None,
    suffix: str = ".laz",
    progress: ProgressCallback | None = None,
) -> ImportResult:
    """Import ``sources`` into ``output_dir`` as canonical ``plot(N)`` files.

    Parameters
    ----------
    fmt:
        How the plot's row/column are spelled in the source file names.
    universal_string:
        Inserted verbatim right after ``plot(N)`` in every output name (may be empty).
    scale:
        Factor applied to X/Y/Z, for converting the source's units to metres
        (see :data:`UNIT_SCALES`).
    grid:
        When given, place the plots on a grid instead of keeping their source positions
        (see :class:`GridLayout`).

    A file that cannot be read or written is recorded in :attr:`ImportResult.skipped` and the run
    continues, so one corrupt plot never costs you the other 233.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plan = plan_import(sources, fmt)
    result = ImportResult(output_dir=output_dir)
    for src in plan.unreadable:
        result.skipped.append((src.name, "no plot number in the configured label format"))

    pitch: tuple[float, float] | None = None
    cells: dict[Path, tuple[int, int]] = {}
    if grid is not None:
        pitch = grid.spacing(*_largest_spans([s for s, _, _ in plan.entries], scale))
        cells = _grid_cells(plan, grid)

    rows: list[dict[str, object]] = []
    total = len(plan.entries)
    for i, (src, key, number) in enumerate(plan.entries):
        err: str | None = None
        try:
            las = _read_any(src)
            _apply_scale(las, scale)
            if grid is not None and pitch is not None:
                _place_on_grid(las, cells[src], pitch)
            dst = output_dir / build_plot_name(number, universal_string, suffix)
            las_io.write_cloud(las, dst)
            result.written.append(dst)
            rows.append({
                "plot": number,
                "file": dst.name,
                "row": "" if key.row is None else key.row,
                "column": key.col,
                "source_file": src.name,
                "source_path": str(src),
            })
        except Exception as exc:  # noqa: BLE001 - record and carry on to the next file
            err = str(exc)
            result.skipped.append((src.name, err))
        if progress is not None:
            progress(i + 1, total, src.name, err)

    if rows:
        result.manifest_path = _write_manifest(output_dir, rows)
    return result


def _write_manifest(output_dir: Path, rows: list[dict[str, object]]) -> Path:
    """Write (or overwrite) the import manifest, ordered by plot number."""
    path = output_dir / MANIFEST_NAME
    rows = sorted(rows, key=lambda r: int(r["plot"]))  # type: ignore[arg-type]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path

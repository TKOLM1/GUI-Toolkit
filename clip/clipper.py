"""Clipping a whole-field cloud to each plot polygon - exact, label-from-polygon.

:func:`clip_to_masks` is the GUI-free core of the clipping tab (the worker thread calls
it and forwards ``progress``). For every :class:`~clip.masks.PlotMask` it keeps exactly
the points whose ``(x, y)`` fall **inside** that polygon and writes them to
``plot(<label>)<universal_string>.laz`` - carrying *every* point dimension
(``RelativeHeight``, classification, RGB, ...) through :mod:`common.las_io`.

Correctness (the whole point of building this in-app): the interior test is
``shapely.contains_xy`` against the real polygon geometry, and the output's plot number is
the polygon's own label - so a point can never be assigned to the wrong plot and the label
can never drift, which is what the old "crop in a separate tool, then import" round-trip
got wrong.

A cheap bounding-box pre-filter keeps the exact test fast on million-point clouds.
Accepts ``.las`` or ``.laz`` input; always writes ``.laz``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import shapely

from common import las_io
from common.naming import build_plot_name

# Called once per polygon with (done, total, label, error_or_None).
ProgressCallback = Callable[[int, int, str, str | None], None]


@dataclass
class ClipResult:
    """Outcome of a clipping run."""

    output_dir: Path
    written: list[Path] = field(default_factory=list)      # plot files actually written
    empty: list[str] = field(default_factory=list)         # labels whose polygon held no points
    failed: list[tuple[str, str]] = field(default_factory=list)  # (label, reason)

    @property
    def n_written(self) -> int:
        return len(self.written)


def clip_to_masks(
    cloud_path: str | Path,
    masks,
    output_dir: str | Path,
    universal_string: str = "",
    suffix: str = ".laz",
    progress: ProgressCallback | None = None,
) -> ClipResult:
    """Clip ``cloud_path`` to each polygon in ``masks`` and write one ``.laz`` per plot.

    Parameters
    ----------
    cloud_path:
        The whole-field ``.las``/``.laz`` cloud (already in the masks' CRS).
    masks:
        A list of :class:`~clip.masks.PlotMask` (label + shapely geometry).
    universal_string:
        Inserted verbatim right after ``plot(N)`` in every output name (may be empty).

    A polygon with no points inside is recorded in :attr:`ClipResult.empty` and skipped;
    a polygon that fails for any other reason is recorded in :attr:`ClipResult.failed`. The
    run always completes.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    las = las_io.read_cloud(cloud_path)
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)

    result = ClipResult(output_dir=output_dir)
    total = len(masks)
    for i, mask in enumerate(masks):
        label = mask.label
        err: str | None = None
        try:
            inside = _points_inside(mask.geometry, x, y)
            n_inside = int(inside.sum())
            if n_inside == 0:
                result.empty.append(str(label))
            else:
                subset = las_io.take_points(las, inside)
                out_path = output_dir / build_plot_name(label, universal_string, suffix)
                las_io.write_cloud(subset, out_path)
                result.written.append(out_path)
        except Exception as exc:  # noqa: BLE001 - record and continue to the next polygon
            result.failed.append((str(label), str(exc)))
            err = str(exc)
        if progress is not None:
            progress(i + 1, total, str(label), err)

    return result


def _points_inside(geometry, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Boolean mask of the points strictly inside ``geometry`` (bbox-prefiltered)."""
    minx, miny, maxx, maxy = geometry.bounds
    inside = np.zeros(x.shape, dtype=bool)
    bbox = (x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy)
    if not bbox.any():
        return inside
    cand = np.flatnonzero(bbox)
    hit = shapely.contains_xy(geometry, x[cand], y[cand])
    inside[cand[np.asarray(hit, dtype=bool)]] = True
    return inside

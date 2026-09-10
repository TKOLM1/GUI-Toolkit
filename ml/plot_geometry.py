"""Per-plot real-world geometry for the results map: cached centroids + convex hulls.

This backs the **true-footprint map** view, which draws every plot as its convex hull at its
real-world position. Each plot's cloud is read only **once** per results refresh (just the X/Y
dimensions) and cached, so redraws and pan/zoom touch only the cached arrays.

The grid view's cell layout is *not* derived here - it comes from the explicit layout spec in
:mod:`ml.plot_layout` (dimensions + fill order, saved per project), so there is nothing to
auto-detect and nothing that can be subtly wrong.

Everything here is plain numpy/scipy; laspy is imported lazily inside the reader so importing
this module stays cheap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from featuregen.geometry import convex_hull_2d


@dataclass
class PlotGeom:
    """One plot's cached 2-D geometry in absolute world coordinates.

    ``hull`` is the footprint the field-map view draws. It is the plot's **minimum-area oriented
    bounding rectangle** (always 4 vertices) rather than the raw convex hull: plots are roughly
    rectangular, and a 4-vertex box keeps the field view fast to pan/zoom (80 dense hulls would
    be thousands of vertices). ``None`` when the points are degenerate.

    The rectangle is fitted to a lightly outlier-trimmed copy of the points (see
    :func:`_trim_outliers`) so a single stray point cannot inflate the footprint — the convex hull
    a min-area box is built on is otherwise pinned by its most extreme point. On already-clean
    clipped plots this trim changes the box by ~1-2%; on a noisy import it keeps the footprint honest.
    """

    centroid: tuple[float, float]
    hull: np.ndarray | None  # 4 world-XY corners of the oriented bounding box, or None


def _read_xy(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Read just the X/Y arrays from a cloud file, or ``None`` on any failure.

    Only the two horizontal dimensions are pulled (not the full record) to keep the one-time
    pass over every plot light.
    """
    if path is None or not path.exists():
        return None
    try:
        import laspy

        las = laspy.read(path)
        return np.asarray(las.x, dtype=float), np.asarray(las.y, dtype=float)
    except Exception:  # noqa: BLE001 - unreadable / unexpected file
        return None


def build_plot_geoms(file_map: dict, aug_index: int = 0) -> dict[int, PlotGeom]:
    """``plot -> PlotGeom`` for every plot at ``aug_index`` (default: the originals).

    Reads each cloud's XY a single time. Intended to be called once per results refresh and
    cached; redraws and pan/zoom then touch only the returned arrays.
    """
    geoms: dict[int, PlotGeom] = {}
    for (plot, aug), file in file_map.items():
        if aug != aug_index:
            continue
        xy = _read_xy(Path(file))
        if xy is None:
            continue
        x, y = xy
        if x.size == 0:
            continue
        tx, ty = _trim_outliers(x, y)
        geoms[plot] = PlotGeom(
            centroid=(float(x.mean()), float(y.mean())),
            hull=min_area_rect(tx, ty),
        )
    return geoms


def _trim_outliers(x: np.ndarray, y: np.ndarray, lo: float = 0.5, hi: float = 99.5) -> tuple[np.ndarray, np.ndarray]:
    """Drop points outside the ``[lo, hi]`` percentile band on either axis (robust footprint).

    The min-area box is pinned by the convex hull's most extreme point, so one stray return would
    stretch the whole footprint. Clipping the outer ~1% per axis removes such fliers while leaving
    the real plot edge intact. Returns the points unchanged if too few survive the clip.
    """
    if x.size < 20:
        return x, y  # too few points for percentiles to mean anything; trust them all
    xlo, xhi = np.percentile(x, [lo, hi])
    ylo, yhi = np.percentile(y, [lo, hi])
    keep = (x >= xlo) & (x <= xhi) & (y >= ylo) & (y <= yhi)
    if keep.sum() < 3:
        return x, y
    return x[keep], y[keep]


def min_area_rect(x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """The minimum-area oriented bounding rectangle of the points, as 4 ordered ``(x, y)`` corners.

    Rotating calipers over the convex hull: for each hull edge, rotate so that edge is axis-
    aligned, take the axis-aligned bounding box, and keep the smallest-area one. Returns ``None``
    for degenerate (collinear / too-few) inputs.
    """
    hull = convex_hull_2d(x, y)
    if hull is None or len(hull) < 3:
        return None
    edges = np.diff(np.vstack([hull, hull[:1]]), axis=0)
    best_area = np.inf
    best_corners = None
    for ex, ey in edges:
        length = np.hypot(ex, ey)
        if length == 0:
            continue
        # Unit edge direction and its perpendicular form the candidate box axes.
        ux, uy = ex / length, ey / length
        rot = np.array([[ux, uy], [-uy, ux]])
        proj = hull @ rot.T
        lo = proj.min(axis=0)
        hi = proj.max(axis=0)
        area = float((hi[0] - lo[0]) * (hi[1] - lo[1]))
        if area < best_area:
            best_area = area
            # The four box corners in the rotated frame, mapped back to world coordinates.
            corners = np.array([
                [lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]],
            ])
            best_corners = corners @ rot  # inverse of rot (orthonormal) is rot.T -> @ rot
    return best_corners

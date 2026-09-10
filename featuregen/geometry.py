"""Shared point-cloud geometry helpers (pure numpy/scipy, no laspy or Qt).

These back two consumers that must agree exactly: the scalar **feature** functions in
:mod:`featuregen.features` (e.g. ``sigma_z``) and the 3-D **visualisations** drawn in the
polyscope viewer (:mod:`tools.view_laz`). Keeping the maths here once means the roughness plane
the user sees is fitted by the *same* code that produced the roughness number on the map - they
can never drift.

Everything operates on plain arrays so the viewer process can import this without dragging in
laspy or the GUI.
"""

from __future__ import annotations

import numpy as np


def fit_plane(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares fit ``z = a*x + b*y + c`` through the points.

    Returns ``(coef, residuals)`` where ``coef = [a, b, c]`` are in the **centred** frame
    (x and y shifted to their means for conditioning, which does not change the residuals) and
    ``residuals = z - predicted``. This is the exact fit ``features.sigma_z`` reduces to an RMS.
    """
    xc = x - x.mean()
    yc = y - y.mean()
    design = np.column_stack([xc, yc, np.ones_like(xc)])
    coef, *_ = np.linalg.lstsq(design, z, rcond=None)
    residuals = z - design @ coef
    return coef, residuals


def plane_z(coef: np.ndarray, x: np.ndarray, y: np.ndarray, x0: float, y0: float) -> np.ndarray:
    """Evaluate a :func:`fit_plane` plane at ``(x, y)``; ``x0, y0`` are the fit's centring means."""
    a, b, c = coef
    return a * (x - x0) + b * (y - y0) + c


def height_percentiles(h: np.ndarray) -> dict[str, float]:
    """The height levels the map's central-tendency features describe, keyed by feature.

    Mirrors ``features.h_mean / h_median / h_p25 / h_p75 / h_p95 / h_max`` so each plane the
    viewer draws is labelled with the feature whose value it is.
    """
    return {
        "h_mean": float(np.mean(h)),
        "h_median": float(np.median(h)),
        "h_p25": float(np.percentile(h, 25)),
        "h_p75": float(np.percentile(h, 75)),
        "h_p95": float(np.percentile(h, 95)),
        "h_max": float(np.max(h)),
    }


def xy_bbox(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Axis-aligned XY extent ``(xmin, xmax, ymin, ymax)`` of the points."""
    return float(x.min()), float(x.max()), float(y.min()), float(y.max())


def convex_hull_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """Ordered ``(k, 2)`` vertices of the XY convex hull, or ``None`` if it is degenerate."""
    pts = np.column_stack([x, y])
    if len(pts) < 3:
        return None
    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(pts)
    except Exception:  # noqa: BLE001 - collinear / degenerate inputs
        return None
    return pts[hull.vertices]


def above_mean_mask(h: np.ndarray) -> np.ndarray:
    """Boolean mask of points above the mean height - the ``frac_above_mean`` subset."""
    return h > float(np.mean(h))


# --------------------------------------------------------------------------- #
# PCA axis frame - the basis of the hand-crafted horizontal feature class      #
# --------------------------------------------------------------------------- #
# Unlike the metrics above (which follow the open-source LiDAR_Vegetation_Metrics
# reference notebook), the helpers below were hand-crafted for this project. They
# describe the *horizontal* structure of a plot in the plot's **own** axis frame -
# found by a 2-D PCA of the vegetation (x, y) - so every value is invariant to how
# the plot happens to be rotated in the world coordinate system. Both the scalar
# features in ``featuregen.features`` and the polyscope viewer call these, so the
# drawn axes/planes always match the computed numbers.


def pca_axes_2d(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Principal axes of the XY point pattern, as the plot's own (major, minor) frame.

    Returns ``(centroid, axes, eigvals)`` or ``None`` when undefined (< 2 points):

    * ``centroid`` - ``[cx, cy]``, the mean XY position (the frame's origin).
    * ``axes`` - a ``(2, 2)`` array whose **rows** are the two unit axis vectors,
      row 0 = major (largest spread, ~ the long plot edge), row 1 = minor (~ the
      short edge), always perpendicular.
    * ``eigvals`` - the variances along those two axes, ``[lambda1, lambda2]`` with
      ``lambda1 >= lambda2 >= 0`` (covariance eigenvalues; units of m^2).

    The eigenvector **sign is fixed deterministically**: each axis is flipped, if
    needed, so the side of the centroid holding **more points** is the positive
    direction. Without this, the sign that ``eigh`` returns is arbitrary and could
    flip between otherwise-identical plots, which would scramble the front/back and
    left/right asymmetry features. With it, "positive" means the same physical thing
    on every plot - the side where most of the points sit. A tie (equal counts on
    both sides) leaves the axis sign as ``eigh`` returned it.
    """
    if x.size < 2:
        return None
    pts = np.column_stack([x, y]).astype(np.float64)
    centroid = pts.mean(axis=0)
    cov = np.cov(pts - centroid, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending; columns are eigenvectors
    order = np.argsort(eigvals)[::-1]       # -> descending (major first)
    eigvals = np.clip(eigvals[order], 0.0, None)
    axes = eigvecs[:, order].T              # rows = axis vectors
    # Sign each axis so the side of the centroid with more points is positive.
    centred = pts - centroid
    for i in range(2):
        proj = centred @ axes[i]
        if int(np.sum(proj > 0)) < int(np.sum(proj < 0)):
            axes[i] = -axes[i]
    return centroid, axes, eigvals


def project_to_axes(
    x: np.ndarray, y: np.ndarray, centroid: np.ndarray, axes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project ``(x, y)`` onto the PCA frame -> ``(s_major, s_minor)`` coordinates (metres)."""
    centred = np.column_stack([x, y]).astype(np.float64) - centroid
    return centred @ axes[0], centred @ axes[1]


def balance_index(positive: int, negative: int) -> float:
    """Symmetric, bounded split ratio ``(pos - neg) / (pos + neg)``, in ``[-1, 1]``.

    0 when the two sides hold equally many points; +1 = all on the positive side,
    -1 = all on the negative. Chosen over a raw ``pos/neg`` ratio because that is
    asymmetric (1 at balance, bounded below by 0 but unbounded above) and would make
    equal-and-opposite imbalances look numerically very different. NaN if no points.
    """
    total = positive + negative
    if total <= 0:
        return float("nan")
    return (positive - negative) / total


def directional_slope(
    s_along: np.ndarray, h: np.ndarray, top_pct: float
) -> tuple[float, float, float] | None:
    """Top/bottom strata slope lines along one axis, and the angle between them.

    Within one PCA axis, the points whose height is in the top ``top_pct`` percent and
    the bottom ``top_pct`` percent are each fit (least squares, degree 1) as a line in
    the ``(s_along, height)`` plane - i.e. a plane that may tilt only *along this axis*
    and stays flat across the other. Returns ``(top_slope, bottom_slope, angle_deg)``
    where the slopes are dh/ds and ``angle_deg`` is the angle between the two fitted
    lines (``|atan(top) - atan(bottom)|`` in degrees) - a terrain-robust measure of how
    differently the upper and lower canopy surfaces tilt. ``None`` if either stratum has
    < 2 points (no line) or the percentile is degenerate.
    """
    n = s_along.size
    if n < 2 or not (0.0 < top_pct < 50.0):
        return None
    hi_cut = np.percentile(h, 100.0 - top_pct)
    lo_cut = np.percentile(h, top_pct)
    top = s_along[h >= hi_cut]
    bottom = s_along[h <= lo_cut]
    if top.size < 2 or bottom.size < 2:
        return None
    # polyfit degree 1 -> [slope, intercept]; guard against a degenerate (single-x) stratum.
    if np.ptp(top) == 0 or np.ptp(bottom) == 0:
        return None
    top_slope = float(np.polyfit(top, h[h >= hi_cut], 1)[0])
    bottom_slope = float(np.polyfit(bottom, h[h <= lo_cut], 1)[0])
    angle = abs(np.degrees(np.arctan(top_slope) - np.arctan(bottom_slope)))
    return top_slope, bottom_slope, float(angle)


def directional_roughness(s_along: np.ndarray, h: np.ndarray) -> float:
    """1-D roughness along one axis: RMS of heights about their linear trend in ``s_along``.

    The directional analogue of ``features.sigma_z`` - detrend the height against this
    single axis (degree-1 fit) and take the RMS (ddof=0) of the residuals. Captures how
    bumpy the canopy top is *along this direction* specifically. NaN if < 2 points or the
    projection is degenerate.
    """
    if s_along.size < 2 or np.ptp(s_along) == 0:
        return float("nan")
    coef = np.polyfit(s_along, h, 1)
    residuals = h - np.polyval(coef, s_along)
    return float(np.std(residuals, ddof=0))

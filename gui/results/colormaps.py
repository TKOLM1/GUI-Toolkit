"""Shared colour-map choices for the results sub-tabs.

Both the map view (square fill colour) and the point-cloud view (per-cloud colour) let the user
pick a colour map from the same small menu and always stretch it over the *actual* data range
(never a hard-coded 0..30). Keeping the list and the green->red definition here means the two
tabs offer identical options and the viewer's ``error`` colours stay in sync with the map's.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

# The original results green -> yellow -> orange -> red ramp, offered as the default everywhere.
GREEN_RED = LinearSegmentedColormap.from_list(
    "green_red", ["#1a9850", "#f4e000", "#fd8d3c", "#d73027"]
)

# Menu of named maps: label -> matplotlib colormap name (GREEN_RED is the custom one above).
COLORMAP_NAMES = ["green→red", "jet", "viridis", "plasma", "turbo", "coolwarm", "Greys"]
_DEFAULT = "green→red"


def get_cmap(name: str):
    """Resolve a menu label to a matplotlib colormap (``green→red`` is our custom ramp)."""
    if name == _DEFAULT:
        return GREEN_RED
    import matplotlib

    return matplotlib.colormaps[name]


def full_range_norm(values) -> Normalize:
    """A :class:`Normalize` stretched over the finite min..max of ``values`` (never empty)."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    lo, hi = (float(arr.min()), float(arr.max())) if arr.size else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1.0
    return Normalize(vmin=lo, vmax=hi)


def qcolor_for(cmap, norm, value) -> "object":
    """A :class:`PySide6.QtGui.QColor` for ``value`` through ``cmap``/``norm`` (grey if non-finite)."""
    from PySide6.QtGui import QColor

    if value is None or not np.isfinite(value):
        return QColor("#dddddd")
    r, g, b, _ = cmap(norm(value))
    return QColor.fromRgbF(float(r), float(g), float(b))


def gradient_stops(cmap, n: int = 12) -> list[tuple[float, "object"]]:
    """``(position, QColor)`` stops sampling ``cmap`` from 0..1, for a Qt gradient legend."""
    from PySide6.QtGui import QColor

    stops = []
    for i in range(n):
        t = i / (n - 1)
        r, g, b, _ = cmap(t)
        stops.append((t, QColor.fromRgbF(float(r), float(g), float(b))))
    return stops


def text_color_for(facecolor) -> str:
    """Pick black or white text for readability on ``facecolor`` (any matplotlib colour).

    Uses the WCAG relative-luminance of the (already alpha-blended) RGB and flips to white once
    the background is dark enough, so labels stay legible across every colour map — light cells
    keep black text, dark cells (e.g. the high end of ``jet``/``Greys``) get white.
    """
    from matplotlib.colors import to_rgb

    r, g, b = to_rgb(facecolor)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luminance > 0.5 else "white"

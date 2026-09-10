"""A persistent polyscope viewer driven by a JSON control file.

Run as a *separate process* (``python tools/view_laz.py <control_file> [first.laz]``) so the
OpenGL viewer never shares the Qt GUI thread. Unlike a one-shot ``ps.show()``, this opens a
single window and then loops on :func:`polyscope.frame_tick`, re-reading a small JSON control
file each frame. When the file's ``seq`` advances, the current structures are cleared and the
new payload is loaded **in place** - so the GUI can swap the displayed cloud (or show several
clouds together) without ever opening a second window.

The viewer is deliberately the *only* place that imports polyscope: the GUI process must never
import it (one ``ps.init`` per process; Qt + GL in one process is unstable).

Control-file schema (all keys optional except ``seq``)::

    {
      "seq": 7,                       # monotonic; a higher value than last-applied = reload
      "origin": [x, y, z] | null,     # shared offset subtracted from absolute coords
      "clouds": [                     # one or more clouds to show together
        {"path": "...", "coloring": "error|rgb|height", "error": 12.3,
         "color": [r,g,b], "cmap": "viridis", "features": true}
      ],
      "features": {"path": "..."},    # optional feature geometry for one plot (map view)
      "slope_strata_pct": 5.0,        # optional: strata % for the hand-crafted slope planes
      "height_channel": "RelativeHeight"  # optional: the point dimension holding height-above-ground
    }

Colourings: ``rgb`` uses the file's colours (falls back to height when absent); ``height`` maps
the height channel through the selected matplotlib colour map (``cmap``); ``error``
paints the whole cloud one solid colour (either the precomputed GUI ``color`` or the built-in
green->red error scale) so many clouds read as a field map in 3D.

Height axis: points are drawn with their **height-above-ground** (the project's configured height
channel, e.g. ``RelativeHeight`` or ``Z``) as the Z coordinate, so the cloud lives in exactly the
coordinate system the height features are computed in - every feature plane/box then sits at its
literal feature value with no ground reconstruction.
Each cloud is split into a vegetation sub-cloud (classification == 1, the feature input) and a
``... non-veg`` sub-cloud (everything else); a single in-window checkbox hides the non-veg ones.

Feature geometry is registered per cloud and each item (every height-percentile plane, the
roughness plane, the bounding box, the convex hull, the above-mean points, plus the hand-crafted
Group G geometry - the two PCA axes and the top/bottom slope planes) gets its **own** in-window
checkbox with a colour swatch, so they can be toggled individually and the colours are
self-documenting. Everything starts disabled.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Run from anywhere: make the project root importable so ``featuregen.geometry`` (the shared
# feature/visualisation maths) resolves whatever the launch cwd is.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# --------------------------------------------------------------------------- #
# Colour helpers (kept tiny and dependency-free)                              #
# --------------------------------------------------------------------------- #
# The same green -> yellow -> orange -> red anchors the results map uses, and the error
# value (%) at which the colour saturates to red. Kept in sync with gui/results.
_ERROR_STOPS = np.array(
    [[0.10, 0.60, 0.31], [0.96, 0.88, 0.0], [0.99, 0.55, 0.24], [0.84, 0.19, 0.15]]
)
_MAX_ERROR = 30.0


# Every footprint-spanning visualisation is drawn this much larger than the tight point extent
# (1.10 = +10% in X and Y) so it overhangs the canopy edge and peeks out instead of being buried
# inside the cloud. Shared by the Cartesian-footprint planes (percentile, sigma_z), the PCA-frame
# slope planes, and the PCA axis segments, so they all overhang by the same amount. Wireframe
# outlines (bbox, hull) and the above-mean point cloud deliberately keep the tight extent (they're
# reference outlines / real data points) and do not use this.
_PLANE_OVERHANG = 1.10


def _scale_about_centroid(poly_xy: np.ndarray, factor: float) -> np.ndarray:
    """Scale a polygon's vertices outward from their centroid by ``factor`` (1.10 = +10%).

    Used to give the drawn percentile/roughness planes a small overhang past the canopy edge so
    the lower planes are not hidden inside the point cloud.
    """
    centroid = poly_xy.mean(axis=0)
    return centroid + (poly_xy - centroid) * factor


def _expand_about_mid(lo: float, hi: float, factor: float) -> tuple[float, float]:
    """Grow the interval ``[lo, hi]`` outward about its midpoint by ``factor`` (1.10 = +10%).

    The 1-D analogue of :func:`_scale_about_centroid`, used for the slope planes whose extent is
    expressed as min/max along each PCA axis rather than as polygon vertices - so they overhang the
    canopy edge by the same amount as the Cartesian-footprint planes.
    """
    mid = 0.5 * (lo + hi)
    return mid + (lo - mid) * factor, mid + (hi - mid) * factor


def _error_colour(error_pct: float) -> tuple[float, float, float]:
    """Map an error percentage to one RGB triple on the green->red scale."""
    if error_pct != error_pct:  # NaN -> grey
        return (0.6, 0.6, 0.6)
    t = max(0.0, min(1.0, float(error_pct) / _MAX_ERROR)) * (len(_ERROR_STOPS) - 1)
    lo = int(np.floor(t))
    hi = min(lo + 1, len(_ERROR_STOPS) - 1)
    frac = t - lo
    rgb = _ERROR_STOPS[lo] * (1 - frac) + _ERROR_STOPS[hi] * frac
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]))


def _cmap_colours(values: np.ndarray, cmap_name: str) -> np.ndarray:
    """Map a 1-D array to Nx3 colours through the selected colour map, normalised over its range.

    Replaces the old hard-coded jet ramp so the RelativeHeight colouring honours whichever colour
    map the user picked in the GUI. The GUI's custom ``green->red`` map (not a matplotlib name) is
    reproduced from the shared error stops; any other name is looked up in matplotlib (viridis as
    a final fallback).
    """
    import matplotlib

    v = np.asarray(values, dtype=np.float64)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return np.tile([0.5, 0.5, 0.5], (v.size, 1))
    lo, hi = float(finite.min()), float(finite.max())
    t = (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
    t = np.nan_to_num(np.clip(t, 0.0, 1.0), nan=0.0)  # non-finite -> low end (grey-ish), never a bad index
    if cmap_name in ("green→red", "green->red"):  # the GUI default; mirror its 4 stops
        idx = t * (len(_ERROR_STOPS) - 1)
        lo_i = np.floor(idx).astype(int)
        hi_i = np.minimum(lo_i + 1, len(_ERROR_STOPS) - 1)
        frac = (idx - lo_i)[:, None]
        return _ERROR_STOPS[lo_i] * (1 - frac) + _ERROR_STOPS[hi_i] * frac
    try:
        cmap = matplotlib.colormaps[cmap_name]
    except KeyError:
        cmap = matplotlib.colormaps["viridis"]
    return cmap(t)[:, :3]


# --------------------------------------------------------------------------- #
# Feature-geometry colours (self-documenting via the in-window swatches)      #
# --------------------------------------------------------------------------- #
# A per-percentile colour ramp (cool -> warm with rising percentile) so multiple planes read
# apart at a glance; the other categories get their own distinct colours.
_PLANE_COLOURS = {
    "h_p25": (0.20, 0.55, 0.90),
    "h_median": (0.20, 0.80, 0.75),
    "h_mean": (0.35, 0.80, 0.30),
    "h_p75": (0.95, 0.80, 0.20),
    "h_p95": (0.95, 0.55, 0.20),
    "h_max": (0.90, 0.25, 0.20),
}
_PLANE_DEFAULT_COLOUR = (0.70, 0.70, 0.70)
_SIGMA_COLOUR = (0.75, 0.40, 0.85)
_BBOX_COLOUR = (0.95, 0.95, 0.95)
_HULL_COLOUR = (0.30, 0.90, 0.95)
_ABOVE_MEAN_COLOUR = (1.0, 0.25, 0.25)
# Hand-crafted horizontal feature class (Group G) - its own distinct colours.
_AXIS_MAJOR_COLOUR = (1.0, 0.45, 0.0)    # the plot's long (major) PCA axis
_AXIS_MINOR_COLOUR = (0.0, 0.55, 1.0)    # the plot's short (minor) PCA axis
_SLOPE_TOP_COLOUR = (0.95, 0.85, 0.10)   # top-stratum slope plane
_SLOPE_BOT_COLOUR = (0.55, 0.20, 0.70)   # bottom-stratum slope plane
# Top/bottom height percentage for the drawn slope planes. Set from the control file's
# ``slope_strata_pct`` (the value the project's features were generated with) so the planes match
# the numbers; falls back to Config.slope_strata_pct's 5.0 default when the payload omits it.
_SLOPE_STRATA_PCT = [5.0]

# The point dimension holding height-above-ground, set from the control file's ``height_channel``
# (the project's configured Feature-tab channel) so the drawn cloud and feature geometry use the
# same height the scalar features did. Falls back to ``RelativeHeight`` when the payload omits it.
_HEIGHT_CHANNEL = ["RelativeHeight"]


# --------------------------------------------------------------------------- #
# Registries driving the in-window checkboxes                                 #
# --------------------------------------------------------------------------- #
# Non-veg points are always-available and share one global toggle.
_SHOW_NONVEG = [False]
_NONVEG_NAMES: list[str] = []

# Each feature-geometry item is one toggle: key -> {label, colour, show:[bool], names:[...]}.
# Built fresh in _apply_payload from whatever geometry registered, so several plots' matching
# items share one checkbox (e.g. one "h_p95 plane" toggle flips every selected plot's p95 plane).
_FEATURE_ITEMS: dict[str, dict] = {}
_FEATURE_LEGEND_ACTIVE = [False]


def _load_xyz_rgb_height(path: Path):
    """Return ``(xyz, rgb_or_None, height_or_None, veg_mask)`` for a .las/.laz file.

    The returned ``xyz`` uses absolute X and Y but the **configured height channel as Z** (falling
    back to the absolute z only when that channel is genuinely absent) - so the drawn cloud and the
    height features share one coordinate system. The channel is read scale-aware (so picking ``Z``
    yields metres, not raw integers). ``veg_mask`` flags the vegetation points (classification == 1),
    the exact points the features are computed over.
    """
    import laspy

    from featuregen.io_las import read_channel

    las = laspy.read(path)
    dims = set(las.point_format.dimension_names)
    channel = _HEIGHT_CHANNEL[0]
    height = read_channel(las, channel) if channel in dims else None
    z_axis = height if height is not None else np.asarray(las.z, dtype=np.float64)
    xyz = np.column_stack([np.asarray(las.x), np.asarray(las.y), z_axis]).astype(np.float64)

    rgb = None
    if {"red", "green", "blue"} <= dims:
        rgb = np.column_stack(
            [np.asarray(las.red), np.asarray(las.green), np.asarray(las.blue)]
        ).astype(np.float64)
        peak = rgb.max() if rgb.size else 0
        if peak > 0:
            rgb = rgb / peak
    veg_mask = np.asarray(las.classification) == 1
    return xyz, rgb, height, veg_mask


def _register_cloud(spec: dict, origin: np.ndarray) -> None:
    """Register one cloud as a vegetation sub-cloud + a toggleable ``... non-veg`` sub-cloud.

    Splitting on classification keeps the vegetation points (the feature input) as the primary
    structure while letting a single in-window checkbox hide the non-vegetation context.
    """
    import polyscope as ps

    path = Path(spec["path"])
    if not path.exists():
        return
    xyz, rgb, height, veg = _load_xyz_rgb_height(path)
    if not len(xyz):
        return
    xyz = xyz - origin  # shared offset keeps multi-cloud coords near the origin

    coloring = spec.get("coloring", "rgb")
    solid = spec.get("color")  # a precomputed [r,g,b] (e.g. the GUI's colour-mapped value)
    cmap_name = spec.get("cmap", "viridis")
    # Register vegetation as the named cloud and the remainder as a "... non-veg" sibling, so the
    # primary handle is always the feature input. Same colouring is applied to both halves.
    for name, mask, enabled in ((path.stem, veg, True),
                                (f"{path.stem} non-veg", ~veg, _SHOW_NONVEG[0])):
        if not mask.any():
            continue
        cloud = ps.register_point_cloud(name, xyz[mask], enabled=enabled)
        if solid is not None:
            cloud.set_color(tuple(float(c) for c in solid))
        elif coloring == "error":
            cloud.set_color(_error_colour(spec.get("error", float("nan"))))
        elif coloring == "height" and height is not None:
            cloud.add_color_quantity("RelativeHeight", _cmap_colours(height[mask], cmap_name),
                                     enabled=True)
        elif rgb is not None:
            cloud.add_color_quantity("RGB", rgb[mask], enabled=True)
        elif height is not None:
            cloud.add_color_quantity("RelativeHeight", _cmap_colours(height[mask], cmap_name),
                                     enabled=True)
        if mask is not veg:
            _NONVEG_NAMES.append(name)


# What each feature-geometry category maps to - shown in the in-window legend.
_FEATURE_LEGEND = [
    ("Height percentile planes",
     "horizontal planes at h_mean, h_median, h_p25, h_p75, h_p95, h_max (the central-tendency "
     "features), each spanning the plot's XY extent. Toggle each individually."),
    ("Roughness fit plane",
     "the least-squares plane z = a*x + b*y + c whose residual RMS is the sigma_z roughness "
     "feature - the tilt it removes before measuring bumpiness."),
    ("Bounding box / convex hull",
     "the vegetation points' XY axis-aligned extent and 2.5-D convex-hull footprint."),
    ("Above-mean points",
     "vegetation points above the mean height highlighted - the subset measured by "
     "frac_above_mean."),
    ("PCA axes (hand-crafted)",
     "the plot's own major (long) and minor (short) axes found by a 2-D PCA of the "
     "vegetation XY - the frame the hand-crafted horizontal features are computed in."),
    ("Top/bottom slope planes (hand-crafted)",
     "for each PCA axis, the surfaces fitted to the top and bottom height strata, each "
     "tilting only along that axis. The angle between a top and bottom plane is the "
     "slope_angle feature; planes use the default strata percentage."),
]


def _feature_item(key: str, label: str, colour: tuple) -> dict:
    """Get-or-create the toggle registry entry for a feature-geometry item."""
    item = _FEATURE_ITEMS.get(key)
    if item is None:
        item = {"label": label, "colour": colour, "show": [False], "names": []}
        _FEATURE_ITEMS[key] = item
    return item


def _register_feature_geometry(spec: dict, origin: np.ndarray) -> None:
    """Register the feature visualisations for one cloud's vegetation points.

    Everything is computed from :mod:`featuregen.geometry` - the same maths the scalar features
    use - so a drawn plane sits at exactly the value shown on the map. Each item is registered
    under its own toggle key (per-percentile plane, sigma_z, bbox, hull, above-mean) and given a
    distinct colour, so it can be toggled individually and the swatches document the colours.
    """
    import polyscope as ps

    from featuregen import geometry as geo

    path = Path(spec["path"])
    if not path.exists():
        return
    xyz, _, height, veg = _load_xyz_rgb_height(path)
    if height is None or not veg.any():
        return
    stem = path.stem

    def _add(key: str, label: str, colour: tuple, register):
        """Register a structure (via ``register(name, enabled)``), recording it on its toggle."""
        item = _feature_item(key, label, colour)
        name = f"{stem} | {label}"
        struct = register(name, item["show"][0])
        item["names"].append(name)
        return struct

    # xyz already carries the configured height channel as its Z axis, so the vegetation height h
    # *is* vz. X and Y keep the shared origin offset for multi-cloud positioning.
    vx = xyz[veg, 0] - origin[0]
    vy = xyz[veg, 1] - origin[1]
    h = height[veg]

    xmin, xmax, ymin, ymax = geo.xy_bbox(vx, vy)

    # Planes are drawn over the **convex-hull footprint** (fan-triangulated), so they hug the
    # actual point extent instead of an axis-aligned rectangle that overshoots a rotated plot.
    # The hull is computed once and reused for both the planes and the hull wireframe; a degenerate
    # hull (collinear / <3 points) falls back to the bbox quad. ``footprint_xy`` is the ordered
    # polygon (k, 2); ``footprint_faces`` its triangle fan.
    hull = geo.convex_hull_2d(vx, vy)
    if hull is not None and len(hull) >= 3:
        footprint_xy = hull
        footprint_faces = np.array([[0, i, i + 1] for i in range(1, len(hull) - 1)])
    else:
        footprint_xy = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
        footprint_faces = np.array([[0, 1, 2], [0, 2, 3]])

    # Planes are drawn over a footprint scaled 10% outward from its centroid, so they overhang the
    # canopy edge a little and the lower-percentile planes (e.g. h_p25) peek out instead of being
    # buried inside the point cloud. The wireframe hull below still uses the tight footprint.
    plane_xy = _scale_about_centroid(footprint_xy, _PLANE_OVERHANG)

    # (a) Height percentile planes - one translucent polygon per central-tendency feature at its
    # literal RelativeHeight level, each its own toggle + colour.
    for pkey, level in geo.height_percentiles(h).items():
        colour = _PLANE_COLOURS.get(pkey, _PLANE_DEFAULT_COLOUR)
        verts = np.column_stack([plane_xy, np.full(len(plane_xy), level)])

        def _reg(name, enabled, verts=verts, colour=colour):
            mesh = ps.register_surface_mesh(name, verts, footprint_faces, enabled=enabled)
            mesh.set_transparency(0.4)
            mesh.set_color(colour)
            return mesh

        _add(f"plane:{pkey}", f"plane {pkey}", colour, _reg)

    # (b) Roughness fit plane (sigma_z) - the lstsq plane sampled over the same expanded footprint.
    coef, _ = geo.fit_plane(vx, vy, h)
    sigma_z = geo.plane_z(coef, plane_xy[:, 0], plane_xy[:, 1], vx.mean(), vy.mean())
    sigma_verts = np.column_stack([plane_xy, sigma_z])

    def _reg_sigma(name, enabled):
        mesh = ps.register_surface_mesh(name, sigma_verts, footprint_faces, enabled=enabled)
        mesh.set_transparency(0.5)
        mesh.set_color(_SIGMA_COLOUR)
        return mesh

    _add("sigma_z", "sigma_z plane", _SIGMA_COLOUR, _reg_sigma)

    # (c) Bounding box + 2.5-D convex hull as wireframes (two separate toggles).
    zlo, zhi = float(h.min()), float(h.max())
    box_nodes = np.array(
        [[xmin, ymin, zlo], [xmax, ymin, zlo], [xmax, ymax, zlo], [xmin, ymax, zlo],
         [xmin, ymin, zhi], [xmax, ymin, zhi], [xmax, ymax, zhi], [xmin, ymax, zhi]]
    )
    box_edges = np.array(
        [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
         [0, 4], [1, 5], [2, 6], [3, 7]]
    )

    def _reg_box(name, enabled):
        net = ps.register_curve_network(name, box_nodes, box_edges, enabled=enabled)
        net.set_color(_BBOX_COLOUR)
        return net

    _add("bbox", "bbox", _BBOX_COLOUR, _reg_box)

    # Reuse the hull computed above for the plane footprints (no second ConvexHull pass).
    if hull is not None:
        hull_nodes = np.column_stack([hull, np.full(len(hull), zlo)])
        hull_edges = np.column_stack([np.arange(len(hull)), (np.arange(len(hull)) + 1) % len(hull)])

        def _reg_hull(name, enabled, nodes=hull_nodes, edges=hull_edges):
            net = ps.register_curve_network(name, nodes, edges, enabled=enabled)
            net.set_color(_HULL_COLOUR)
            return net

        _add("hull", "convex hull", _HULL_COLOUR, _reg_hull)

    # (d) Above-mean points - their own cloud, vivid + a larger radius so they stand out from the
    # vegetation cloud they sit on top of (the previous version was hidden under the main cloud).
    above = geo.above_mean_mask(h)
    if above.any():
        pts = np.column_stack([vx[above], vy[above], h[above]])

        def _reg_above(name, enabled, pts=pts):
            cloud = ps.register_point_cloud(name, pts, enabled=enabled)
            cloud.set_color(_ABOVE_MEAN_COLOUR)
            cloud.set_radius(0.012, relative=True)  # larger than the default so it reads on top
            return cloud

        _add("above_mean", "above-mean points", _ABOVE_MEAN_COLOUR, _reg_above)

    # (e) Hand-crafted horizontal feature class (Group G), all in the plot's PCA axis frame.
    _register_horizontal_geometry(_add, vx, vy, h)


def _register_horizontal_geometry(_add, vx, vy, h) -> None:
    """Draw the hand-crafted horizontal features' geometry: PCA axes + top/bottom slope planes.

    Computed from the same ``featuregen.geometry`` helpers the scalar features use, so the drawn
    axes and slope planes match the numbers. ``_add(key, label, colour, register)`` registers a
    structure on its own in-window toggle (see :func:`_register_feature_geometry`).
    """
    import polyscope as ps

    from featuregen import geometry as geo

    res = geo.pca_axes_2d(vx, vy)
    if res is None:
        return
    centroid, axes, eigvals = res
    s_major, s_minor = geo.project_to_axes(vx, vy, centroid, axes)
    zlo, zhi = float(h.min()), float(h.max())
    zmid = 0.5 * (zlo + zhi)

    # (i) The two PCA axes as line segments through the centroid, each spanning its own extent,
    # drawn at mid-height so they read against the cloud. Major = long axis, minor = short axis.
    for key, label, colour, s_axis, axis_vec in (
        ("pca_major", "PCA major axis", _AXIS_MAJOR_COLOUR, s_major, axes[0]),
        ("pca_minor", "PCA minor axis", _AXIS_MINOR_COLOUR, s_minor, axes[1]),
    ):
        # Extend the segment the same +10% past the point extent as the planes, so its endpoints
        # poke out of the canopy edge instead of ending buried inside the cloud.
        lo, hi = _expand_about_mid(float(s_axis.min()), float(s_axis.max()), _PLANE_OVERHANG)
        p_lo = centroid + axis_vec * lo
        p_hi = centroid + axis_vec * hi
        nodes = np.array([[p_lo[0], p_lo[1], zmid], [p_hi[0], p_hi[1], zmid]])
        edges = np.array([[0, 1]])

        def _reg_axis(name, enabled, nodes=nodes, edges=edges, colour=colour):
            net = ps.register_curve_network(name, nodes, edges, enabled=enabled)
            net.set_color(colour)
            net.set_radius(0.004, relative=True)
            return net

        _add(key, label, colour, _reg_axis)

    # (ii) Top/bottom slope planes for each axis. A plane is the fitted line (dh/ds along its own
    # axis) extruded flat across the *other* axis - exactly the one-direction tilt the slope-angle
    # feature measures - so the visible angle between the top and bottom planes IS the feature.
    # The percentile mirrors the project's Config.slope_strata_pct (set from the control file).
    pct = _SLOPE_STRATA_PCT[0]
    for axis_name, s_along, s_across, along_vec, across_vec in (
        ("major", s_major, s_minor, axes[0], axes[1]),
        ("minor", s_minor, s_major, axes[1], axes[0]),
    ):
        out = geo.directional_slope(s_along, h, pct)
        if out is None:
            continue
        # Overhang the canopy edge by the same +10% as the percentile/sigma_z planes, here in the
        # PCA-axis frame: expand the along- and across-axis spans about their midpoints. The tilt
        # below (z follows ``a``) then extends with the widened surface, matching how the
        # Cartesian-footprint planes' Z tracks their expanded XY.
        a_lo, a_hi = _expand_about_mid(float(s_along.min()), float(s_along.max()), _PLANE_OVERHANG)
        c_lo, c_hi = _expand_about_mid(float(s_across.min()), float(s_across.max()), _PLANE_OVERHANG)
        for stratum, slope, colour in (
            ("top", out[0], _SLOPE_TOP_COLOUR),
            ("bottom", out[1], _SLOPE_BOT_COLOUR),
        ):
            # Intercept the plane at the stratum's own mean height so it sits where the points are.
            cut = np.percentile(h, 100.0 - pct) if stratum == "top" \
                else np.percentile(h, pct)
            mask = (h >= cut) if stratum == "top" else (h <= cut)
            s0 = float(s_along[mask].mean())
            h0 = float(h[mask].mean())
            # Four corners: vary s_along (height tilts by `slope`) x s_across (height constant).
            verts = []
            for a in (a_lo, a_hi):
                z = h0 + slope * (a - s0)
                for c in (c_lo, c_hi):
                    xy = centroid + along_vec * a + across_vec * c
                    verts.append([xy[0], xy[1], z])
            verts = np.array(verts)            # order: (a_lo,c_lo),(a_lo,c_hi),(a_hi,c_lo),(a_hi,c_hi)
            faces = np.array([[0, 1, 3], [0, 3, 2]])

            def _reg_slope(name, enabled, verts=verts, faces=faces, colour=colour):
                mesh = ps.register_surface_mesh(name, verts, faces, enabled=enabled)
                mesh.set_transparency(0.5)
                mesh.set_color(colour)
                return mesh

            _add(f"slope_{axis_name}_{stratum}", f"slope {axis_name} ({stratum} {pct:g}%)",
                 colour, _reg_slope)


def _set_enabled(name: str, show: bool) -> None:
    """Enable/disable a registered structure by name, whatever its type."""
    import polyscope as ps

    for has, get in ((ps.has_point_cloud, ps.get_point_cloud),
                     (ps.has_surface_mesh, ps.get_surface_mesh),
                     (ps.has_curve_network, ps.get_curve_network)):
        if has(name):
            get(name).set_enabled(show)
            return


# Collapse polyscope's own panels once, a few frames in (they must exist before SetWindowCollapsed
# takes; the first couple of frames they may not be laid out yet).
_COLLAPSE_FRAMES = [3]


def _collapse_default_panels(imgui) -> None:
    """One-shot: fold polyscope's built-in panels so the viewer opens uncluttered."""
    if _COLLAPSE_FRAMES[0] <= 0:
        return
    _COLLAPSE_FRAMES[0] -= 1
    for name in ("Polyscope", "Structures", "Selection"):
        imgui.SetWindowCollapsed(name, True)


# Feature items are listed in a stable, readable order regardless of dict insertion.
_ITEM_ORDER = ["plane:h_p25", "plane:h_median", "plane:h_mean", "plane:h_p75", "plane:h_p95",
               "plane:h_max", "sigma_z", "bbox", "hull", "above_mean",
               "pca_major", "pca_minor",
               "slope_major_top", "slope_major_bottom", "slope_minor_top", "slope_minor_bottom"]


def _viewer_ui_callback() -> None:
    """Polyscope per-frame UI: the per-item toggles (with colour swatches) + the legend."""
    import polyscope.imgui as imgui

    _collapse_default_panels(imgui)

    # The always-present non-veg toggle.
    if _NONVEG_NAMES:
        changed, show = imgui.Checkbox("Show non-vegetation points", _SHOW_NONVEG[0])
        if changed:
            _SHOW_NONVEG[0] = show
            for name in _NONVEG_NAMES:
                _set_enabled(name, show)

    if not _FEATURE_ITEMS:
        return

    imgui.Separator()
    imgui.TextUnformatted("Feature visualisations (colour = item):")
    keys = [k for k in _ITEM_ORDER if k in _FEATURE_ITEMS]
    keys += [k for k in _FEATURE_ITEMS if k not in keys]  # any unexpected items, after the known
    for key in keys:
        item = _FEATURE_ITEMS[key]
        r, g, b = item["colour"]
        # A small colour swatch so enabling several at once stays legible.
        imgui.ColorButton(f"##swatch_{key}", (r, g, b, 1.0))
        imgui.SameLine()
        changed, show = imgui.Checkbox(item["label"], item["show"][0])
        if changed:
            item["show"][0] = show
            for name in item["names"]:
                _set_enabled(name, show)

    if not _FEATURE_LEGEND_ACTIVE[0]:
        return
    imgui.Separator()
    imgui.TextUnformatted("What each visualisation maps to:")
    for title, desc in _FEATURE_LEGEND:
        imgui.Bullet()
        imgui.TextUnformatted(title)
        imgui.TextWrapped(f"    {desc}")


def _apply_payload(payload: dict) -> None:
    """Clear all structures and register everything the payload asks for."""
    import polyscope as ps

    ps.remove_all_structures()
    _NONVEG_NAMES.clear()
    _FEATURE_ITEMS.clear()  # rebuilt below; drives the per-item checkboxes
    # The slope planes use the project's strata % when the control file carries it (else the default).
    pct = payload.get("slope_strata_pct")
    if pct is not None:
        try:
            _SLOPE_STRATA_PCT[0] = float(pct)
        except (TypeError, ValueError):
            pass
    # The drawn cloud + feature geometry use the project's configured height channel when the
    # control file carries it (else the RelativeHeight default), matching the scalar features.
    channel = payload.get("height_channel")
    if isinstance(channel, str) and channel.strip():
        _HEIGHT_CHANNEL[0] = channel.strip()
    origin = np.asarray(payload.get("origin") or [0.0, 0.0, 0.0], dtype=np.float64)
    origin = origin.astype(np.float64).copy()
    origin[2] = 0.0  # Z is RelativeHeight; never offset it (offsetting Z blanks the scene)
    clouds = payload.get("clouds") or []
    # A bare cold-start cloud (no shared origin) centres on itself for comfortable framing. Only
    # X and Y matter for framing now that Z is RelativeHeight.
    if len(clouds) == 1 and payload.get("origin") is None:
        path = Path(clouds[0]["path"])
        if path.exists():
            xyz, *_ = _load_xyz_rgb_height(path)
            if len(xyz):
                origin = xyz.mean(axis=0)
                origin[2] = 0.0
    for spec in clouds:
        _register_cloud(spec, origin)

    # Feature geometry: either a top-level ``features`` (map view's single plot) or any cloud
    # flagged ``"features": true`` (point-cloud view's selected plots). Drawn per cloud so several
    # plots' matching items toggle together.
    feature_specs = []
    if (top := payload.get("features")) and top.get("path"):
        feature_specs.append(top)
    feature_specs += [c for c in clouds if c.get("features")]
    for spec in feature_specs:
        _register_feature_geometry(spec, origin)
    _FEATURE_LEGEND_ACTIVE[0] = bool(feature_specs)

    # The UI callback is always installed: it hosts the toggles (and the legend when geometry is
    # present).
    ps.set_user_callback(_viewer_ui_callback)
    ps.reset_camera_to_home_view()


# --------------------------------------------------------------------------- #
# Control-file polling + main loop                                            #
# --------------------------------------------------------------------------- #
def _read_control(path: Path) -> dict | None:
    """Read the JSON control file, tolerating a half-written, locked or absent file.

    The reader is opened and closed immediately (read the bytes, parse afterwards) so the GUI's
    atomic ``os.replace`` is rarely blocked by an open read handle on Windows.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, OSError):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: view_laz.py <control_file.json> [first_cloud.las|.laz]")
        return 2
    control_path = Path(argv[1])

    import polyscope as ps

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_background_color((0.0, 0.0, 0.0))
    ps.set_ground_plane_mode("none")

    last_seq = -1
    last_mtime = -1.0

    # Cold start: if a first cloud was passed on argv, show it before any control file lands.
    if len(argv) >= 3 and Path(argv[2]).exists():
        _apply_payload({"clouds": [{"path": argv[2], "coloring": "rgb"}]})

    while not ps.window_requests_close():
        # Only touch the file when it actually changed, so the GUI's atomic rename is seldom
        # racing an open read handle (a Windows ACCESS_DENIED source).
        try:
            mtime = control_path.stat().st_mtime
        except OSError:
            mtime = last_mtime
        if mtime != last_mtime:
            last_mtime = mtime
            payload = _read_control(control_path)
            if payload is not None and int(payload.get("seq", -1)) > last_seq:
                last_seq = int(payload["seq"])
                try:
                    _apply_payload(payload)
                except Exception as exc:  # noqa: BLE001 - one bad payload must not kill the viewer
                    print(f"viewer: could not apply payload: {exc}")
        ps.frame_tick()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

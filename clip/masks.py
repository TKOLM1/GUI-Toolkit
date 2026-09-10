"""Reading the plot-mask polygons from a GeoPackage (``.gpkg``).

The clipping tab takes a whole-field LiDAR cloud and a ``.gpkg`` of labelled plot
polygons and clips the cloud to each polygon. This module is the thin, GUI-free reader:

* :func:`list_mask_fields` lists the attribute columns so the GUI can offer one as the
  "plot label" field;
* :func:`load_masks` turns the chosen label column + the polygon geometries into a list
  of :class:`PlotMask` (a label + a shapely geometry), which :mod:`clip.clipper` then
  clips against.

The chosen column's values are read through the user's :class:`~common.naming.LabelFormat`
(the import tab's "starts with" / "ends with" boxes), exactly as imported *file names* are, so a
mask labelled ``P_429_a`` still yields ``plot(429)``. This matters more than it looks: every later
stage finds a plot by the integer inside ``plot(...)``, so a label that is not reduced to a plain
number produces files that are written successfully and then silently ignored by augmentation,
feature generation and ML. Polygons whose label cannot be read as a number are therefore skipped
and reported rather than exported under an unusable name.

Geometry is read with geopandas (pyogrio backend - pure wheels, no system GDAL). The
``.gpkg`` and the cloud are assumed to share a coordinate system (the field's projected
CRS, e.g. EPSG:32636); the clip is a direct (x, y) point-in-polygon test, so getting the
label straight from the polygon is what removes the old import-time misalignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd

from common.naming import LabelFormat, plot_number_from_label


@dataclass
class PlotMask:
    """One plot polygon: the label that becomes ``plot(<label>)`` and its geometry."""

    label: int             # the plot number read out of the chosen attribute value
    geometry: object       # a shapely (Multi)Polygon in the mask's CRS


def list_mask_fields(gpkg_path: str | Path) -> list[str]:
    """Return the non-geometry attribute column names of ``gpkg_path`` (label candidates)."""
    gdf = gpd.read_file(gpkg_path)
    geom_col = gdf.geometry.name
    return [str(c) for c in gdf.columns if c != geom_col]


def load_masks(
    gpkg_path: str | Path,
    label_field: str,
    fmt: LabelFormat | None = None,
) -> tuple[list[PlotMask], str, list[str]]:
    """Read ``gpkg_path`` into ``(masks, crs, unreadable)`` keyed on ``label_field``.

    Each polygon's ``label_field`` value is reduced to a plot number through ``fmt`` (the active
    :data:`common.naming.LABEL_FORMAT` when not given), so both a bare ``429`` and a decorated
    ``P_429_a`` resolve to ``429``.

    ``unreadable`` lists the label values that yielded no number; those polygons are left out of
    ``masks`` (see the module note on why exporting them would be worse than skipping them).
    Rows with a missing/empty geometry are dropped silently. ``crs`` is the layer's CRS as a
    string, e.g. ``"EPSG:32636"``.

    Raises
    ------
    KeyError
        If ``label_field`` is not a column of the GeoPackage.
    ValueError
        If the file has no usable polygons at all.
    """
    gdf = gpd.read_file(gpkg_path)
    if label_field not in gdf.columns:
        available = ", ".join(str(c) for c in gdf.columns)
        raise KeyError(f"Label field '{label_field}' not in {Path(gpkg_path).name}. "
                       f"Available columns: {available}")

    masks: list[PlotMask] = []
    unreadable: list[str] = []
    for label, geom in zip(gdf[label_field], gdf.geometry):
        if geom is None or geom.is_empty:
            continue
        number = plot_number_from_label(label, fmt)
        if number is None:
            unreadable.append(str(label))
            continue
        masks.append(PlotMask(label=number, geometry=geom))
    if not masks:
        raise ValueError(
            f"No usable polygons found in {Path(gpkg_path).name}."
            + (f" {len(unreadable)} polygon(s) had a label with no readable plot number "
               f"(e.g. '{unreadable[0]}') - check the 'Label starts with / ends with' boxes."
               if unreadable else "")
        )

    crs = str(gdf.crs) if gdf.crs is not None else "unknown"
    return masks, crs, unreadable

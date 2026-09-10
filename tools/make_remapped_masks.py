"""Write a copy of the plot-mask GeoPackage with the label-remap applied.

Geometry is kept *exactly* as digitised. The remap (the 180 deg numbering swap in
``plot_label_remap.csv``) describes which plot number physically belongs to each polygon, so
the **whole attribute row** (plot, Block, variety, Species, ...) is relocated to the geometry
of its swap partner - not just the ``plot`` label. In other words: the polygon QGIS labelled
``429`` keeps its shape but receives every attribute of the row that should physically sit
there. Each original attribute is also preserved under an ``orig_*`` column so the corrected
and original assignments can be compared field-by-field in QGIS.

Usage:
    python tools/make_remapped_masks.py [SRC.gpkg] [REMAP.csv] [OUT.gpkg]
Defaults point at the project's standard files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd

# Make the project root importable so we can reuse the exact same remap loader the app uses.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clip.masks import load_label_remap  # noqa: E402

DEFAULT_SRC = Path(r"c:/Projects (VScode)/GP 2026/Data/masks 32636.gpkg")
DEFAULT_CSV = Path(r"c:/Projects (VScode)/GP 2026/Python/General Pipeline/plot_label_remap.csv")
DEFAULT_OUT = Path(r"c:/Projects (VScode)/GP 2026/Data/masks 32636 remapped.gpkg")
LABEL_FIELD = "plot"


def _as_int(value: object) -> object:
    """Coerce a whole-numbered label to int so it matches the remap dict keys."""
    s = str(value).strip()
    return int(float(s)) if s.lstrip("-").replace(".", "", 1).isdigit() else value


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC
    csv = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CSV
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_OUT

    gdf = gpd.read_file(src)
    geom_col = gdf.geometry.name
    remap = load_label_remap(csv)

    # ``fid`` is a file-internal feature ID, not plot data: the GeoPackage writer reassigns it
    # (which reorders rows), so drop it before remapping and let the writer create fresh IDs.
    if "fid" in gdf.columns:
        gdf = gdf.drop(columns=["fid"])

    attr_cols = [c for c in gdf.columns if c != geom_col]

    # Map: current plot label -> the row index that currently carries it.
    key_of_row = gdf[LABEL_FIELD].apply(_as_int)
    row_by_label = {key_of_row.iloc[i]: i for i in range(len(gdf))}

    # For each polygon (kept in place), find which physical plot belongs there and pull THAT
    # plot's whole attribute row onto this geometry. The remap is an involution, so the source
    # polygon for geometry-at-label `cur` is the one currently labelled `remap[cur]`.
    new_rows = []
    moved = 0
    for i in range(len(gdf)):
        cur = key_of_row.iloc[i]
        target = remap.get(cur, cur)                 # physical plot that belongs on this polygon
        src_idx = row_by_label.get(target, i)        # row currently carrying that plot's attributes
        src_attrs = {c: gdf.iloc[src_idx][c] for c in attr_cols}
        orig_attrs = {f"orig_{c}": gdf.iloc[i][c] for c in attr_cols}
        new_rows.append({**src_attrs, **orig_attrs, geom_col: gdf.iloc[i][geom_col]})
        if src_idx != i:
            moved += 1

    result = gpd.GeoDataFrame(new_rows, geometry=geom_col, crs=gdf.crs)
    result.to_file(out, driver="GPKG")

    print(f"Read   {len(gdf)} polygons from {src.name}")
    print(f"Remap  {len(remap)} entries from {csv.name}")
    print(f"Moved  attribute rows on {moved} of {len(gdf)} polygons "
          f"(columns moved: {', '.join(attr_cols)}; geometry unchanged)")
    print(f"Wrote  {out}")


if __name__ == "__main__":
    main()

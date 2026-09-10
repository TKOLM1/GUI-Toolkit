"""clip - get plot point clouds into a project, by clipping or by importing (the Import tab).

Two routes, one destination (canonical ``plot(N)`` files in the project's ``plots/`` folder):

* **Clip** one whole-field cloud with a ``.gpkg`` of labelled plot polygons -
  ``PlotMask`` / ``list_mask_fields`` / ``load_masks`` read the masks, ``clip_to_masks`` cuts.
* **Import** a folder of already-separated per-plot files - ``list_sources`` finds them,
  ``plan_import`` reads each name into a ``common.naming.PlotKey`` (a column, optionally paired
  with a row) and derives its canonical plot number, ``import_plots`` converts and writes them.
  ``.pcd`` sources are read by ``read_pcd``.

Both routes derive the plot number from the source data rather than assigning one, so an import
is reproducible and the reference sheet - re-keyed through the same rule - stays joined to it.
"""

from .masks import PlotMask, list_mask_fields, load_masks
from .clipper import ClipResult, clip_to_masks
from .importer import (
    DEFAULT_UNIT,
    IMPORT_SUFFIXES,
    MANIFEST_NAME,
    GridLayout,
    ImportPlan,
    ImportResult,
    UNIT_SCALES,
    import_plots,
    list_sources,
    plan_import,
    detect_units,
    unit_name,
    unit_scale,
)
from .pcd import PcdError, read_pcd

__all__ = [
    "PlotMask",
    "list_mask_fields",
    "load_masks",
    "ClipResult",
    "clip_to_masks",
    "DEFAULT_UNIT",
    "IMPORT_SUFFIXES",
    "MANIFEST_NAME",
    "GridLayout",
    "ImportPlan",
    "ImportResult",
    "UNIT_SCALES",
    "import_plots",
    "list_sources",
    "plan_import",
    "detect_units",
    "unit_name",
    "unit_scale",
    "PcdError",
    "read_pcd",
]

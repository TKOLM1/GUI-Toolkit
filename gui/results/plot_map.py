"""A minimal, read-only 2-D field map: the plot footprints with their numbers inside.

The Results tab's :class:`~gui.results.map_view.MapView` is the full-featured map (colour maps,
per-plot displays, 3-D hand-off, grid layout). The Clipping tab wants none of that — just a look at
*where the plots landed* right after clipping — so this widget reuses the same
:class:`~gui.results.field_canvas.FieldCanvas` (pan / zoom / Home) with a single flat fill and one
label per plot: its number. Nothing is selectable and nothing is linked to a model.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ml.plot_geometry import build_plot_geoms
from .field_canvas import FieldCanvas, PlotPatch, box_angle_deg

# A flat, neutral fill: this map shows geometry only, so no value is being encoded by colour.
_FILL = QColor("#b7d7f0")


class PlotMapView(QWidget):
    """The plain field map (footprints + plot numbers) built straight from a folder of plots."""

    def __init__(self, title: str = "Plot layout") -> None:
        super().__init__()
        self._title = title
        self._empty_reason = "No plots yet."

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = FieldCanvas()
        self.canvas.colorbar.setVisible(False)  # nothing is colour-coded here
        # Keep the canvas short enough that the hosting page never needs to scroll to see it,
        # while still giving the field room to read.
        self.canvas.setMinimumHeight(220)
        self.message = QLabel(self._empty_reason)
        self.message.setStyleSheet("color: gray;")
        self.message.setWordWrap(True)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.message)

    # ------------------------------------------------------------------ #
    def refresh(self, file_map: dict) -> None:
        """Redraw from ``{(plot, aug): path}`` (only the originals, ``aug == 0``, are drawn)."""
        geoms = build_plot_geoms(file_map, aug_index=0) if file_map else {}
        patches = [
            PlotPatch(plot=plot, polygon=geom.hull, fill=_FILL,
                      angle_deg=box_angle_deg(geom.hull))
            for plot, geom in sorted(geoms.items())
            if geom.hull is not None
        ]
        self.canvas.set_scene(
            patches, field_mode=True, title=self._title if patches else "",
            label_for=lambda plot: [(str(plot), 0, True)],
        )
        self.message.setText(
            f"{len(patches)} plot footprint(s). Scroll to zoom, drag to pan."
            if patches else self._empty_reason
        )

    def clear(self) -> None:
        """Empty the map (used when the project changes)."""
        self.refresh({})


def file_map_from_folder(folder: Path | None) -> dict:
    """``{(plot, 0): path}`` for the plot clouds directly in ``folder`` (augmented copies skipped).

    The Clipping tab's map only ever wants the originals it just produced, so this is a much
    simpler scan than the Results tab's (which also tracks augmentation levels).
    """
    from common.naming import LAS_SUFFIXES, aug_number_from_name, plot_number_from_name

    if not folder or not Path(folder).is_dir():
        return {}
    out: dict = {}
    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lower() not in LAS_SUFFIXES or aug_number_from_name(path.name):
            continue
        plot = plot_number_from_name(path.name)
        if plot is not None:
            out.setdefault((plot, 0), path)
    return out

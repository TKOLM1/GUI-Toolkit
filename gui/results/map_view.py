"""Results sub-tab 1: the field map, with selectable per-plot displays.

The map is drawn on a hardware-accelerated :class:`gui.results.field_canvas.FieldCanvas`
(QGraphicsView), so pan, zoom and rotate are view transforms that stay at 60 fps+ regardless of
plot count — the previous matplotlib canvas re-rendered the whole figure on every drag and capped
the field view at ~10 fps.

Two views share the canvas. The **field map** draws each plot as its real oriented footprint at
its true position, with a single value drawn inside it, sized to fill the box and rotated to match
the plot, plus a slider to rotate the whole field. The **grid** draws uniform squares from a chosen
layout, each with up to three stacked value lines (the schematic view). The fill colour is driven
by any field through a selectable colour map (default green->red), stretched over the data range;
held-out plots get a thick border.

Left-clicking a plot opens a distribution view (in this tab, still matplotlib — small and not on
the hot path); right-click offers "Open in polyscope". A select mode toggles left-click into
add/remove for a combined 3-D view of several plots.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ml.plot_geometry import build_plot_geoms
from ml.plot_layout import (
    DEFAULT_BASE,
    DEFAULT_COLS,
    DEFAULT_MAJOR,
    DEFAULT_ROWS,
    DEFAULT_SNAKE,
    DEFAULT_START,
    MAJOR_AXES,
    START_CORNERS,
    build_layout,
)
from ..export import attach_export_menu, export_preview
from ..widgets import style_button
from .colormaps import (
    COLORMAP_NAMES,
    full_range_norm,
    get_cmap,
    gradient_stops,
    qcolor_for,
)
from .field_canvas import FieldCanvas, PlotPatch, box_angle_deg
from .grid import cell_origin, make_button

# Special (non-feature) fields the user can display / colour by.
_PLOT_FIELD = "Plot number"
_ERROR_FIELD = "Model error (%)"
_NONE_FIELD = "(none)"
# Per-point colour modes used only when sending clouds to the 3-D viewer (not for the map fill).
_RGB_FIELD = "Original RGB"
_HEIGHT_FIELD = "Relative height"
# A flat single colour for every plot/cloud, picked from a colour dialog (no field, no colour map).
_CUSTOM_FIELD = "Custom single colour"
# The colour a freshly chosen "Custom single colour" starts at (a neutral mid blue).
_DEFAULT_CUSTOM = "#1f77b4"

# Border / role-indication modes. The thick black border marks held-out plots and is the default
# (the old "T/V marker" lettering read poorly on the map, so it was dropped).
_ROLE_BORDER = "thick border on held-out"
_ROLE_BLANK = "blank"
# Default held-out border thickness (px); matches the historical hardcoded pen width.
_DEFAULT_BORDER_WIDTH = 2.4

# Per-plot "normalize" display modes (the dropdown next to each display line).
_NORM_RAW = "raw"
_NORM_Z = "z-score"
_NORM_PCT = "percentile"
_NORM_MODES = [_NORM_RAW, _NORM_Z, _NORM_PCT]

# Distribution view styles (left-click panel).
_DIST_NORMAL = "normal curve"
_DIST_DENSITY = "actual density"
_DIST_BOX = "box plot"

# The two ways to lay out the map. "Field map" draws each plot's real footprint at its true
# position (always works, no layout settings); "Grid" draws uniform squares from a chosen layout.
_VIEW_FIELD = "Field map (true positions)"
_VIEW_GRID = "Grid"


class GroupedFieldButton(QToolButton):
    """A field picker that folds out into groups (Plot number / Target / Features ▶ / Others ▶).

    Replaces a flat combo whose long list mixed the plot number, the target feature, the ordinary
    features and the per-point colour modes. The button shows the current choice; clicking pops a
    :class:`QMenu` with the singletons inline and the longer groups as submenus, so the user reads
    the *kind* of each field at a glance. Emits :attr:`changed` whenever the selection changes.
    """

    changed = Signal(str)

    def __init__(self, allow_none: bool = False) -> None:
        super().__init__()
        self.setPopupMode(QToolButton.InstantPopup)
        self.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        # Fill the width of its cell and never collapse: an empty picker (e.g. the distribution
        # field before a model is loaded) must still be wide enough to see and click, not a 24px
        # sliver. A QSizePolicy of Expanding + a minimum width keeps it a usable target.
        from PySide6.QtWidgets import QSizePolicy

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumWidth(90)
        self._menu = QMenu(self)
        self.setMenu(self._menu)
        self._allow_none = allow_none
        self._value = ""
        self._items: list[str] = []
        self._placeholder = "(none)" if allow_none else "(none available)"
        self.setText(self._placeholder)

    def set_groups(self, singletons: list[str], groups: list[tuple[str, list[str]]],
                   default: str | None = None) -> None:
        """Rebuild the menu. ``singletons`` are top-level entries; ``groups`` are (title, items)
        submenus. ``default`` (or the first available entry) becomes the current value."""
        self._menu.clear()
        self._items = []
        if self._allow_none:
            self._add_action(self._menu, _NONE_FIELD)
            self._menu.addSeparator()
        for name in singletons:
            self._add_action(self._menu, name)
        for title, items in groups:
            items = [i for i in items if i]
            if not items:
                continue
            sub = self._menu.addMenu(title)
            for name in items:
                self._add_action(sub, name)
        # Choose the initial value: the requested default if present, else the first real entry.
        want = default if (default and default in self._items) else None
        if want is None:
            want = self._items[0] if self._items else (_NONE_FIELD if self._allow_none else "")
        self.set_value(want, emit=False)

    def _add_action(self, menu: QMenu, name: str) -> None:
        self._items.append(name)
        act = menu.addAction(name)
        act.triggered.connect(lambda _=False, n=name: self.set_value(n, emit=True))

    def set_value(self, name: str, *, emit: bool) -> None:
        if name == self._value:
            self.setText(name or self._placeholder)
            return
        self._value = name
        self.setText(name or self._placeholder)
        if emit:
            self.changed.emit(name)

    def value(self) -> str:
        return self._value

    def has(self, name: str) -> bool:
        return name in self._items or (self._allow_none and name == _NONE_FIELD)


class _DistributionDialog(QDialog):
    """A large, resizable pop-up of one plot's distribution (right-click "View distribution").

    Holds its own matplotlib canvas and a style selector but delegates the actual drawing to the
    owning :class:`MapView` so the chart matches the side panel exactly (same field, same stats).
    Several can be open at once, like the polyscope windows.
    """

    def __init__(self, owner: "MapView", plot: int) -> None:
        super().__init__(owner)
        self._owner = owner
        self._plot = plot
        self.setWindowTitle(f"Distribution — plot({plot})")
        self.setMinimumSize(560, 420)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        v = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Feature:"))
        # Same fold-out grouped picker as the side panel, seeded to its current field.
        self.field_button = GroupedFieldButton()
        singletons, groups = owner._distribution_groups()
        self.field_button.set_groups(singletons, groups, default=owner.dist_field.value())
        self.field_button.changed.connect(self._redraw)
        controls.addWidget(self.field_button, 1)
        controls.addWidget(QLabel("Style:"))
        self.style_combo = QComboBox()
        self.style_combo.addItems([_DIST_NORMAL, _DIST_DENSITY, _DIST_BOX])
        self.style_combo.setCurrentText(owner.dist_style.currentText())
        self.style_combo.currentIndexChanged.connect(self._redraw)
        controls.addWidget(self.style_combo)
        v.addLayout(controls)

        self._figure = Figure(figsize=(6.0, 4.0), layout="constrained")
        self._canvas = FigureCanvas(self._figure)
        self._ax = self._figure.add_subplot(111)
        v.addWidget(self._canvas, 1)
        attach_export_menu(self._canvas, self)
        self._redraw()

    def draw_into(self, ax) -> None:
        """Export hook: re-render this pop-up's distribution into a standardised export figure."""
        if not self._owner._render_distribution(
            ax, self._plot, self.field_button.value(),
            self.style_combo.currentText(), big=True,
        ):
            ax.clear()
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.5, 0.5, "Not enough data for a distribution.", ha="center", va="center",
                    color="gray", transform=ax.transAxes)

    def export_title(self) -> str:
        return f"distribution_plot({self._plot})"

    def _redraw(self) -> None:
        ok = self._owner._render_distribution(
            self._ax, self._plot, self.field_button.value(),
            self.style_combo.currentText(), big=True)
        if not ok:
            self._ax.clear()
            self._ax.text(0.5, 0.5, "Not enough data for a distribution.", ha="center",
                          va="center", color="gray", transform=self._ax.transAxes)
            self._ax.set_xticks([]); self._ax.set_yticks([])
        self._canvas.draw_idle()


class MapView(QWidget):
    """The selectable field-map sub-tab."""

    def __init__(self, viewer) -> None:
        super().__init__()
        self._viewer = viewer
        self._project = None         # common.project.Project, for layout persistence
        self._predictions = None     # compute_plot_predictions() DataFrame
        self._features = None        # dataset.frame (features + target), indexed by file name
        self._file_map: dict[tuple[int, int], object] = {}
        self._map_index = 0
        self._max_aug = 0
        self._fields: list[str] = []  # selectable field names, in display order
        self._target_column: str | None = None  # the dataset's target feature name, if known
        self._dist_plot: int | None = None  # plot currently shown in the distribution view
        self._selected: set[int] = set()    # plots selected for the combined 3-D view
        self._geoms: dict[int, object] = {}  # plot -> PlotGeom (originals), for the field view
        # Remembered custom single colours for each view's "Custom single colour" mode.
        self._custom_2d = QColor(_DEFAULT_CUSTOM)
        self._custom_3d = QColor(_DEFAULT_CUSTOM)

        root = QHBoxLayout(self)
        # Only the (tall) settings column scrolls. Letting the whole sub-tab scroll instead made
        # the map itself taller than the window, so the field could not be seen in one go — now the
        # canvas always sizes to whatever height the tab has.
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QFrame.NoFrame)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        controls_scroll.setWidget(self._build_controls())
        controls_scroll.setMaximumWidth(420)
        root.addWidget(controls_scroll, 0)
        root.addWidget(self._build_canvas(), 1)  # the map takes all remaining width

    # ------------------------------------------------------------------ #
    # Build                                                              #
    # ------------------------------------------------------------------ #
    def _build_controls(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(360)  # the field rows are wide; give the settings column room
        layout = QVBoxLayout(panel)

        nav = QHBoxLayout()
        nav.addWidget(QLabel("Map:"))
        self.map_combo = QComboBox()
        self.map_combo.setToolTip("Which map level to show: the originals or an augmented copy.")
        self.map_combo.currentIndexChanged.connect(self._on_map_chosen)
        nav.addWidget(self.map_combo, 1)
        layout.addLayout(nav)

        layout.addWidget(self._build_view_2d_box())
        layout.addWidget(self._build_view_3d_box())

        layout.addStretch(1)
        # Field map is the default view, so the grid-layout controls start hidden.
        self._layout_box.setVisible(False)
        return panel

    # -- the two view boxes ---------------------------------------------- #
    def _build_view_2d_box(self) -> QWidget:
        """The "2D view" box: how the field map looks, plus its colour, distribution and chart.

        Everything that draws the flat map lives here: the view-mode picker and its layout, the
        held-out indication, the per-plot display lines, the 2D colour-by, the distribution
        selector and the distribution chart itself.
        """
        box, layout = self._titled_box("2D view")

        # View mode + (for the grid) its layout. Field map is the default — it draws each plot's
        # true footprint at its real position and needs no layout settings.
        view_row = QHBoxLayout()
        view_row.addWidget(QLabel("View:"))
        self.view_combo = QComboBox()
        self.view_combo.addItems([_VIEW_FIELD, _VIEW_GRID])
        self.view_combo.setToolTip(
            "Field map: each plot drawn as its real footprint at its true position (pan/zoom). "
            "Grid: uniform squares laid out from the dimensions and fill order below."
        )
        self.view_combo.currentIndexChanged.connect(self._on_view_changed)
        view_row.addWidget(self.view_combo, 1)
        layout.addLayout(view_row)
        layout.addWidget(self._build_field_box())
        layout.addWidget(self._build_layout_box())

        # Held-out indication. The thick black border is the only mode kept (the T/V lettering was
        # not useful on the map); the user can switch it off with "blank". A compact thickness spinbox
        # sits right of the dropdown and only shows when the border mode is active.
        self._header(layout, "Training / held-out")
        role_row = QHBoxLayout()
        self.role_combo = QComboBox()
        self.role_combo.addItems([_ROLE_BORDER, _ROLE_BLANK])
        self.role_combo.setToolTip("Mark held-out plots with a thick black border, or leave them plain.")
        self.role_combo.currentIndexChanged.connect(self._on_role_changed)
        role_row.addWidget(self.role_combo, 1)
        self.border_width_spin = QDoubleSpinBox()
        self.border_width_spin.setRange(0.5, 12.0)
        self.border_width_spin.setSingleStep(0.5)
        self.border_width_spin.setDecimals(1)
        self.border_width_spin.setValue(_DEFAULT_BORDER_WIDTH)
        self.border_width_spin.setSuffix(" px")
        self.border_width_spin.setToolTip("Held-out border thickness (pixels).")
        self.border_width_spin.valueChanged.connect(self._redraw)
        role_row.addWidget(self.border_width_spin)
        layout.addLayout(role_row)
        self.border_width_spin.setVisible(self.role_combo.currentText() == _ROLE_BORDER)

        # Three display-line rows (field + normalize + font size + bold). Only the first line is
        # enabled by default (the plot number); both the field map and the grid stack up to three.
        # Feature values load automatically when the map is shown, so there is no loader button.
        self._header(layout, "Display in each plot (up to 3 lines)")
        self._disp_box = QWidget()
        self._disp_box.setToolTip(
            "Predictions (and model error) come from the saved model. For deep models they use the "
            "exact point sampling the model was trained and scored with, so they match its reported "
            "metrics rather than a fresh resampling of each cloud."
        )
        disp_layout = QVBoxLayout(self._disp_box)
        disp_layout.setContentsMargins(0, 0, 0, 0)
        self._lines = [self._build_line_row(i) for i in range(3)]
        for row in self._lines:
            disp_layout.addWidget(row["widget"])
        layout.addWidget(self._disp_box)

        # Colour source + colour map for the flat map (stretched over the field's full data range),
        # side by side. Plot number is excluded (its grading is meaningless); a custom single colour
        # paints every plot one flat colour.
        self._header(layout, "Colour by")
        colour_row = QHBoxLayout()
        self.colour_button_2d = GroupedFieldButton()
        self.colour_button_2d.changed.connect(self._on_colour_2d_changed)
        self.cmap_combo_2d = QComboBox()
        self.cmap_combo_2d.addItems(COLORMAP_NAMES)
        self.cmap_combo_2d.currentIndexChanged.connect(self._redraw)
        colour_row.addWidget(self.colour_button_2d, 1)
        colour_row.addWidget(self.cmap_combo_2d, 1)
        layout.addLayout(colour_row)

        # Distribution feature selector + style (used by the left-click view), side by side. Same
        # fold-out grouped picker the colour-by box uses.
        self._header(layout, "Distribution")
        dist_row = QHBoxLayout()
        self.dist_field = GroupedFieldButton()
        self.dist_field.changed.connect(self._redraw_distribution)
        self.dist_style = QComboBox()
        self.dist_style.addItems([_DIST_NORMAL, _DIST_DENSITY, _DIST_BOX])
        self.dist_style.currentIndexChanged.connect(self._redraw_distribution)
        dist_row.addWidget(self.dist_field, 1)
        dist_row.addWidget(self.dist_style, 1)
        layout.addLayout(dist_row)
        # The chart itself sits at the bottom of the 2D box, so the map canvas on the right keeps
        # the full height/width instead of losing a strip to a distribution below it.
        layout.addWidget(self._build_distribution())
        return box

    def _build_view_3d_box(self) -> QWidget:
        """The "3D view" box: plot selection, the 3-D colour-by, and the feature-overlay toggle."""
        box, layout = self._titled_box("3D view")

        # The "Select plots" toggle changes what a left-click does: when on, clicks add/remove plots
        # (marked with a black corner triangle) for a combined 3-D view; when off, a left-click shows
        # the distribution (the original behaviour).
        self.select_toggle = QPushButton("Select plots")
        self.select_toggle.setCheckable(True)
        self.select_toggle.setToolTip(
            "When on, left-click toggles a plot in/out of the selection (marked by a black corner "
            "triangle). When off, left-click shows the distribution. Double-click always opens one "
            "plot; right-click opens all selected plots in polyscope."
        )
        self.select_toggle.toggled.connect(self._on_select_toggled)
        layout.addWidget(self.select_toggle)
        sel_row = QHBoxLayout()
        sel_row.addWidget(make_button("Select all", self._select_all))
        sel_row.addWidget(style_button(make_button("Clear", self._clear_selection), "clear"))
        layout.addLayout(sel_row)
        self.view_button = make_button("3-D view selected →", self._open_selected)
        style_button(self.view_button, "primary")
        layout.addWidget(self.view_button)
        self.count_label = QLabel("0 selected.")
        self.count_label.setStyleSheet("color: gray;")
        layout.addWidget(self.count_label)

        # Colour source + colour map for the 3-D clouds, side by side and independent of the 2D map.
        # Beyond the scalar fields it offers the per-point modes (Original RGB / Relative height) and
        # a custom single colour.
        self._header(layout, "Colour by")
        colour_row = QHBoxLayout()
        self.colour_button_3d = GroupedFieldButton()
        self.colour_button_3d.changed.connect(self._on_colour_3d_changed)
        self.cmap_combo_3d = QComboBox()
        self.cmap_combo_3d.addItems(COLORMAP_NAMES)
        self.cmap_combo_3d.currentIndexChanged.connect(lambda: self._push_to_viewer())
        colour_row.addWidget(self.colour_button_3d, 1)
        colour_row.addWidget(self.cmap_combo_3d, 1)
        layout.addLayout(colour_row)

        # One toggle for the feature-geometry overlays (height-percentile planes, sigma_z plane,
        # bounding box / convex hull, above-mean points). It governs both the combined 3-D view of
        # the selected plots and a single plot opened via double-click / right-click.
        self.feature_check = QCheckBox("Feature visualisations")
        self.feature_check.setToolTip(
            "Draw the height-percentile planes, sigma_z plane, bounding box / convex hull and "
            "above-mean points alongside each cloud sent to the polyscope window."
        )
        self.feature_check.toggled.connect(lambda: self._push_to_viewer())
        layout.addWidget(self.feature_check)
        return box

    @staticmethod
    def _titled_box(title: str) -> tuple[QFrame, QVBoxLayout]:
        """A light bordered frame with a bold title header — the section container for a view box."""
        box = QFrame()
        box.setFrameShape(QFrame.StyledPanel)
        v = QVBoxLayout(box)
        header = QLabel(title)
        header.setStyleSheet("font-weight: bold; font-size: 13px;")
        v.addWidget(header)
        return box, v

    @staticmethod
    def _header(layout: QVBoxLayout, text: str, *, top_gap: int = 8) -> None:
        """A concise bold cluster sub-header (matches the classical-ML tab's grouping style)."""
        if top_gap:
            layout.addSpacing(top_gap)
        label = QLabel(text)
        label.setStyleSheet("font-weight: bold;")
        layout.addWidget(label)

    def _build_field_box(self) -> QWidget:
        """Field-view-only control: the whole-field rotation slider.

        Both views now share the three display-line controls; the only thing unique to the field
        map is rotating the true-position field, so that slider is all this box holds. It is hidden
        (and the rotation reset) in grid view, where uniform squares have no meaningful rotation.
        """
        box = QGroupBox("Field view")
        self._field_box = box
        v = QVBoxLayout(box)

        rot = QHBoxLayout()
        rot.addWidget(QLabel("Rotate:"))
        self.rotate_slider = QSlider(Qt.Horizontal)
        self.rotate_slider.setRange(0, 359)
        self.rotate_slider.setValue(0)
        self.rotate_slider.setToolTip("Rotate the whole field view (degrees).")
        self.rotate_slider.valueChanged.connect(self._on_rotate)
        rot.addWidget(self.rotate_slider, 1)
        self.rotate_label = QLabel("0°")
        self.rotate_label.setFixedWidth(34)
        rot.addWidget(self.rotate_label)
        rot.addWidget(make_button("Reset", self._reset_field_view))
        v.addLayout(rot)
        return box

    def _build_layout_box(self) -> QWidget:
        """The grid view's layout controls: dimensions + fill order.

        Shown only in Grid view. Rows / Cols / Base are auto-derived from the plot set on every
        refresh (a square-ish grid that always fits all plots), but stay editable — a manual override
        is persisted to the project and restored on reopen. The fill-order controls map directly onto
        ``build_layout``.
        """
        box = QGroupBox("Grid layout")
        self._layout_box = box
        v = QVBoxLayout(box)

        dims = QHBoxLayout()
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(1, 100)
        self.rows_spin.setValue(DEFAULT_ROWS)
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(1, 100)
        self.cols_spin.setValue(DEFAULT_COLS)
        self.base_spin = QSpinBox()
        self.base_spin.setRange(0, 1_000_000)
        self.base_spin.setValue(DEFAULT_BASE)
        for label, w in (("Rows", self.rows_spin), ("Cols", self.cols_spin), ("Base", self.base_spin)):
            dims.addWidget(QLabel(label))
            dims.addWidget(w)
        v.addLayout(dims)

        self.start_combo = QComboBox()
        self.start_combo.addItems(START_CORNERS)
        self.start_combo.setCurrentText(DEFAULT_START)
        self.start_combo.setToolTip("Which corner the lowest plot number sits in.")
        self.major_combo = QComboBox()
        self.major_combo.addItems(MAJOR_AXES)
        self.major_combo.setCurrentText(DEFAULT_MAJOR)
        self.major_combo.setToolTip("Whether numbers run along a row or down a column first.")
        self.snake_check = QCheckBox("Snake (alternate lines reverse)")
        self.snake_check.setChecked(DEFAULT_SNAKE)
        for w in (self.start_combo, self.major_combo, self.snake_check):
            v.addWidget(w)

        # Any change rebuilds the layout, redraws, and persists.
        for w in (self.rows_spin, self.cols_spin, self.base_spin):
            w.valueChanged.connect(self._on_layout_changed)
        self.start_combo.currentIndexChanged.connect(self._on_layout_changed)
        self.major_combo.currentIndexChanged.connect(self._on_layout_changed)
        self.snake_check.toggled.connect(self._on_layout_changed)
        return box

    def _build_line_row(self, index: int) -> dict:
        """One display line: a grouped field picker, a normalize mode, an optional font size, bold.

        Line 0 is enabled by default (the plot number); lines 1–2 default to ``(none)``. The field
        picker can pick ``(none)`` to switch a line off. ``font`` is a manual override of the
        auto-chosen size: when left at 0 (its special "auto" floor) the view sizes the text itself.
        """
        widget = QWidget()
        h = QHBoxLayout(widget)
        h.setContentsMargins(0, 0, 0, 0)
        field = GroupedFieldButton(allow_none=True)
        field.changed.connect(self._redraw)
        norm_mode = QComboBox()
        norm_mode.addItems(_NORM_MODES)
        norm_mode.setToolTip(
            "raw value, its z-score across this map, or its percentile rank (0–100)."
        )
        norm_mode.currentIndexChanged.connect(self._redraw)
        font = QSpinBox()
        font.setRange(0, 40)
        font.setValue(0)            # 0 = auto-size; any other value is a manual point size
        font.setSpecialValueText("auto")
        font.setToolTip("Font size for this line (auto fits it to the plot).")
        font.valueChanged.connect(self._redraw)
        bold = QCheckBox("B")
        bold.setToolTip("Bold this line.")
        bold.toggled.connect(self._redraw)
        h.addWidget(field, 1)
        h.addWidget(norm_mode)
        h.addWidget(font)
        h.addWidget(bold)
        return {"widget": widget, "field": field, "norm_mode": norm_mode, "font": font, "bold": bold}

    def _build_canvas(self) -> QWidget:
        wrap = QWidget()
        layout = QHBoxLayout(wrap)
        layout.setContentsMargins(0, 0, 0, 0)
        # The map is a scene-graph canvas (hardware-friendly pan/zoom/rotate). The colour legend,
        # title and Home button are overlay chrome the canvas paints *over* the rendered field, so
        # the canvas takes the full width and nothing sits in the window margins.
        self.canvas = FieldCanvas()
        self.canvas.plot_clicked.connect(self._on_plot_clicked)
        self.colorbar = self.canvas.colorbar  # the legend now lives over the canvas
        layout.addWidget(self.canvas, 1)
        return wrap

    def _build_distribution(self) -> QWidget:
        """The left-click distribution view, shown in the left column so the map gets full width."""
        # constrained_layout reflows the (wrapped, two-line) title so it never clips in this short
        # panel, where tight_layout used to cut the stats line off.
        self.dist_figure = Figure(figsize=(3.0, 2.2), layout="constrained")
        self.dist_canvas = FigureCanvas(self.dist_figure)
        self.dist_canvas.setMinimumHeight(190)
        self.dist_ax = self.dist_figure.add_subplot(111)
        self._draw_empty_distribution("Left-click a plot to see its z-score.")
        attach_export_menu(self.dist_canvas, _DistributionExport(self))
        return self.dist_canvas

    # ------------------------------------------------------------------ #
    # Data in                                                            #
    # ------------------------------------------------------------------ #
    def refresh(self, predictions, features, file_map, project=None, target_column=None) -> None:
        """Receive the latest predictions + feature table from the host page and redraw.

        ``features`` is the dataset frame; its columns (which include the target) become the
        selectable display/colour fields. ``target_column`` (when given) names which of those
        columns is the prediction target, so the field picker can list it on its own. ``project``
        (when given) supplies and stores the persisted grid layout.
        """
        self._project = project
        self._predictions = predictions
        self._features = features
        self._target_column = target_column
        self._file_map = file_map or {}
        # The highest augmented level to offer in the map dropdown: from the predictions when a model
        # is loaded, else (inspection-only) from the aug levels of the clouds on disk.
        if predictions is not None and len(predictions):
            self._max_aug = int(predictions["aug"].max())
        else:
            self._max_aug = max((aug for _plot, aug in self._file_map), default=0)
        self._map_index = 0
        self._selected.clear()
        # Read the original plots' footprints once; the field view reuses them at every map level.
        self._geoms = build_plot_geoms(self._file_map, aug_index=0) if self._file_map else {}
        self._load_layout_settings()
        self._populate_map_combo()
        self._rebuild_field_lists()
        self._redraw()
        self._draw_empty_distribution("Left-click a plot to see its z-score.")

    # ------------------------------------------------------------------ #
    # Layout (grid view) settings                                        #
    # ------------------------------------------------------------------ #
    def _layout_settings(self) -> dict:
        """The current grid-layout controls as a ``build_layout`` kwargs dict."""
        return {
            "rows": self.rows_spin.value(),
            "cols": self.cols_spin.value(),
            "base": self.base_spin.value(),
            "start": self.start_combo.currentText(),
            "major": self.major_combo.currentText(),
            "snake": self.snake_check.isChecked(),
        }

    def _load_layout_settings(self) -> None:
        """Seed the layout controls from the plot set, letting a saved override win, without saving.

        Rows / Cols / Base are auto-derived so every plot is always visible in a square-ish grid
        (:meth:`_auto_grid`); a value the user previously changed (stored in the project's layout)
        takes precedence over the auto value, so a hand-tuned grid survives a reopen.
        """
        saved = dict(getattr(self._project, "layout", {}) or {})
        auto_rows, auto_cols, auto_base = self._auto_grid()
        widgets = (
            (self.rows_spin, "rows", auto_rows),
            (self.cols_spin, "cols", auto_cols),
            (self.base_spin, "base", auto_base),
        )
        for w in (self.rows_spin, self.cols_spin, self.base_spin,
                  self.start_combo, self.major_combo, self.snake_check):
            w.blockSignals(True)
        for spin, key, auto in widgets:
            spin.setValue(int(saved.get(key, auto)))
        self.start_combo.setCurrentText(str(saved.get("start", DEFAULT_START)))
        self.major_combo.setCurrentText(str(saved.get("major", DEFAULT_MAJOR)))
        self.snake_check.setChecked(bool(saved.get("snake", DEFAULT_SNAKE)))
        for w in (self.rows_spin, self.cols_spin, self.base_spin,
                  self.start_combo, self.major_combo, self.snake_check):
            w.blockSignals(False)

    def _auto_grid(self) -> tuple[int, int, int]:
        """A square-ish ``(rows, cols, base)`` that fits every plot on the current map.

        ``base`` is the lowest plot number present. For ``n`` plots the grid is kept as square as
        possible: ``rows = ceil(sqrt(n))`` and ``cols = ceil(n / rows)`` (so ``rows * cols >= n``,
        every plot has a cell). Falls back to the historical default when no plots are known yet.
        """
        plots = self._all_plots()
        if not plots:
            return DEFAULT_ROWS, DEFAULT_COLS, DEFAULT_BASE
        import math

        n = len(plots)
        rows = max(1, math.ceil(math.sqrt(n)))
        cols = max(1, math.ceil(n / rows))
        return rows, cols, min(plots)

    def _all_plots(self) -> list[int]:
        """Every distinct plot number known for the current data (predictions or clouds on disk)."""
        if self._predictions is not None and len(self._predictions):
            return sorted({int(p) for p in self._predictions["plot"].unique()})
        return sorted({plot for plot, _aug in self._file_map})

    def _current_layout(self) -> dict[int, tuple[int, int]]:
        """The active ``plot -> (row, col)`` mapping from the current controls."""
        return build_layout(**self._layout_settings())

    def _on_layout_changed(self) -> None:
        if self._project is not None:
            self._project.set_layout(self._layout_settings())
        self._redraw()

    def _on_view_changed(self) -> None:
        is_grid = self.view_combo.currentText() == _VIEW_GRID
        self._layout_box.setVisible(is_grid)
        self._field_box.setVisible(not is_grid)  # rotation slider is field-map only
        # The grid view's uniform squares have no meaningful rotation, and it must NOT inherit the
        # field map's: force the canvas back to 0° in grid view, and restore the slider's value when
        # returning to the field map.
        self.canvas.set_rotation(0.0 if is_grid else float(self.rotate_slider.value()))
        self._redraw()

    def _on_rotate(self, value: int) -> None:
        self.rotate_label.setText(f"{value}°")
        self.canvas.set_rotation(float(value))

    def _reset_field_view(self) -> None:
        self.rotate_slider.setValue(0)
        self.canvas.reset_view()

    def _on_role_changed(self) -> None:
        # The thickness spinbox only makes sense when the held-out border is drawn; hide it otherwise.
        self.border_width_spin.setVisible(self.role_combo.currentText() == _ROLE_BORDER)
        self._redraw()

    def _populate_map_combo(self) -> None:
        """Fill the map-level dropdown: the originals plus one entry per augmented copy."""
        self.map_combo.blockSignals(True)
        self.map_combo.clear()
        for index in range(self._max_aug + 1):
            self.map_combo.addItem(self._map_name(index), index)
        self.map_combo.setCurrentIndex(0)
        self.map_combo.blockSignals(False)

    @staticmethod
    def _map_name(index: int) -> str:
        return "Originals" if index == 0 else f"Augmented copy aug({index})"

    def _rebuild_field_lists(self) -> None:
        """Populate the grouped field/colour pickers and the distribution combo from the columns.

        Fields are grouped so the picker distinguishes the plot number, the target feature, the
        ordinary features and (colour only) the extra colour modes:

        * **display lines** list ``Plot number``, ``Model error (%)``, the target and a **Features**
          submenu (line 0 defaults to the plot number);
        * the **2D colour** picker drops the plot number (its grading is meaningless) and offers a
          custom single colour under **Others**;
        * the **3D colour** picker adds the per-point modes ``Original RGB`` / ``Relative height`` to
          that **Others** submenu (those are per-point and make no sense as a flat 2-D fill).
        """
        feature_cols = list(self._features.columns) if self._features is not None else []
        target = self._target_column if (self._target_column in feature_cols) else None
        plain_features = [c for c in feature_cols if c != target]
        # Model error needs a model: in inspection-only mode (no predictions) it is dropped from the
        # pickers and the plot number becomes the default *display* field instead.
        has_predictions = self._predictions is not None and len(self._predictions) > 0
        error_singleton = [_ERROR_FIELD] if has_predictions else []
        self._fields = [_PLOT_FIELD, *error_singleton, *feature_cols]

        # Display lines keep the plot number; the colour pickers drop it.
        disp_singletons = [_PLOT_FIELD, *error_singleton] + ([target] if target else [])
        colour_singletons = [*error_singleton] + ([target] if target else [])
        groups = [("Features ▸", plain_features)]

        # Display lines: line 0 defaults to the plot number, the rest to (none).
        for i, row in enumerate(self._lines):
            row["field"].set_groups(disp_singletons, groups,
                                    default=_PLOT_FIELD if i == 0 else _NONE_FIELD)
        # Default colour/distribution field: model error when a model is loaded, else the target (so
        # the inspection map is immediately informative), else the first feature; the picker falls
        # back to its first entry (or Custom) when none of those exist.
        scalar_default = _ERROR_FIELD if has_predictions else (target or (plain_features[0] if plain_features else _CUSTOM_FIELD))
        # 2D colour by: scalar fields + a custom single colour. No plot number, no per-point modes.
        self.colour_button_2d.set_groups(
            colour_singletons, [*groups, ("Others ▸", [_CUSTOM_FIELD])], default=scalar_default)
        # 3D colour by: the same, plus the per-point cloud modes (RGB / relative height).
        self.colour_button_3d.set_groups(
            colour_singletons,
            [*groups, ("Others ▸", [_RGB_FIELD, _HEIGHT_FIELD, _CUSTOM_FIELD])],
            default=scalar_default)
        # Distribution: the same fold-out grouping, minus the plot number (its spread is meaningless)
        # and the per-point 3-D colour modes (not scalar fields). Model error needs a model; the
        # target/features make the distribution meaningful before training too.
        dist_singletons, dist_groups = self._distribution_groups()
        self.dist_field.set_groups(dist_singletons, dist_groups, default=scalar_default)
        self._sync_cmap_enabled()

    def _distribution_groups(self) -> tuple[list[str], list[tuple[str, list[str]]]]:
        """The (singletons, groups) for the distribution field picker — shared by the side panel
        button and the pop-up's, so both fold out identically."""
        feature_cols = list(self._features.columns) if self._features is not None else []
        target = self._target_column if (self._target_column in feature_cols) else None
        plain_features = [c for c in feature_cols if c != target]
        # Model error needs a model; drop it from the distribution picker in inspection-only mode.
        has_predictions = self._predictions is not None and len(self._predictions) > 0
        singletons = ([_ERROR_FIELD] if has_predictions else []) + ([target] if target else [])
        return singletons, [("Features ▸", plain_features)]

    # ------------------------------------------------------------------ #
    # Per-map value lookups                                              #
    # ------------------------------------------------------------------ #
    def _map_rows(self) -> dict[int, dict]:
        """plot -> {'error_pct', 'role', feature columns...} for the current map level.

        With no predictions loaded (inspection-only mode, no model trained) the rows are
        synthesised straight from the cloud files on disk: every plot present at the current map
        level, with no model-derived ``error_pct`` / ``role``. This keeps the field map drawable —
        plot footprints, plot-number labels, and polyscope inspection — before any model exists.
        """
        if self._predictions is None or not len(self._predictions):
            return self._inspection_rows()
        sub = self._predictions[self._predictions["aug"] == self._map_index]
        rows: dict[int, dict] = {}
        for r in sub.itertuples():
            plot = int(r.plot)
            rec = {"error_pct": float(r.error_pct), "role": str(r.role)}
            file = self._file_map.get((plot, self._map_index))
            if file is not None and self._features is not None and file.name in self._features.index:
                frow = self._features.loc[file.name]
                for col in self._features.columns:
                    rec[col] = _to_float(frow[col])
            rows[plot] = rec
        return rows

    def _inspection_rows(self) -> dict[int, dict]:
        """plot -> record for every cloud on disk at the current map level (no model needed).

        Each record has a NaN ``error_pct`` and no role (no model has scored these plots). When the
        project's features workbook is loaded its columns are merged in (joined on the cloud file
        name, the same key the prediction path uses), so the target and every feature are valid
        colour-by / distribution / display fields **before any model is trained** — only model error
        and the held-out role need a model and stay absent here.
        """
        rows: dict[int, dict] = {}
        for (plot, aug), file in self._file_map.items():
            if aug != self._map_index:
                continue
            rec = {"error_pct": float("nan"), "role": ""}
            if (file is not None and self._features is not None
                    and getattr(file, "name", None) in self._features.index):
                frow = self._features.loc[file.name]
                for col in self._features.columns:
                    rec[col] = _to_float(frow[col])
            rows[int(plot)] = rec
        return rows

    def _raw_value(self, plot: int, rec: dict, field: str) -> float:
        if field == _PLOT_FIELD:
            return float(plot)
        if field == _ERROR_FIELD:
            return rec.get("error_pct", float("nan"))
        return rec.get(field, float("nan"))

    def _column_values(self, rows: dict[int, dict], field: str) -> dict[int, float]:
        return {plot: self._raw_value(plot, rec, field) for plot, rec in rows.items()}

    @staticmethod
    def _zscores(values: dict[int, float]) -> dict[int, float]:
        arr = np.array([v for v in values.values()], dtype=float)
        finite = arr[np.isfinite(arr)]
        mean = finite.mean() if finite.size else 0.0
        std = finite.std(ddof=0) if finite.size else 0.0
        return {p: ((v - mean) / std if std > 0 and np.isfinite(v) else float("nan"))
                for p, v in values.items()}

    @staticmethod
    def _percentiles(values: dict[int, float]) -> dict[int, float]:
        """Map each value to its percentile rank (0–100) among the finite values on this map."""
        finite = np.array([v for v in values.values() if np.isfinite(v)], dtype=float)
        if finite.size == 0:
            return {p: float("nan") for p in values}
        return {p: (float((finite < v).mean() * 100.0) if np.isfinite(v) else float("nan"))
                for p, v in values.items()}

    def _transform(self, values: dict[int, float], mode: str) -> dict[int, float]:
        """Apply the per-line display transform (raw / z-score / percentile)."""
        if mode == _NORM_Z:
            return self._zscores(values)
        if mode == _NORM_PCT:
            return self._percentiles(values)
        return values

    # ------------------------------------------------------------------ #
    # Drawing                                                            #
    # ------------------------------------------------------------------ #
    def _on_map_chosen(self) -> None:
        index = self.map_combo.currentData()
        if index is None:
            return
        self._map_index = int(index)
        self._selected.clear()  # selection is per map level
        self._redraw()
        self._push_to_viewer()

    def _redraw(self) -> None:
        # With neither predictions nor any clouds on disk there is nothing to draw. (A non-empty
        # file map with no predictions is the inspection-only mode and draws the footprints below.)
        if self._predictions is None and not self._file_map:
            self.canvas.set_scene([], field_mode=True, title="No predictions yet.",
                                  label_for=lambda p: [])
            self.colorbar.set_scale([], 0.0, 1.0, "")
            return
        model = self._compute_draw_model()
        if self.view_combo.currentText() == _VIEW_FIELD:
            self._draw_field(model)
        else:
            self._draw_grid(model)
        self.count_label.setText(f"{len(self._selected)} selected.")

    def _compute_draw_model(self) -> dict:
        """The shape-independent "what each plot shows": value tables + colour scale + colorbar.

        Both views consume this identical model, so the colour grading is computed in exactly one
        place and cannot drift between them. The field view's single value and the grid view's three
        lines are both built here so the per-plot text is one source of truth.
        """
        rows = self._map_rows()

        # Up to three configured value lines (raw / z-score / percentile), computed once and shared
        # by both views. A line whose field is "(none)" is skipped.
        line_specs = []
        for row in self._lines:
            field = row["field"].value()
            if field == _NONE_FIELD or not field:
                line_specs.append(None)
                continue
            mode = row["norm_mode"].currentText()
            values = self._transform(self._column_values(rows, field), mode)
            line_specs.append({"values": values, "font": row["font"].value(),
                               "bold": row["bold"].isChecked(), "mode": mode, "field": field})

        # Colour table for the flat map. "Custom single colour" paints every plot one flat colour
        # (no field, no scale, blank legend). Otherwise the chosen field drives the fill, stretched
        # over its full finite range and reflected in the legend.
        colour_field = self.colour_button_2d.value()
        if colour_field == _CUSTOM_FIELD:
            self.colorbar.set_scale([], 0.0, 1.0, "")
            return {
                "rows": rows, "line_specs": line_specs, "custom": QColor(self._custom_2d),
                "role_mode": self.role_combo.currentText(),
                "border_width": self.border_width_spin.value(),
            }
        fill_field = colour_field
        colour_values = self._column_values(rows, fill_field)
        cmap, norm = self._colour_scale(colour_values)
        # Update the legend widget (low/high of the finite range + the field's name).
        finite = np.array([v for v in colour_values.values() if np.isfinite(v)], dtype=float)
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        self.colorbar.set_scale(gradient_stops(cmap), lo, hi, fill_field)

        return {
            "rows": rows, "line_specs": line_specs,
            "colour_values": colour_values, "cmap": cmap, "norm": norm, "fill_field": fill_field,
            "role_mode": self.role_combo.currentText(),
            "border_width": self.border_width_spin.value(),
        }

    def _facecolor(self, model: dict, plot: int):
        """The fill QColor for a plot's patch (grey when it has no finite colour value)."""
        custom = model.get("custom")
        if custom is not None:
            return QColor(custom)
        cval = model["colour_values"].get(plot, float("nan"))
        if model["rows"].get(plot) is None:
            return qcolor_for(model["cmap"], model["norm"], float("nan"))
        return qcolor_for(model["cmap"], model["norm"], cval)

    def _patches_common(self, model: dict, polygon_for, angle_for):
        """Build the :class:`PlotPatch` list shared by both views from a geometry source.

        ``polygon_for(plot)`` returns the plot's world polygon (footprint or unit square) or None;
        ``angle_for(plot)`` its label rotation. Held-out borders and the selection markers are
        derived from the per-plot role and the current selection.
        """
        rows, role_mode = model["rows"], model["role_mode"]
        border_width = float(model.get("border_width", _DEFAULT_BORDER_WIDTH))
        patches = []
        for plot in rows:
            poly = polygon_for(plot)
            if poly is None:
                continue
            held = role_mode == _ROLE_BORDER and rows.get(plot, {}).get("role") == "V"
            patches.append(PlotPatch(
                plot=plot, polygon=poly, fill=self._facecolor(model, plot),
                angle_deg=angle_for(plot), held_out=held, selected=plot in self._selected,
                border_width=border_width,
            ))
        return patches

    def _label_for(self, model: dict):
        """A ``plot -> [(text, font_pt, bold), ...]`` builder shared by both views from the lines."""
        def label_for(plot):
            lines = []
            for spec in model["line_specs"]:
                if spec is None:
                    continue
                val = spec["values"].get(plot, float("nan"))
                lines.append((_format_value(val, spec["mode"], spec["field"]),
                              float(spec["font"]), spec["bold"]))
            return lines
        return label_for

    def _map_title(self, model: dict) -> str:
        """The canvas title for the current map level and its 2-D colour source."""
        by = "a custom colour" if model.get("custom") is not None else model["fill_field"]
        return f"{self._map_name(self._map_index)} — coloured by {by}"

    def _draw_field(self, model: dict) -> None:
        """Each plot as its real oriented footprint at its true position (the scene-graph view)."""
        title = self._map_title(model)
        if not self._geoms:
            self.canvas.set_scene([], field_mode=True, title="No plot footprints available.",
                                  label_for=lambda p: [])
            return

        def polygon_for(plot):
            geom = self._geoms.get(plot)
            return geom.hull if (geom is not None and geom.hull is not None) else None

        def angle_for(plot):
            geom = self._geoms.get(plot)
            return box_angle_deg(geom.hull) if (geom is not None and geom.hull is not None) else 0.0

        patches = self._patches_common(model, polygon_for, angle_for)
        self.canvas.set_scene(patches, field_mode=True, title=title, label_for=self._label_for(model))

    def _draw_grid(self, model: dict) -> None:
        """Uniform unit squares laid out by the chosen layout (the schematic view)."""
        title = self._map_title(model)
        grid_layout = self._current_layout()

        def polygon_for(plot):
            cell = grid_layout.get(plot)
            if cell is None:
                return None
            x, y = cell_origin(*cell, grid_layout)
            return np.array([[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1]], dtype=float)

        patches = self._patches_common(model, polygon_for, lambda p: 0.0)
        self.canvas.set_scene(patches, field_mode=False, title=title, label_for=self._label_for(model))

    def _colour_scale(self, values: dict[int, float]):
        """The chosen 2-D colour map, always stretched over the field's full (finite) data range."""
        return get_cmap(self.cmap_combo_2d.currentText()), full_range_norm(values.values())

    # ------------------------------------------------------------------ #
    # Left-click distribution                                            #
    # ------------------------------------------------------------------ #
    def _draw_empty_distribution(self, message: str) -> None:
        self.dist_ax.clear()
        self.dist_ax.set_xticks([])
        self.dist_ax.set_yticks([])
        self.dist_ax.text(0.5, 0.5, message, ha="center", va="center", color="gray",
                          transform=self.dist_ax.transAxes)
        self.dist_canvas.draw_idle()

    def _redraw_distribution(self) -> None:
        if self._dist_plot is not None:
            self._show_distribution(self._dist_plot)

    def _show_distribution(self, plot: int) -> None:
        """Draw the selected feature's distribution across this map into the side panel."""
        self._dist_plot = plot
        field = self.dist_field.value()
        style = self.dist_style.currentText()
        ok = self._render_distribution(self.dist_ax, plot, field, style, big=False)
        if not ok:
            self._draw_empty_distribution("Not enough data for a distribution.")
            return
        self.dist_canvas.draw_idle()

    def _render_distribution(self, ax, plot: int, field: str, style: str, *, big: bool) -> bool:
        """Render one plot's distribution for ``field`` into ``ax``. Returns False if no data.

        Shared by the small side panel (``big=False``) and the large pop-up (``big=True``); the only
        difference is font sizes and how the title is laid out so it fits. Styles: a fitted normal
        curve, the actual density (KDE + rug), or a vertical box plot.
        """
        if not field:
            return False
        rows = self._map_rows()
        values = self._column_values(rows, field)
        arr = np.array([v for v in values.values() if np.isfinite(v)], dtype=float)
        if arr.size < 2:
            return False
        mean, std = float(arr.mean()), float(arr.std(ddof=0))
        val = values.get(plot, float("nan"))
        ax.clear()
        is_box = style == _DIST_BOX

        if is_box:
            self._draw_box(ax, arr, val)
        elif style == _DIST_DENSITY:
            self._draw_density(ax, arr)
            if np.isfinite(val):
                ax.axvline(val, color="#d73027", linewidth=2)
            ax.set_yticks([])
        else:
            if std > 0:
                xs = np.linspace(mean - 4 * std, mean + 4 * std, 200)
                ys = np.exp(-0.5 * ((xs - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
                ax.plot(xs, ys, color="#1f77b4")
                ax.fill_between(xs, ys, color="#1f77b4", alpha=0.1)
            if np.isfinite(val):
                ax.axvline(val, color="#d73027", linewidth=2)
            ax.set_yticks([])

        # Title. The small panel can't fit the full one-liner, so we wrap it onto two lines and use a
        # smaller font; the pop-up has room for a single larger line.
        if np.isfinite(val):
            z = (val - mean) / std if std > 0 else 0.0
            pct = float((arr < val).mean() * 100.0)
            head = f"plot({plot})  {field} = {val:.3g}"
            stats = f"z = {z:+.2f},  pct = {pct:.0f}%,  mean {mean:.3g},  sd {std:.3g}"
            if big:
                ax.set_title(f"{head}    ({stats})", fontsize=13)
            else:
                ax.set_title(f"{head}\n{stats}", fontsize=8)
        return True

    def _draw_box(self, ax, arr: np.ndarray, val: float) -> None:
        """A vertical box plot of the distribution with this plot's value marked as a red dot."""
        ax.boxplot(arr, vert=True, widths=0.5, showfliers=True,
                   medianprops={"color": "#1f77b4"},
                   flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "#999",
                               "markeredgecolor": "none", "alpha": 0.5})
        if np.isfinite(val):
            ax.plot(1, val, marker="o", color="#d73027", markersize=8, zorder=5)
        ax.set_xticks([])

    def _draw_density(self, ax, arr: np.ndarray) -> None:
        """Every sample as a tick on the x-axis, plus a smoothed density outline (KDE)."""
        kde_ys = None
        if np.ptp(arr) > 0:
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(arr)
            xs = np.linspace(arr.min(), arr.max(), 200)
            kde_ys = kde(xs)
            ax.plot(xs, kde_ys, color="#1f77b4", linewidth=1.5)
        # Place the rug just below the baseline so the ticks never overlap the density curve.
        peak = float(kde_ys.max()) if kde_ys is not None and kde_ys.size else 1.0
        ax.plot(arr, np.zeros_like(arr), marker="|", linestyle="none",
                color="#1f77b4", markersize=12, markeredgewidth=1.2, alpha=0.7)
        ax.set_ylim(-0.06 * peak, 1.08 * peak)

    # ------------------------------------------------------------------ #
    # Clicks                                                             #
    # ------------------------------------------------------------------ #
    def _on_plot_clicked(self, plot: int, button, dblclick: bool) -> None:
        """Handle a plot click reported by the canvas (button is a Qt.MouseButton)."""
        if button == Qt.RightButton:        # right-click -> context menu
            self._show_context_menu(plot)
        elif dblclick:                      # double-click -> open this single plot in polyscope
            self._open_in_polyscope(plot)
        elif self.select_toggle.isChecked():  # select-mode left-click -> toggle selection
            self._toggle_select(plot)
        else:                               # left-click -> distribution (the original behaviour)
            self._show_distribution(plot)

    # -- selection ------------------------------------------------------- #
    def _on_select_toggled(self, on: bool) -> None:
        self.select_toggle.setText("Selecting plots — click to add/remove" if on else "Select plots")

    def _toggle_select(self, plot: int) -> None:
        if plot in self._selected:
            self._selected.discard(plot)
        else:
            self._selected.add(plot)
        self._redraw()
        self._push_to_viewer()

    def _select_all(self) -> None:
        self._selected = set(self._map_rows().keys())
        self.select_toggle.setChecked(True)
        self._redraw()
        self._push_to_viewer()

    def _clear_selection(self) -> None:
        self._selected.clear()
        self._redraw()
        self._push_to_viewer()

    def _open_selected(self) -> None:
        """The '3-D view selected' button: open all selected plots together (opening the viewer)."""
        if not self._selected:
            self.count_label.setText("Select at least one plot first.")
            return
        self._push_to_viewer(force=True)

    def _on_colour_2d_changed(self) -> None:
        # Picking "Custom single colour" opens a colour dialog; otherwise the chosen field drives the
        # map fill. Either way the flat map is redrawn (the 3-D clouds are independent now).
        if self.colour_button_2d.value() == _CUSTOM_FIELD:
            self._pick_custom_colour(is_3d=False)
        self._sync_cmap_enabled()
        self._redraw()

    def _on_colour_3d_changed(self) -> None:
        # Same idea for the 3-D clouds: a custom colour opens the dialog; RGB ignores the map.
        # Only the polyscope clouds are affected (live-update any open viewer).
        if self.colour_button_3d.value() == _CUSTOM_FIELD:
            self._pick_custom_colour(is_3d=True)
        self._sync_cmap_enabled()
        self._push_to_viewer()

    def _sync_cmap_enabled(self) -> None:
        """Grey each colour map combo when its field doesn't use a ramp (custom colour, or 3-D RGB)."""
        self.cmap_combo_2d.setEnabled(self.colour_button_2d.value() != _CUSTOM_FIELD)
        self.cmap_combo_3d.setEnabled(
            self.colour_button_3d.value() not in (_CUSTOM_FIELD, _RGB_FIELD))

    def _pick_custom_colour(self, *, is_3d: bool) -> None:
        """Open a colour dialog for the custom single-colour mode, remembering the choice.

        Seeded with the view's current custom colour; if the user cancels, the previous colour is
        kept (the picker stays on "Custom single colour"). Called from the colour-by handlers.
        """
        current = self._custom_3d if is_3d else self._custom_2d
        chosen = QColorDialog.getColor(current, self, "Pick a single colour")
        if chosen.isValid():
            if is_3d:
                self._custom_3d = chosen
            else:
                self._custom_2d = chosen

    def _show_context_menu(self, plot: int) -> None:
        menu = QMenu(self)
        if self.select_toggle.isChecked() and self._selected:
            # In selection mode with a selection, the right-click opens every selected plot.
            n = len(self._selected)
            action = menu.addAction(f"Open {n} selected plot(s) in polyscope")
            if menu.exec(QCursor.pos()) is action:
                self._push_to_viewer(force=True)
            return
        file = self._file_map.get((plot, self._map_index))
        open_action = menu.addAction(f"Open plot({plot}) in polyscope")
        open_action.setEnabled(file is not None and file.exists())
        dist_action = menu.addAction(f"View plot({plot}) distribution")
        menu.addSeparator()
        export_action = menu.addAction("Export map…")
        chosen = menu.exec(QCursor.pos())
        if chosen is open_action:
            self._open_in_polyscope(plot)
        elif chosen is dist_action:
            self._open_distribution_popup(plot)
        elif chosen is export_action:
            export_preview(self, self.canvas)

    def _open_distribution_popup(self, plot: int) -> None:
        """Open this plot's distribution in a large, resizable window (right-click action).

        Mirrors "Open in polyscope": a focused pop-up so the small side-panel chart can be read in
        detail. It reuses the current feature + style selection and updates if those change while
        it's open.
        """
        dlg = _DistributionDialog(self, plot)
        dlg.show()

    def _slope_strata_pct(self) -> float | None:
        """The slope-strata % the project's features were generated with (for the viewer planes)."""
        return getattr(self._project, "slope_strata_pct", None) if self._project is not None else None

    def _height_channel(self) -> str | None:
        """The height channel the project's features used (so the viewer draws from the same one)."""
        return getattr(self._project, "height_channel", None) if self._project is not None else None

    def _open_in_polyscope(self, plot: int) -> None:
        """Show one ``plot``'s cloud in the shared viewer (double-click / menu).

        The feature-geometry overlays are drawn only when the "Polyscope feature visualisations"
        toggle is on, matching the combined 3-D view.
        """
        file = self._file_map.get((plot, self._map_index))
        if file is None or not file.exists():
            return
        features = {"path": str(file)} if self.feature_check.isChecked() else None
        # Honour the 3-D "Colour by"/colour-map settings, exactly as the combined 3-D push does;
        # otherwise a single-plot open would silently show a different colouring from the one the
        # controls describe.
        coloring, cmap_name, solid = self._colouring_for([plot])
        spec = {"path": str(file), "coloring": coloring, "cmap": cmap_name}
        if solid is not None:
            spec["color"] = solid[plot]
        self._viewer.send(
            [spec],
            features=features,
            slope_strata_pct=self._slope_strata_pct(),
            height_channel=self._height_channel(),
        )

    # -- combined 3-D push (ported from the old point-cloud view) -------- #
    def _scalar_value(self, plot: int, rec: dict, field: str) -> float:
        if field == _PLOT_FIELD:
            return float(plot)
        if field == _ERROR_FIELD:
            return rec.get("error_pct", float("nan"))
        return rec.get(field, float("nan"))

    def _colouring_for(self, plots) -> tuple[str, str, dict[int, list] | None]:
        """The viewer colouring for ``plots`` from the 3-D colour controls.

        Returns ``(coloring, cmap_name, solid)``: a per-plot field yields one solid colour per
        plot through the chosen map (stretched over every plot on this map, so a cloud matches its
        map-view square); "Relative height" is a per-point gradient; "Original RGB" uses the file
        colours; "Custom single colour" paints every cloud the same picked colour.
        """
        field = self.colour_button_3d.value()
        cmap_name = self.cmap_combo_3d.currentText()
        solid: dict[int, list] | None = None
        if field == _CUSTOM_FIELD:
            rgb = [self._custom_3d.redF(), self._custom_3d.greenF(), self._custom_3d.blueF()]
            solid = {p: list(rgb) for p in plots}
        elif field not in (_RGB_FIELD, _HEIGHT_FIELD):
            rows = self._map_rows()
            all_vals = {p: self._scalar_value(p, rec, field) for p, rec in rows.items()}
            solid = solid_colours(all_vals, plots, cmap_name)
        coloring = "rgb" if field == _RGB_FIELD else "height" if field == _HEIGHT_FIELD else "error"
        return coloring, cmap_name, solid

    def _push_to_viewer(self, force: bool = False) -> None:
        """Send the current selection to the shared polyscope window (live update).

        Only pushes when the viewer is already open, unless ``force`` (the 3-D view button / the
        right-click "open selected") asks to open it. Colouring follows the **3-D** colour-by (now
        independent of the flat map): a per-plot field colours each cloud one solid colour through
        the chosen map (stretched over every plot on this map); "Relative height" is a per-point
        gradient; "Original RGB" uses the file colours; "Custom single colour" paints every cloud
        the same picked colour.
        """
        if not force and not self._viewer.is_alive():
            return
        want_features = self.feature_check.isChecked()
        coloring, cmap_name, solid = self._colouring_for(self._selected)

        clouds = []
        for plot in sorted(self._selected):
            file = self._file_map.get((plot, self._map_index))
            if file is None or not file.exists():
                continue
            spec = {"path": str(file), "coloring": coloring, "cmap": cmap_name}
            if solid is not None:
                spec["color"] = solid[plot]
            if want_features:
                spec["features"] = True
            clouds.append(spec)
        if not clouds:
            return
        origin = _cloud_centroid(Path(clouds[0]["path"]))
        self._viewer.send(clouds, origin=origin, slope_strata_pct=self._slope_strata_pct(),
                          height_channel=self._height_channel())


class _DistributionExport:
    """Export provider for the left-click distribution panel (and its right-click pop-up).

    Re-renders the currently shown plot's distribution into an export figure via the owner's
    ``_render_distribution``; exports the empty placeholder when no plot is selected yet.
    """

    def __init__(self, owner: "MapView") -> None:
        self._owner = owner

    def draw_into(self, ax) -> None:
        owner = self._owner
        if owner._dist_plot is None or not owner._render_distribution(
            ax, owner._dist_plot, owner.dist_field.value(),
            owner.dist_style.currentText(), big=True,
        ):
            ax.clear()
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.5, 0.5, "Left-click a plot to see its distribution.", ha="center",
                    va="center", color="gray", transform=ax.transAxes)

    def export_title(self) -> str:
        plot = self._owner._dist_plot
        return f"distribution_plot({plot})" if plot is not None else "distribution"


# --------------------------------------------------------------------------- #
# Small helpers                                                               #
# --------------------------------------------------------------------------- #
def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def solid_colours(all_values: dict[int, float], selected, cmap_name: str) -> dict[int, list]:
    """One solid RGB per selected plot, the colour map stretched over **all** plots' values.

    ``all_values`` is every plot on the current map (not just ``selected``), so a cloud's colour
    equals its map-view square regardless of which subset is selected. Non-finite values map to grey.
    """
    cmap = get_cmap(cmap_name)
    norm = full_range_norm(all_values.values())
    out: dict[int, list] = {}
    for p in selected:
        v = all_values.get(p, float("nan"))
        out[p] = list(cmap(norm(v))[:3]) if np.isfinite(v) else [0.6, 0.6, 0.6]
    return out


def _cloud_centroid(path) -> list[float] | None:
    """Centroid of a cloud's absolute XYZ, used as the shared multi-cloud origin."""
    if path is None or not path.exists():
        return None
    try:
        import laspy

        las = laspy.read(path)
        return [float(np.mean(las.x)), float(np.mean(las.y)), float(np.mean(las.z))]
    except Exception:  # noqa: BLE001
        return None


def _format_value(val: float, mode: str, field: str) -> str:
    if not np.isfinite(val):
        return ""
    if mode == _NORM_Z:
        return f"{val:+.2f}"
    if mode == _NORM_PCT:
        return f"{val:.0f}%"
    if field == _PLOT_FIELD:  # plot numbers are integers; never show "429.0"
        return f"{int(round(val))}"
    if abs(val) >= 1000 or (abs(val) < 0.01 and val != 0):
        return f"{val:.2g}"
    return f"{val:.1f}"

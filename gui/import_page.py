"""Tab 1: **Import** — get plot point clouds (and their reference sheet) into the project.

Everything the pipeline needs to start comes in here, by one of two routes the user picks with a
radio button:

* **A folder of separate plot files** — one file per plot already, from a previous crop, a
  collaborator, or a published dataset. ``.las``/``.laz``/``.pcd`` are all accepted;
  :mod:`clip.importer` reads the plot number out of each file name, converts to ``.laz`` and
  writes the canonical ``plot(N)``. It can also rescale the coordinates (a dataset recorded in
  millimetres) and spread overlapping plots onto a grid.
* **One whole-field cloud + a ``.gpkg`` of plot masks** — :mod:`clip.clipper` cuts the cloud to
  each polygon, taking the plot number from the polygon's own chosen attribute.

Both routes end in the same place: canonical ``plot(N)`` files in the project's ``plots/`` folder,
which is what every later tab reads. Both also share the two "Label starts with / ends with" boxes,
because both have to reduce something external — a file name in one case, a mask attribute value in
the other — to the plain integer the rest of the pipeline keys off.

The tab also takes the **reference sheet** (the measured ground truth). It owns the file and the
link only: which column holds the plot number, and how many plots actually matched a row — the
question you want answered at import time, not two tabs later. Choosing what to *do* with those
columns (which become features, which is the ML target) stays on the Feature generation tab, which
reads the loaded sheet from the shared :class:`~common.session.Session`.

Inputs are *pinned* in the active project, so re-opening one restores them.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from clip import (
    DEFAULT_UNIT,
    UNIT_SCALES,
    GridLayout,
    list_mask_fields,
    detect_units,
    list_sources,
    load_masks,
    plan_import,
    unit_name,
    unit_scale,
)
from common import las_io
from common.config import CONFIG
from common.naming import (
    LAS_SUFFIXES,
    KeyFormat,
    LabelFormat,
    aug_number_from_name,
    plot_number_from_name,
)
from common.session import Session
from featuregen.external import (
    column_values,
    default_reference_dir,
    list_reference_columns,
    load_external,
)
from .results.plot_map import PlotMapView, file_map_from_folder
from .widgets import (
    ClearSection,
    DropLineEdit,
    cascade_clear_dialog,
    downstream_feature_files,
    downstream_model_files,
    make_console,
    make_status_indicator,
    open_folder,
    set_status_indicator,
    style_button,
)
from .workers import ClipWorker, ImportWorker

# The two source routes, as stacked-widget indices.
_MODE_FOLDER = 0
_MODE_MASKS = 1

# Placeholder for the reference sheet's key-column selector: "no explicit choice", which
# load_external resolves to the sheet's first column.
_NO_KEY = "(none)"

# How many offending names a duplicate/unreadable warning lists before trailing off.
_MAX_LISTED = 6

# The units dropdown's first entry: take the unit from the coordinate system the files
# declare. Only a declaration counts - when there is none the import falls back to metres
# and says so, rather than inferring a unit from the shape of the data.
_AUTO_UNIT = "Read from the files"


class ImportPage(QWidget):
    """The import module UI (the first pipeline tab)."""

    request_next = Signal()  # ask the shell to switch to data augmentation
    data_invalidated = Signal()  # plots changed (Clear): shell drops stale cached datasets

    def __init__(self, session: Session) -> None:
        super().__init__()
        self._session = session
        self._worker: ClipWorker | ImportWorker | None = None
        self._masks: list | None = None
        self._output_dir_value: Path | None = None
        self._point_cache: dict[Path, tuple[float, int]] = {}  # path -> (mtime, point count)

        root = QHBoxLayout(self)
        root.addWidget(self._build_source_panel(), 5)
        root.addWidget(self._build_run_panel(), 5)
        # Both panels exist now, so the identity wording can reach the reference-sheet captions
        # it also owns (the left panel is built first, before those widgets are made).
        self._apply_identity_wording(self.use_row.isChecked())

        # Pin the chosen inputs into the project whenever they change, so re-opening restores them.
        self.cloud_path.textChanged.connect(lambda t: self._pin("cloud", t))
        self.mask_path.textChanged.connect(lambda t: self._pin("mask", t))
        self.folder_path.textChanged.connect(lambda t: self._pin("import_folder", t))
        self.reference_path.textChanged.connect(lambda t: self._pin("reference", t))

    def _pin(self, key: str, value: str) -> None:
        """Persist an input path into the active project (no-op when no project is open)."""
        if self._session.project is not None:
            self._session.project.set_pin(key, value.strip())

    # ------------------------------------------------------------------ #
    # Panels                                                             #
    # ------------------------------------------------------------------ #
    def _build_source_panel(self) -> QWidget:
        box = QGroupBox("Where the plots come from")
        layout = QVBoxLayout(box)

        self.mode_folder = QRadioButton("A folder of separate plot files")
        self.mode_folder.setToolTip(
            "One file per plot already (.las/.laz/.pcd). They are copied into this project and "
            "renamed to plot(N)."
        )
        self.mode_masks = QRadioButton("One point cloud + plot masks")
        self.mode_masks.setToolTip(
            "A whole-field .las/.laz cut into plots using a .gpkg of labelled polygons."
        )
        self.mode_folder.setChecked(True)
        layout.addWidget(self.mode_folder)
        layout.addWidget(self.mode_masks)

        self.branches = QStackedWidget()
        self.branches.addWidget(self._build_folder_branch())
        self.branches.addWidget(self._build_masks_branch())
        layout.addWidget(self.branches)
        self.mode_folder.toggled.connect(self._on_mode_changed)

        layout.addWidget(self._build_label_box())
        layout.addWidget(self._build_map_box(), 1)
        return box

    def _build_folder_branch(self) -> QWidget:
        """Route A: a folder of per-plot files, with unit scaling and the optional grid layout."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        row = QHBoxLayout()
        # folder_only: dropping one of the plot files resolves to the folder that holds them,
        # so "drag the plots in" works as naturally as dragging the folder itself.
        self.folder_path = DropLineEdit(folder_only=True)
        self.folder_path.setPlaceholderText("Drag the plot files (or their folder) here…")
        self.folder_path.textChanged.connect(self._on_folder_changed)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._on_browse_folder)
        row.addWidget(self.folder_path, 1)
        row.addWidget(browse)
        layout.addLayout(row)

        self.folder_status = QLabel("No folder chosen.")
        self.folder_status.setWordWrap(True)
        layout.addWidget(self.folder_status)

        form = QFormLayout()
        # Units matter because every downstream metric assumes metres: canopy-height thresholds,
        # entropy bin sizes and the viewer's radii are all absolute. A dataset stored in
        # millimetres imported as-is produces features that are 1000x off and silently wrong.
        self.units = QComboBox()
        self.units.addItem(_AUTO_UNIT)
        self.units.addItems(list(UNIT_SCALES))
        self.units.setCurrentText(_AUTO_UNIT)
        self.units.setToolTip(
            "The units the source files are recorded in. Everything downstream assumes metres, "
            "so anything else has to be converted on the way in.\n"
            "Read from the files uses the coordinate system a .las/.laz declares. A .pcd never "
            "carries one, and many .las files do not either — then the unit is unknown, the import "
            "reads them as metres, and the line under the folder above says so. Pick a unit here "
            "to state it yourself."
        )
        self.units.currentTextChanged.connect(self._on_folder_changed)
        form.addRow("Source units", self.units)
        layout.addLayout(form)

        self.use_grid = QCheckBox("Spread the plots onto a grid")
        self.use_grid.setToolTip(
            "For datasets that store every plot re-centred on its own origin, so they all pile up "
            "on one spot when loaded together. This lays them out in rows instead.\n"
            "Positions only — plot shapes, heights and every computed feature are unaffected."
        )
        self.use_grid.toggled.connect(self._on_grid_toggled)
        layout.addWidget(self.use_grid)

        grid_form = QFormLayout()
        self.grid_columns = QSpinBox()
        self.grid_columns.setRange(1, 1000)
        self.grid_columns.setValue(10)
        self.grid_columns.setToolTip("How many plots per row of the grid.")
        self.grid_padding = QDoubleSpinBox()
        self.grid_padding.setRange(0.0, 500.0)
        self.grid_padding.setValue(10.0)
        self.grid_padding.setSuffix(" %")
        self.grid_padding.setToolTip(
            "Gap between plots, as a percentage of the largest plot's size."
        )
        grid_form.addRow("Columns", self.grid_columns)
        grid_form.addRow("Spacing", self.grid_padding)
        self.grid_box = QWidget()
        self.grid_box.setLayout(grid_form)
        self.grid_box.setEnabled(False)
        layout.addWidget(self.grid_box)
        return page

    def _build_masks_branch(self) -> QWidget:
        """Route B: one whole-field cloud cut by a GeoPackage of labelled plot polygons."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Whole-field point cloud (.las/.laz)"))
        cloud_row = QHBoxLayout()
        self.cloud_path = DropLineEdit()
        self.cloud_path.setPlaceholderText("Drag a .las/.laz here or browse…")
        cloud_browse = QPushButton("Browse…")
        cloud_browse.clicked.connect(self._on_browse_cloud)
        cloud_row.addWidget(self.cloud_path, 1)
        cloud_row.addWidget(cloud_browse)
        layout.addLayout(cloud_row)

        layout.addWidget(QLabel("Plot masks (GeoPackage .gpkg)"))
        mask_row = QHBoxLayout()
        self.mask_path = DropLineEdit()
        self.mask_path.setPlaceholderText("Drag a .gpkg here or browse…")
        # Masks load automatically once a path is present (no separate "Load masks" button).
        self.mask_path.textChanged.connect(self._on_load_masks)
        mask_browse = QPushButton("Browse…")
        mask_browse.clicked.connect(self._on_browse_masks)
        mask_row.addWidget(self.mask_path, 1)
        mask_row.addWidget(mask_browse)
        layout.addLayout(mask_row)

        form = QFormLayout()
        self.label_field = QComboBox()
        self.label_field.setToolTip(
            "Which polygon attribute holds the plot number. Its value becomes plot(N), read "
            "through the label boxes below."
        )
        self.label_field.currentTextChanged.connect(self._on_label_field_changed)
        form.addRow("Plot label field", self.label_field)
        layout.addLayout(form)

        self.mask_status = QLabel("No masks loaded.")
        self.mask_status.setWordWrap(True)
        layout.addWidget(self.mask_status)
        return page

    def _build_label_box(self) -> QWidget:
        """Where the plot's identity sits in the source name — one number, or a row/column pair.

        Shared by both routes on purpose: the folder route reads the identity out of a *file name*
        and the mask route out of a *polygon attribute value*, but it is one idea.

        The optional **row** pair is what makes grid-shaped datasets work. Some data identifies a
        plot by a single id; some identifies it by its position in the field, where neither
        coordinate is unique alone (SGCBP has 234 plots but only 18 distinct range numbers). Naming
        both halves keeps every plot distinct, lets the reference sheet be keyed the same way, and
        tells the grid option where each plot actually belongs.
        """
        box = QGroupBox("How plots are identified")
        layout = QVBoxLayout(box)
        self.label_hint = QLabel()
        self.label_hint.setWordWrap(True)
        self.label_hint.setStyleSheet("color: gray;")
        layout.addWidget(self.label_hint)

        self.use_row = QCheckBox("By a row and column (a position in the field)")
        self.use_row.setToolTip(
            "Off: each plot has one id number, e.g. plot(429).\n"
            "On: a plot is identified by where it sits in the field, and neither number is unique "
            "on its own — e.g. 12-13-1-b.pcd is row 12, column 13.\n"
            "Both numbers are then used to identify the plot, key the reference sheet, and rebuild "
            "the field layout."
        )
        self.use_row.toggled.connect(self._on_use_row_toggled)
        layout.addWidget(self.use_row)

        # The row pair sits *above* the column pair, so the fields read in the same order as the
        # name they describe (12-13-1-b.pcd is row, then column).
        row_form = QFormLayout()
        self.row_start = QLineEdit(CONFIG.label_row_start)
        self.row_start.setToolTip("Text immediately before the row number (often nothing).")
        self.row_start.textChanged.connect(self._on_label_format_changed)
        self.row_end = QLineEdit(CONFIG.label_row_end)
        self.row_end.setToolTip("Text immediately after the row number, e.g. '-'.")
        self.row_end.textChanged.connect(self._on_label_format_changed)
        row_form.addRow("Row starts with", self.row_start)
        row_form.addRow("Row ends with", self.row_end)
        self.row_box = QWidget()
        self.row_box.setLayout(row_form)
        self.row_box.setVisible(False)
        layout.addWidget(self.row_box)

        fmt = QFormLayout()
        self.label_start = QLineEdit(CONFIG.label_start)
        self.label_start.textChanged.connect(self._on_label_format_changed)
        self.label_end = QLineEdit(CONFIG.label_end)
        self.label_end.textChanged.connect(self._on_label_format_changed)
        # Kept as fields so the captions can switch between "Plot number" and "Column" — the same
        # two boxes mean different halves of the identity depending on the toggle above.
        self.col_start_label = QLabel()
        self.col_end_label = QLabel()
        fmt.addRow(self.col_start_label, self.label_start)
        fmt.addRow(self.col_end_label, self.label_end)
        layout.addLayout(fmt)

        return box

    def _apply_identity_wording(self, paired: bool) -> None:
        """Word every identity field for the active mode: one plot number, or a row and a column.

        This covers the reference sheet's key selectors as well as the file-name boxes: the sheet
        has to be keyed on whatever identifies a plot, so both sides are the same question and are
        named the same way.
        """
        if paired:
            self.col_start_label.setText("Column starts with")
            self.col_end_label.setText("Column ends with")
            self.label_start.setToolTip("Text immediately before the column number.")
            self.label_end.setToolTip("Text immediately after the column number.")
            self.label_hint.setText(
                "The plot is identified by the two numbers between these strings.\n"
                "Example: 12-13-1-b.pcd is row 12, column 13 — "
                "row '' … '-', column '-' … '-1-b'."
            )
            self.reference_key_label.setText("Column column")
        else:
            self.col_start_label.setText("Plot number starts with")
            self.col_end_label.setText("Plot number ends with")
            self.label_start.setToolTip("Text immediately before the plot number, e.g. 'plot('.")
            self.label_end.setToolTip("Text immediately after the plot number, e.g. ')'.")
            self.label_hint.setText(
                "The plot is identified by the number between these strings.\n"
                "Example: plot(429).laz — 'plot(' … ')'."
            )
            self.reference_key_label.setText("Plot number column")
        # The sheet's row column is meaningless without a pair, so it goes away entirely rather
        # than sitting there greyed out.
        self.reference_row.setVisible(paired)
        self.reference_row_label.setVisible(paired)

    def _build_map_box(self) -> QWidget:
        """A plain 2-D view of the imported plots: their real footprints, each with its number.

        Purely informative — the quickest way to confirm the import produced the plots you expect,
        in the positions and orientation expected, without opening a viewer.
        """
        box = QGroupBox("Plot layout (2-D)")
        layout = QVBoxLayout(box)
        self.plot_map = PlotMapView("Imported plots")
        layout.addWidget(self.plot_map, 1)
        return box

    def _build_run_panel(self) -> QWidget:
        """The right-hand column, top to bottom: the reference sheet, the console, then the run.

        Ordered by when you touch it. The reference sheet is an *input*, so it sits with the other
        inputs at the top; the console reports what happened; and everything that acts — the
        output name, where it goes, and every button — is gathered at the bottom, so the controls
        are in one place instead of straddling the log.
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._build_reference_box())
        layout.addWidget(self._build_console_box(), 1)
        layout.addWidget(self._build_output_box())
        return panel

    def _build_console_box(self) -> QWidget:
        # The shared console block (its own bold header + clear bin) with this tab's progress bar
        # under it, so the run's log and its progress read as one unit.
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        console_box, self.log = make_console(self)
        layout.addWidget(console_box, 1)
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        return box

    def _build_output_box(self) -> QWidget:
        box = QGroupBox("Output & run")
        layout = QVBoxLayout(box)

        form = QFormLayout()
        self.universal_string = QLineEdit(CONFIG.universal_string)
        self.universal_string.setPlaceholderText("e.g. _siteA  (optional)")
        self.universal_string.setToolTip(
            "Text added to every exported name, inserted right after plot(N) and before any "
            "later _aug(k). Leave blank for plain plot(N).laz names."
        )
        form.addRow("Universal name string", self.universal_string)
        layout.addLayout(form)

        status_row = QHBoxLayout()
        self.indicator = make_status_indicator()
        status_row.addWidget(self.indicator)
        # Point totals sit to the right of the "N plots loaded" indicator: the raw size of the
        # imported data set, and its average density per plot.
        self.points_label = QLabel("")
        self.points_label.setStyleSheet("color: #495057;")
        status_row.addWidget(self.points_label)
        status_row.addStretch(1)
        layout.addLayout(status_row)

        self.output_label = QLabel("Output: (no project open)")
        self.output_label.setWordWrap(True)
        self.output_label.setStyleSheet("color: gray;")
        layout.addWidget(self.output_label)

        self.run_button = QPushButton("Import plots")
        self.run_button.clicked.connect(self._on_run)
        style_button(self.run_button, "primary")
        layout.addWidget(self.run_button)

        self.clear_button = QPushButton("Clear imported plots")
        self.clear_button.setToolTip(
            "Delete the imported plot files from this project — and, if you keep them ticked, "
            "everything derived from them (augmented copies, features/targets, saved models)."
        )
        self.clear_button.clicked.connect(self._on_clear)
        style_button(self.clear_button, "clear")
        layout.addWidget(self.clear_button)

        self.open_folder_button = QPushButton("Open output folder")
        self.open_folder_button.setEnabled(False)
        self.open_folder_button.clicked.connect(self._on_open_folder)
        layout.addWidget(self.open_folder_button)

        self.next_button = QPushButton("Go to data augmentation →")
        self.next_button.setEnabled(False)
        self.next_button.clicked.connect(lambda: self.request_next.emit())
        style_button(self.next_button, "next")
        layout.addWidget(self.next_button)
        return box

    def _build_reference_box(self) -> QWidget:
        """The reference sheet: the file, its plot-number column, and how many plots matched.

        Deliberately just the *link*. Which columns become features and which one is the ML target
        are decisions about the model, and stay on the Feature generation tab.
        """
        box = QGroupBox("Reference sheet (ground truth)")
        layout = QVBoxLayout(box)
        row = QHBoxLayout()
        self.reference_path = DropLineEdit()
        self.reference_path.setPlaceholderText("Drag a .csv/.xlsx here or browse…  (optional)")
        self.reference_path.textChanged.connect(self._on_reference_changed)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._on_browse_reference)
        row.addWidget(self.reference_path, 1)
        row.addWidget(browse)
        layout.addLayout(row)

        form = QFormLayout()
        # The sheet's row is listed first, so these read in the same order as the identity fields
        # on the left. Its caption switches with the same toggle: the sheet has to be keyed on
        # whatever identifies a plot, so "plot number" and "column" are the same choice named for
        # the mode it is in.
        self.reference_row = QComboBox()
        self.reference_row.addItem(_NO_KEY)
        self.reference_row.setToolTip(
            "The sheet's row column, when a plot is identified by two numbers. Must be the same "
            "row the file names carry."
        )
        self.reference_row.currentTextChanged.connect(self._on_reference_key_changed)
        self.reference_row_label = QLabel("Row column")
        form.addRow(self.reference_row_label, self.reference_row)

        self.reference_key = QComboBox()
        self.reference_key.addItem(_NO_KEY)
        self.reference_key.setToolTip(
            "Which column of the sheet holds the number that identifies a plot. Leave on (none) "
            "to use the sheet's first column."
        )
        self.reference_key.currentTextChanged.connect(self._on_reference_key_changed)
        self.reference_key_label = QLabel()
        form.addRow(self.reference_key_label, self.reference_key)
        # Row filter: keep only the rows where one column equals one value. A sheet covering
        # several scan dates lists every plot once per date, which makes the plot number ambiguous
        # and lets the lookup pick the wrong date's measurement; narrowing to one date is what
        # makes the key unique again.
        self.reference_filter_col = QComboBox()
        self.reference_filter_col.addItem(_NO_KEY)
        self.reference_filter_col.setToolTip(
            "Optional: use only the sheet rows where this column has a particular value — for a "
            "sheet covering several dates or growth stages, pick the one these plots came from."
        )
        self.reference_filter_col.currentTextChanged.connect(self._on_filter_column_changed)
        form.addRow("Only rows where", self.reference_filter_col)
        self.reference_filter_value = QComboBox()
        self.reference_filter_value.addItem(_NO_KEY)
        self.reference_filter_value.setEnabled(False)
        self.reference_filter_value.setToolTip("The value that column must have.")
        self.reference_filter_value.currentTextChanged.connect(self._on_reference_key_changed)
        form.addRow("equals", self.reference_filter_value)
        layout.addLayout(form)

        self.reference_status = QLabel("No reference sheet loaded.")
        self.reference_status.setWordWrap(True)
        layout.addWidget(self.reference_status)
        return box

    # ------------------------------------------------------------------ #
    # Small helpers                                                      #
    # ------------------------------------------------------------------ #
    def _label_format(self) -> LabelFormat:
        """The single-number format from the start/end boxes (also used for mask values)."""
        return LabelFormat(start=self.label_start.text(), end=self.label_end.text())

    def _row_format(self) -> LabelFormat | None:
        """The optional row format, or ``None`` when plots are identified by one number."""
        if not self.use_row.isChecked():
            return None
        return LabelFormat(start=self.row_start.text(), end=self.row_end.text())

    def _key_format(self) -> KeyFormat:
        """The full source-name key: the plot number, plus the row when the data has one."""
        return KeyFormat(col=self._label_format(), row=self._row_format())

    def _on_use_row_toggled(self, on: bool) -> None:
        self.row_box.setVisible(on)
        self._apply_identity_wording(on)
        self._on_label_format_changed()
        if self.reference_path.text().strip():
            self._load_reference()  # the sheet's key changed shape, so re-key it

    def _mode(self) -> int:
        return _MODE_FOLDER if self.mode_folder.isChecked() else _MODE_MASKS

    def _resolve_units(self, sources: list[Path]) -> tuple[float, str]:
        """The scale factor to apply and a sentence explaining it, honouring an explicit override.

        Detection is only consulted when the dropdown is on "Read from the files", so a unit the
        user picked by hand is never second-guessed. When the files declare nothing the answer is
        "unknown", not "metres": the note says so plainly, because a dataset that is really in
        millimetres would otherwise sail through with every metric 1000x out.
        """
        chosen = self.units.currentText()
        if chosen != _AUTO_UNIT:
            return unit_scale(chosen), f"Units: {unit_name(chosen)} (chosen)."
        if not sources:
            return 1.0, ""
        found = detect_units(sources)
        if found is None:
            return 1.0, (
                "⚠ Units: these files declare no coordinate system, so they are being read as "
                "metres. If they are not, pick the right unit above."
            )
        name, reason = found
        return UNIT_SCALES[name], f"Units: {name} — {reason}."

    def _grid(self) -> GridLayout | None:
        """The grid layout to apply on import, or ``None`` when the source positions are kept."""
        if not self.use_grid.isChecked():
            return None
        return GridLayout(
            columns=self.grid_columns.value(), padding=self.grid_padding.value() / 100.0
        )

    def _on_mode_changed(self) -> None:
        self.branches.setCurrentIndex(self._mode())
        self.run_button.setText(
            "Import plots" if self._mode() == _MODE_FOLDER else "Clip & export plots"
        )
        self._refresh_indicator()

    def _on_grid_toggled(self, on: bool) -> None:
        self.grid_box.setEnabled(on)

    def _on_label_format_changed(self) -> None:
        """Re-check both routes against the edited format so the counts stay honest as you type."""
        if self._mode() == _MODE_FOLDER:
            self._on_folder_changed()
        elif self.mask_path.text().strip() and self.label_field.currentText():
            self._on_label_field_changed(self.label_field.currentText())

    # ------------------------------------------------------------------ #
    # Route A: a folder of plot files                                    #
    # ------------------------------------------------------------------ #
    def _on_browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Choose a folder of plot files",
            self.folder_path.text().strip() or CONFIG.start_dir("clipped"),
        )
        if folder:
            self.folder_path.setText(folder)

    def _on_folder_changed(self) -> None:
        """Preview what the chosen folder would import: how many files, and any numbering trouble."""
        folder = self.folder_path.text().strip()
        if not folder:
            self.folder_status.setText("No folder chosen.")
            return
        sources = list_sources(folder)
        if not sources:
            self.folder_status.setText(
                f"No .las / .laz / .pcd files found in {Path(folder).name} or its sub-folders."
            )
            return
        plan = plan_import(sources, self._key_format())
        parts = [f"{len(sources)} file(s) found, {len(plan.entries)} with a readable plot number."]
        if plan.unreadable:
            parts.append(f"{len(plan.unreadable)} name(s) have no number in this format.")
        if plan.duplicates:
            parts.append(
                f"{len(plan.duplicates)} plot number(s) claimed by more than one file — "
                + ("add the row number to tell them apart."
                   if not self.use_row.isChecked() else
                   "the row and column do not identify these files uniquely.")
            )
        _, units_note = self._resolve_units(sources)
        if units_note:
            parts.append(units_note)
        self.folder_status.setText(" ".join(parts))

    def _confirm_numbering(self, plan) -> bool:
        """Warn about duplicate / unreadable plot identities. True = go ahead anyway.

        A duplicate means two files resolve to the same plot number, so one would overwrite the
        other and you would quietly end up with fewer plots than files. The fix is to describe the
        identity properly — usually by ticking the row number, since a single number is often just
        one coordinate of a grid position — not to invent new numbers, which would break the link
        to the reference sheet.
        """
        if plan.is_clean:
            return True
        lines: list[str] = []
        if plan.duplicates:
            listed = list(plan.duplicates.items())[:_MAX_LISTED]
            lines.append(
                f"{len(plan.duplicates)} plot number(s) are claimed by more than one file, so "
                "importing would overwrite one file with the other:"
            )
            lines += [
                f"  plot({n}): " + ", ".join(p.name for p in paths) for n, paths in listed
            ]
            if len(plan.duplicates) > _MAX_LISTED:
                lines.append(f"  …and {len(plan.duplicates) - _MAX_LISTED} more.")
        if plan.unreadable:
            names = ", ".join(p.name for p in plan.unreadable[:_MAX_LISTED])
            lines.append(
                f"\n{len(plan.unreadable)} file name(s) contain no plot number in this format "
                f"and would be skipped: {names}"
                + (" …" if len(plan.unreadable) > _MAX_LISTED else "")
            )

        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Warning)
        dialog.setWindowTitle("Plot numbering problem")
        dialog.setText("\n".join(lines))
        dialog.setInformativeText(
            "These names do not identify each plot uniquely. If a plot is identified by its "
            "position in the field, tick “The name also carries a row number” and give both "
            "halves — that keeps the plots distinct and keeps them matched to the reference sheet."
        )
        anyway = dialog.addButton("Import anyway", QMessageBox.DestructiveRole)
        dialog.addButton(QMessageBox.Cancel)
        dialog.exec()
        return dialog.clickedButton() is anyway

    def _start_folder_import(self) -> None:
        folder = self.folder_path.text().strip()
        sources = list_sources(folder)
        if not sources:
            QMessageBox.warning(
                self, "No plot files",
                f"No .las / .laz / .pcd files found in:\n{folder}\n(or its sub-folders)",
            )
            return
        fmt = self._key_format()
        plan = plan_import(sources, fmt)
        if not plan.entries:
            QMessageBox.warning(
                self, "No plot numbers",
                "None of the file names contain a plot number in this format, so there is "
                "nothing to import. Adjust the boxes under “Where the plot number sits”.",
            )
            return
        if not self._confirm_numbering(plan):
            return

        self._output_dir_value = self._session.project.plots_dir
        self._set_running(True)
        self.log.clear()
        self.progress.setRange(0, len(plan.entries))
        self.progress.setValue(0)
        self._log(f"Importing {len(sources)} file(s) from {folder} → {self._output_dir_value}")
        if self.use_row.isChecked():
            self._log(f"  Identifying plots by (row, column); packing stride {plan.stride}.")
        scale, units_note = self._resolve_units(sources)
        if units_note:
            self._log(f"  {units_note}")
        if scale != 1.0:
            self._log(f"  Scaling coordinates by {scale}.")
        grid = self._grid()
        if grid is not None:
            self._log(
                "  Rebuilding the field layout from the row/column numbers."
                if self.use_row.isChecked() else
                f"  Laying plots out on a grid: {grid.columns} per row."
            )

        self._worker = ImportWorker(
            sources, self._output_dir_value, fmt, self.universal_string.text().strip(),
            scale, grid,
        )
        self._worker.progressed.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_import_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    # ------------------------------------------------------------------ #
    # Route B: one cloud + masks                                         #
    # ------------------------------------------------------------------ #
    def _on_browse_cloud(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose whole-field point cloud",
            self.cloud_path.text().strip() or CONFIG.start_dir("cloud"),
            "LiDAR point clouds (*.las *.laz)",
        )
        if path:
            self.cloud_path.setText(path)

    def _on_browse_masks(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose plot masks",
            self.mask_path.text().strip() or CONFIG.start_dir("mask"), "GeoPackage (*.gpkg)"
        )
        if path:
            self.mask_path.setText(path)  # textChanged auto-loads the masks

    def _on_load_masks(self) -> None:
        path = self.mask_path.text().strip()
        if not path:
            return  # auto-triggered on every edit; nothing to load yet
        try:
            fields = list_mask_fields(path)
        except Exception as exc:  # noqa: BLE001
            self.mask_status.setText(f"Could not read masks: {exc}")
            return
        self.label_field.blockSignals(True)
        self.label_field.clear()
        self.label_field.addItems(fields)
        # Pre-select the preset's configured label field when it is one of the columns.
        if CONFIG.label_field and CONFIG.label_field in fields:
            self.label_field.setCurrentText(CONFIG.label_field)
        self.label_field.blockSignals(False)
        self._masks = None
        if fields:
            self._on_label_field_changed(self.label_field.currentText())
        else:
            self.mask_status.setText("The GeoPackage has no attribute columns to use as a label.")

    def _on_label_field_changed(self, field: str) -> None:
        path = self.mask_path.text().strip()
        if not path or not field:
            return
        try:
            masks, crs, unreadable = load_masks(path, field, self._label_format())
        except Exception as exc:  # noqa: BLE001
            self._masks = None
            self.mask_status.setText(f"Could not load polygons: {exc}")
            self._refresh_indicator()
            return
        self._masks = masks
        message = (
            f"Loaded {len(masks)} polygon(s) labelled by '{field}'. CRS: {crs}. "
            "The cloud must be in the same CRS."
        )
        if unreadable:
            listed = ", ".join(unreadable[:_MAX_LISTED])
            message += (
                f"\n{len(unreadable)} polygon(s) skipped — no plot number in '{field}' "
                f"(e.g. {listed}). Check the label boxes below."
            )
        self.mask_status.setText(message)
        self._refresh_indicator()

    def _start_clip(self) -> None:
        cloud = self.cloud_path.text().strip()
        if not cloud or Path(cloud).suffix.lower() not in LAS_SUFFIXES:
            QMessageBox.warning(self, "No cloud", "Choose a .las/.laz point cloud.")
            return
        if not self._masks:
            QMessageBox.warning(self, "No masks", "Load a .gpkg and pick a label field first.")
            return

        self._output_dir_value = self._session.project.plots_dir
        self._set_running(True)
        self.log.clear()
        self.progress.setRange(0, len(self._masks))
        self.progress.setValue(0)
        self._log(
            f"Clipping {len(self._masks)} plot(s) from {Path(cloud).name} → {self._output_dir_value}"
        )
        self._worker = ClipWorker(
            cloud, self._masks, self._output_dir_value, self.universal_string.text().strip()
        )
        self._worker.progressed.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_clip_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    # ------------------------------------------------------------------ #
    # Run / results                                                      #
    # ------------------------------------------------------------------ #
    def _on_run(self) -> None:
        if self._session.project is None:
            QMessageBox.warning(
                self, "No project", "Open or create a project on the Setup tab first."
            )
            return
        if self._mode() == _MODE_FOLDER:
            self._start_folder_import()
        else:
            self._start_clip()

    def _on_progress(self, done: int, total: int, label: str, error) -> None:
        self.progress.setValue(done)
        self._log(f"  [{'skipped' if error else 'ok'}] {label}" + (f": {error}" if error else ""))

    def _on_import_finished(self, result) -> None:
        self._set_running(False)
        self._log(
            f"Done. {result.n_written} plot(s) imported, {len(result.skipped)} skipped. "
            f"Output: {result.output_dir}"
        )
        if result.manifest_path is not None:
            self._log(f"Import manifest (source file → plot number): {result.manifest_path.name}")
        self._adopt_output(result.output_dir, result.written)
        QMessageBox.information(
            self, "Import finished",
            f"{result.n_written} plot file(s) imported to:\n{result.output_dir}"
            + (f"\n\n{len(result.skipped)} skipped (see the log)." if result.skipped else ""),
        )

    def _on_clip_finished(self, result) -> None:
        self._set_running(False)
        self._log(
            f"Done. {result.n_written} plot(s) written, {len(result.empty)} empty, "
            f"{len(result.failed)} failed. Output: {result.output_dir}"
        )
        if result.empty:
            self._log("Empty polygons (no points inside): " + ", ".join(result.empty))
        self._adopt_output(result.output_dir, result.written)
        QMessageBox.information(
            self, "Clipping finished",
            f"{result.n_written} plot file(s) written to:\n{result.output_dir}",
        )

    def _adopt_output(self, output_dir: Path, written: list[Path]) -> None:
        """Hand the freshly produced plots to the session and refresh every readout."""
        self._session.input_files = list(written)
        self._session.augment_output_dir = output_dir
        self._session.manifest_path = None
        self._output_dir_value = output_dir
        self.open_folder_button.setEnabled(True)
        self.next_button.setEnabled(bool(written))
        self._point_cache.clear()
        self._refresh_indicator()
        self._refresh_map()
        self._refresh_reference_match()  # the plot set changed, so the match count did too

    def _on_failed(self, message: str) -> None:
        self._set_running(False)
        self._log(f"ERROR: {message}")
        QMessageBox.critical(self, "Import failed", message)

    # ------------------------------------------------------------------ #
    # Reference sheet                                                    #
    # ------------------------------------------------------------------ #
    def _on_browse_reference(self) -> None:
        start = self.reference_path.text().strip() or default_reference_dir()
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose reference sheet", start, "Reference tables (*.csv *.xlsx *.xls)"
        )
        if path:
            self.reference_path.setText(path)

    def _on_reference_changed(self) -> None:
        """A new sheet was chosen: offer its headers as key columns, then load it."""
        path = self.reference_path.text().strip()
        if not path:
            self._clear_reference()
            return
        try:
            headers = list_reference_columns(path)
        except Exception:  # noqa: BLE001 - _load_reference reports the real error below
            headers = []
        previous = self.reference_key.currentText()
        previous_row = self.reference_row.currentText()
        self.reference_key.blockSignals(True)
        self.reference_key.clear()
        self.reference_key.addItem(_NO_KEY)
        self.reference_key.addItems(headers)
        # Keep the user's column across a reload of the same sheet; otherwise take the preset's
        # own reference key column, and fall back to the placeholder (= the sheet's first column).
        # This is the sheet's key, which is a different choice from label_field (the *mask*
        # attribute) — reading that one here was why a preset's reference_key_column did nothing.
        if previous in headers:
            self.reference_key.setCurrentText(previous)
        elif CONFIG.reference_key_column and CONFIG.reference_key_column in headers:
            self.reference_key.setCurrentText(CONFIG.reference_key_column)
        self.reference_key.blockSignals(False)
        self.reference_row.blockSignals(True)
        self.reference_row.clear()
        self.reference_row.addItem(_NO_KEY)
        self.reference_row.addItems(headers)
        if previous_row in headers:
            self.reference_row.setCurrentText(previous_row)
        elif CONFIG.reference_row_column in headers:
            self.reference_row.setCurrentText(CONFIG.reference_row_column)
        self.reference_row.blockSignals(False)

        previous_filter = self.reference_filter_col.currentText()
        self.reference_filter_col.blockSignals(True)
        self.reference_filter_col.clear()
        self.reference_filter_col.addItem(_NO_KEY)
        self.reference_filter_col.addItems(headers)
        if previous_filter in headers:
            self.reference_filter_col.setCurrentText(previous_filter)
        elif CONFIG.reference_filter_column in headers:
            self.reference_filter_col.setCurrentText(CONFIG.reference_filter_column)
        self.reference_filter_col.blockSignals(False)
        self._refresh_filter_values(preferred=CONFIG.reference_filter_value)
        self._load_reference()

    def _refresh_filter_values(self, preferred: str = "") -> None:
        """Offer the distinct values of the chosen filter column, so it is picked, not typed."""
        path = self.reference_path.text().strip()
        column = self._selected_filter_column()
        previous = self.reference_filter_value.currentText()
        values = column_values(path, column) if (path and column) else []
        self.reference_filter_value.blockSignals(True)
        self.reference_filter_value.clear()
        self.reference_filter_value.addItem(_NO_KEY)
        self.reference_filter_value.addItems(values)
        if previous in values:
            self.reference_filter_value.setCurrentText(previous)
        elif preferred and preferred in values:
            self.reference_filter_value.setCurrentText(preferred)
        self.reference_filter_value.blockSignals(False)
        self.reference_filter_value.setEnabled(bool(values))

    def _on_filter_column_changed(self, _text: str) -> None:
        self._refresh_filter_values()
        self._on_reference_key_changed("")

    def _on_reference_key_changed(self, _text: str) -> None:
        if self.reference_path.text().strip():
            self._load_reference()

    def _selected_reference_key(self) -> str | None:
        text = self.reference_key.currentText()
        return None if text == _NO_KEY else text

    def _selected_reference_row(self) -> str | None:
        # Only meaningful when the plots themselves are identified by a pair.
        if not self.use_row.isChecked():
            return None
        text = self.reference_row.currentText()
        return None if text == _NO_KEY else text

    def _selected_filter_column(self) -> str | None:
        text = self.reference_filter_col.currentText()
        return None if text == _NO_KEY else text

    def _selected_filter_value(self) -> str | None:
        text = self.reference_filter_value.currentText()
        return None if text == _NO_KEY else text

    def _import_stride(self) -> int | None:
        """The packing stride the current folder import would use, so the sheet matches it.

        Both sides must fold a (row, column) pair into the same integer. The stride comes from the
        widest column in the *plot* data, so it is read from the folder plan rather than from the
        sheet, which may cover a different subset.
        """
        folder = self.folder_path.text().strip()
        if self._mode() != _MODE_FOLDER or not folder:
            return None
        sources = list_sources(folder)
        if not sources:
            return None
        return plan_import(sources, self._key_format()).stride

    def _clear_reference(self) -> None:
        self._session.reference_path = None
        self._session.reference_key = None
        self._session.external = None
        self.reference_status.setText("No reference sheet loaded.")

    def _load_reference(self) -> None:
        """Read the sheet keyed on the chosen column into the session, and report the match."""
        path = self.reference_path.text().strip()
        key = self._selected_reference_key()
        try:
            external = load_external(
                path,
                key_column=key,
                label_format=self._label_format(),
                row_column=self._selected_reference_row(),
                stride=self._import_stride(),
                filter_column=self._selected_filter_column(),
                filter_value=self._selected_filter_value(),
            )
        except Exception as exc:  # noqa: BLE001
            self._clear_reference()
            self.reference_status.setText(f"Could not load: {exc}")
            return
        self._session.external = external
        self._session.reference_path = Path(path)
        self._session.reference_key = key
        if self._session.project is not None:
            self._session.project.set_pin("reference_key", key or "")
            self._session.project.set_pin("reference_row", self._selected_reference_row() or "")
        self._refresh_reference_match()

    def _refresh_reference_match(self) -> None:
        """Say how many of the project's plots actually found a row — the point of doing this here.

        A sheet that loads fine but matches nothing (wrong key column, or plot numbers that mean
        something different) is the failure worth catching at import time; further down the
        pipeline it surfaces only as a table full of blanks.
        """
        external = self._session.external
        if external is None:
            return
        name = Path(self._session.reference_path or "").name
        base = f"{len(external.frame)} row(s), {len(external.columns)} column(s) from {name}."

        # An ambiguous key is worse than an unmatched one: the lookup silently takes the first
        # matching row, so the plot gets a real-looking value that may belong to a different scan
        # date. Say so before it reaches the feature table.
        if external.duplicate_keys:
            listed = ", ".join(str(k) for k in external.duplicate_keys[:_MAX_LISTED])
            base += (
                f" ⚠ {len(external.duplicate_keys)} plot number(s) appear on more than one row "
                f"(e.g. {listed}) — each match takes the first row, which may be the wrong one. "
                "Use “Only rows where” above to narrow the sheet to one date/stage."
            )

        plots = self._plot_files()
        if not plots:
            self.reference_status.setText(base + " Import plots to check the match.")
            return
        numbers = {plot_number_from_name(p.name) for p in plots}
        numbers.discard(None)
        matched = sum(1 for n in numbers if n in external.frame.index)
        summary = f"{base} {matched} of {len(numbers)} plot(s) matched a row."
        if matched == 0:
            summary += (
                " Nothing matched — check the “Plot number column” above and the label boxes."
            )
        elif matched < len(numbers):
            missing = sorted(n for n in numbers if n not in external.frame.index)
            listed = ", ".join(str(n) for n in missing[:_MAX_LISTED])
            summary += f" No row for plot(s): {listed}" + (" …" if len(missing) > _MAX_LISTED else "")
        self.reference_status.setText(summary)

    # ------------------------------------------------------------------ #
    # Clear                                                              #
    # ------------------------------------------------------------------ #
    def _on_clear(self) -> None:
        """Delete this project's imported plots, cascading into everything derived from them.

        The plots are the root of the pipeline, so the cascade offers every later stage: the
        augmented copies + manifest made from them, the feature/targets tables computed over them,
        and any saved model whose field map they define. All downstream sections default to checked
        (see :func:`gui.widgets.cascade_clear_dialog`).
        """
        if self._session.project is None:
            QMessageBox.warning(self, "No project", "Open or create a project first.")
            return
        project = self._session.project
        folder = project.plots_dir
        clouds = [p for p in folder.glob("*") if p.suffix.lower() in LAS_SUFFIXES]
        originals = [p for p in clouds if aug_number_from_name(p.name) is None]
        augmented = [p for p in clouds if aug_number_from_name(p.name) is not None]
        manifests = list(folder.glob("*manifest*.csv"))
        sections = [
            ClearSection("plots", "Imported plot files", originals, primary=True),
            ClearSection("augmented", "Augmented copies + manifest", augmented + manifests),
            ClearSection("features", "Generated features + targets",
                         downstream_feature_files(project.features_dir)),
            ClearSection("models", "Saved models", downstream_model_files(project.model_dir)),
        ]
        removed = cascade_clear_dialog(self, "Clear imported plots", sections)
        if not removed:
            return
        if removed.get("plots"):
            self._session.input_files = []
            self.next_button.setEnabled(False)
        if removed.get("augmented"):
            self._session.manifest_path = None
        if removed.get("features"):
            self._session.feature_table_path = None
            self._session.targets_table_path = None
        total = sum(removed.values())
        self._log(
            f"Cleared {total} file(s): "
            + ", ".join(f"{k}={n}" for k, n in removed.items() if n)
            + f". ({folder})"
        )
        self._point_cache.clear()
        self._refresh_indicator()
        self._refresh_map()
        # The learning tabs may hold datasets built from the clouds we just deleted.
        self.data_invalidated.emit()

    def _on_open_folder(self) -> None:
        if self._output_dir_value:
            open_folder(self._output_dir_value)

    # ------------------------------------------------------------------ #
    # Page lifecycle                                                     #
    # ------------------------------------------------------------------ #
    def _set_running(self, running: bool) -> None:
        self.run_button.setEnabled(not running)
        if running:
            self.run_button.setText("Working…")
        else:
            self._on_mode_changed()  # restores the per-route button text

    def _log(self, message: str) -> None:
        self.log.appendPlainText(message)

    def on_enter(self) -> None:  # called by the shell when this page is shown
        """Seed every field from the active project's pinned inputs."""
        project = self._session.project
        if project is None:
            self.output_label.setText("Output: (no project open)")
            set_status_indicator(self.indicator, False, "No project open.")
            self.points_label.setText("")
            self.plot_map.clear()
            return
        self.output_label.setText(f"Output → {project.plots_dir}")
        self._apply_preset()
        # Allow advancing if plots already exist from a previous run (mirrors the augmentation /
        # feature tabs, so an opened project can move forward without re-importing).
        n_plots = self._plot_count()
        if n_plots and not self.next_button.isEnabled():
            self.next_button.setEnabled(True)
        # Fill each field from the project pin only when it differs, so we don't clobber an
        # in-progress edit or trigger needless re-saves.
        # The project's own pin wins; a preset's path is the fallback for a project that has not
        # chosen one yet, so opening a fresh project with a dataset preset active arrives ready.
        # Read every pin up front: seeding a field writes its own pin back, which would otherwise
        # make the route check below see paths this call had just filled in.
        pinned = {k: project.pin(k) for k in ("cloud", "mask", "import_folder")}
        self._seed_field(self.cloud_path, pinned["cloud"] or CONFIG.input_path("cloud_file"))
        self._seed_field(self.mask_path, pinned["mask"] or CONFIG.input_path("mask_file"))
        self._seed_field(
            self.folder_path, pinned["import_folder"] or CONFIG.input_path("import_folder")
        )
        # A preset that names a folder of per-plot files (or a cloud + masks) also says which of
        # the two routes it is for, so the radio follows it rather than being set by hand. Only
        # when the project itself has not already chosen — a returning project keeps its route.
        if not pinned["import_folder"] and not pinned["cloud"]:
            if CONFIG.input_path("import_folder"):
                self.mode_folder.setChecked(True)
            elif CONFIG.input_path("cloud_file") or CONFIG.input_path("mask_file"):
                self.mode_masks.setChecked(True)
        self._seed_reference(project)
        self._refresh_indicator(n_plots)
        self._refresh_map()

    def apply_preset_defaults(self) -> None:
        """Re-seed every preset-driven control, letting the newly chosen preset win over the pins.

        Called by the shell when the user picks a preset on Setup. ``on_enter`` deliberately lets a
        project's pinned paths win over the preset (a returning project keeps the inputs it was set
        up with), but choosing a preset by hand is the opposite intent: here the preset's paths,
        route and reference columns overwrite what is on screen, and the new values re-pin
        themselves through the fields' own handlers. A value the preset leaves blank is left alone
        rather than cleared, so a preset that only carries feature settings does not wipe a working
        project's inputs.
        """
        self._apply_preset()
        for line_edit, key in (
            (self.cloud_path, "cloud_file"),
            (self.mask_path, "mask_file"),
            (self.folder_path, "import_folder"),
        ):
            path = CONFIG.input_path(key)
            if path:
                line_edit.setText(path)
        if CONFIG.input_path("import_folder"):
            self.mode_folder.setChecked(True)
        elif CONFIG.input_path("cloud_file") or CONFIG.input_path("mask_file"):
            self.mode_masks.setChecked(True)
        reference = CONFIG.input_path("reference_file")
        if reference:
            # Drop the current column choices first: _on_reference_changed keeps a selection that
            # is still among the headers, which would otherwise hold the previous preset's key
            # column in place even though the new preset names its own.
            for combo in (self.reference_key, self.reference_row, self.reference_filter_col):
                combo.blockSignals(True)
                combo.clear()
                combo.blockSignals(False)
            if self.reference_path.text().strip() == reference:
                self._on_reference_changed()   # same sheet: re-read it so the columns re-seed
            else:
                self.reference_path.setText(reference)   # textChanged re-reads the new sheet

    def _apply_preset(self) -> None:
        """Re-seed every preset-driven default (the active preset may have changed on Setup).

        Signals are blocked while the values are written and one refresh is run at the end, so a
        preset that sets several fields does not trigger a re-plan (and a re-read of every source
        header) once per field.
        """
        widgets = (
            self.label_start, self.label_end, self.row_start, self.row_end,
            self.use_row, self.units, self.use_grid, self.grid_columns, self.grid_padding,
        )
        for widget in widgets:
            widget.blockSignals(True)
        self.label_start.setText(CONFIG.label_start)
        self.label_end.setText(CONFIG.label_end)
        self.row_start.setText(CONFIG.label_row_start)
        self.row_end.setText(CONFIG.label_row_end)
        self.use_row.setChecked(CONFIG.use_row)
        # A preset that names a unit overrides detection; a blank one hands the decision back to
        # "Detect automatically". Either way the value is set, never left over from the previous
        # preset — a stale x1000 would silently corrupt every metric.
        self.units.setCurrentText(
            unit_name(CONFIG.source_units) if CONFIG.source_units.strip() else _AUTO_UNIT
        )
        self.use_grid.setChecked(CONFIG.grid_enabled)
        self.grid_columns.setValue(max(CONFIG.grid_columns, 1))
        self.grid_padding.setValue(CONFIG.grid_padding)
        for widget in widgets:
            widget.blockSignals(False)
        # Apply the dependent state the blocked signals would have handled.
        self._apply_identity_wording(self.use_row.isChecked())
        self.row_box.setVisible(self.use_row.isChecked())
        self.grid_box.setEnabled(self.use_grid.isChecked())
        self._on_folder_changed()

    def _seed_reference(self, project) -> None:
        """Restore the pinned reference sheet + key column, loading it into the session."""
        pinned_key = project.pin("reference_key")
        if pinned_key and self.reference_key.findText(pinned_key) < 0:
            # The combo is only populated once the file loads, so stash the pin as the current
            # text; _on_reference_changed preserves it when the real headers arrive.
            self.reference_key.blockSignals(True)
            self.reference_key.addItem(pinned_key)
            self.reference_key.setCurrentText(pinned_key)
            self.reference_key.blockSignals(False)
        self._seed_field(
            self.reference_path, project.pin("reference") or CONFIG.input_path("reference_file")
        )
        if self._session.external is not None:
            self._refresh_reference_match()

    def reset_for_project(self) -> None:
        """Clear the previous project's inputs so a new one starts clean.

        Called by the shell when a project is created/opened, before ``on_enter`` re-seeds the
        fields from the now-active project's pins. ``on_enter`` only fills a field when the project
        pins a (non-empty) value, so without this an opened project with no pin would keep the
        prior project's paths, loaded masks and reference sheet.

        The fields' ``textChanged`` handlers pin into the active project (already the new one by
        now), so block their signals while clearing — otherwise we'd overwrite the new project's
        pins with empty strings before ``on_enter`` restores them.
        """
        for field in (self.cloud_path, self.mask_path, self.folder_path, self.reference_path):
            field.blockSignals(True)
            field.clear()
            field.blockSignals(False)
        for combo in (self.reference_key, self.reference_row,
                      self.reference_filter_col, self.reference_filter_value):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(_NO_KEY)
            combo.blockSignals(False)
        self._clear_reference()
        self._masks = None
        self._output_dir_value = None
        self._point_cache.clear()
        self.folder_status.setText("No folder chosen.")
        self.mask_status.setText("No masks loaded.")
        self.plot_map.clear()
        self.next_button.setEnabled(False)

    # ------------------------------------------------------------------ #
    # Readouts                                                           #
    # ------------------------------------------------------------------ #
    def _plot_files(self) -> list[Path]:
        """The imported plot files (originals only) in the project's plots/ folder."""
        project = self._session.project
        if project is None:
            return []
        return [
            p for p in sorted(project.plots_dir.glob("*"))
            if p.suffix.lower() in LAS_SUFFIXES and aug_number_from_name(p.name) is None
        ]

    def _plot_count(self) -> int:
        """How many plot files already exist in the project's plots/ folder."""
        return len(self._plot_files())

    def _refresh_indicator(self, n_plots: int | None = None) -> None:
        """Show "N plots loaded" plus the point totals, or what is still missing."""
        if n_plots is None:
            n_plots = self._plot_count()
        if n_plots:
            set_status_indicator(self.indicator, True, f"{n_plots} plots loaded")
        elif self._mode() == _MODE_MASKS and self._masks:
            set_status_indicator(
                self.indicator, False,
                f"{len(self._masks)} mask(s) loaded — clip to export plots.",
            )
        elif self._mode() == _MODE_MASKS:
            set_status_indicator(self.indicator, False, "Load a cloud and a .gpkg of masks.")
        else:
            set_status_indicator(self.indicator, False, "Choose a folder of plot files.")
        self._refresh_points_label(n_plots)

    def _refresh_points_label(self, n_plots: int) -> None:
        """Total / average point counts across the project's plots (header reads only)."""
        if not n_plots:
            self.points_label.setText("")
            return
        total = 0
        counted = 0
        for path in self._plot_files():
            total += self._cached_point_count(path)
            counted += 1
        if not counted or not total:
            self.points_label.setText("")
            return
        self.points_label.setText(
            f"— {total:,} points total, {round(total / counted):,} per plot on average"
        )

    def _cached_point_count(self, path: Path) -> int:
        """``path``'s point count, cached against its modification time (headers only, but many)."""
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return 0
        hit = self._point_cache.get(path)
        if hit is not None and hit[0] == mtime:
            return hit[1]
        count = las_io.point_count(path)
        self._point_cache[path] = (mtime, count)
        return count

    def _refresh_map(self) -> None:
        """Redraw the 2-D plot-layout view from the project's plots/ folder."""
        project = self._session.project
        self.plot_map.refresh(file_map_from_folder(project.plots_dir if project else None))

    @staticmethod
    def _seed_field(line_edit, value: str) -> None:
        if value and line_edit.text().strip() != value:
            line_edit.setText(value)

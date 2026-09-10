"""Results sub-tab: compare any number of saved models side by side.

Unlike the other sub-tabs (which view the single model the host has selected), this one keeps its own
collapsible **project → models** tree, so several models can be compared at once. The active project
is one group; **Import project…** adds *other* projects as further groups (session-only — dropped when
a different project is opened), since a bundle is fully self-contained. For the chosen metric it draws
four bar charts — one per metric column shared with Split Consistency: **held-out**, **training**,
**delta (held-out − training)** and **overall** — with one bar per included model.

Each project (parent) row has two checkboxes: a tri-state *include* box (column 0) that ticks/unticks
all of its models, and an *Avg* box (column 1) that collapses the project's **ticked** models into a
single averaged bar per panel (the mean of the per-model values). Children stay tickable when a project
is averaged — their ticks pick exactly which models feed the mean. So the active project's models can
stay individual while an imported project shows as one bar, or both can be averaged, etc.

Each bar reads its model's precomputed metric cube under the host's two "Aug in … metrics" toggles
(passed in as a :class:`~gui.results.variants.VariantSelection`), so no recomputation happens here
either. A model contributes either its *best split* (its lowest held-out rRMSE split,
``history.best_split`` — one coherent split across all four panels) or the *average across splits*,
chosen by a toggle. Models trained on different targets may be compared (rRMSE/R/R² are scale-free) but
a warning is shown, since MAPE and raw deltas need not be comparable across targets.

Every bar can be renamed by double-click — a model row, or a project row (for its averaged bar).
Local-model names persist in ``project.json`` (as before); imported-model and averaged-bar names are
session-only.

Bundles are loaded lazily — only when a model is first ticked — and only their histories are needed
(no data source, no prediction).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from common.session import Session
from ml import bundle_info, list_bundles, load_bundle
from ml.trainer import variant_metrics

from ..export import ChartValues, attach_export_menu
from .variants import VariantSelection

# (key, label, unit-suffix) for the four standard metrics — same surface as split consistency.
_METRICS = (
    ("rrmse", "rRMSE", "%"),
    ("r2", "R²", ""),
    ("r", "R", ""),
    ("mape", "MAPE", "%"),
)

# The four bar-chart panels (kind → title), one per metric column.
_KINDS = (
    ("held_out", "Held-out"),
    ("train", "Training"),
    ("delta", "Delta (held-out − training)"),
    ("overall", "Overall"),
)

# Distinct colours cycled across the included models (matplotlib tab10).
_COLOURS = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)

_BEST, _AVERAGE = "best", "average"

# Combine modes for a bar that spans several models (project-average bars and custom bars). The
# reducer is applied NaN-safely to the members' already-aggregated per-model values.
_COMBINE_AVG, _COMBINE_MIN, _COMBINE_MAX, _COMBINE_SUM = "avg", "min", "max", "sum"
_COMBINE_LABELS = {
    _COMBINE_AVG: "Average",
    _COMBINE_MIN: "Min",
    _COMBINE_MAX: "Max",
    _COMBINE_SUM: "Sum",
}
_COMBINE_FUNCS = {
    _COMBINE_AVG: np.nanmean,
    _COMBINE_MIN: np.nanmin,
    _COMBINE_MAX: np.nanmax,
    _COMBINE_SUM: np.nansum,
}

# Sort cycle for the "Sort bars" button: original tick order → descending → ascending → back.
_SORT_NONE, _SORT_DESC, _SORT_ASC = "none", "desc", "asc"
_SORT_CYCLE = {_SORT_NONE: _SORT_DESC, _SORT_DESC: _SORT_ASC, _SORT_ASC: _SORT_NONE}
_SORT_LABELS = {
    _SORT_NONE: "Sort bars (high → low)",
    _SORT_DESC: "Sort bars (low → high)",
    _SORT_ASC: "Sort bars (reset)",
}

# Short, capitalised chart abbreviations per model_key, used by the "rename to abbreviation"
# button. Anything not listed falls back to title-casing its key (e.g. "hist_gbt" -> "Hist Gbt").
_ABBREVIATIONS = {
    "pls": "PLS",
    "elastic_net": "Elastic",
    "lasso": "Lasso",
    "ridge": "Ridge",
    "knn": "kNN",
    "svr": "SVR",
    "random_forest": "RF",
    "hist_gbt": "HistGBT",
    "gpr": "GPR",
}


def _abbreviation(model_key: str) -> str:
    """The capitalised chart abbreviation for ``model_key`` (title-cased fallback if unmapped)."""
    return _ABBREVIATIONS.get(model_key, model_key.replace("_", " ").title())


# The tree has one column for the include checkbox + label and a second narrow column carrying the
# per-project "average into one bar" checkbox (only ever set on project rows).
_COL_MAIN = 0
_COL_AVG = 1

# Per-item data roles on the model tree. Project (parent) rows carry _ROOT_ROLE + _IS_PROJECT_ROLE;
# model (child) rows carry the bundle metadata roles.
_PATH_ROLE = Qt.UserRole          # str: absolute bundle path (history cache key) — model rows
_TARGET_ROLE = Qt.UserRole + 1    # str: target column (cross-target warning) — model rows
_NAME_ROLE = Qt.UserRole + 2      # str: bundle filename (custom-name lookup key) — model rows
_AUTO_ROLE = Qt.UserRole + 3      # str: the bundle's auto-generated label — model rows
_KEY_ROLE = Qt.UserRole + 4       # str: the bundle's model_key (abbreviation lookup) — model rows
_ROOT_ROLE = Qt.UserRole + 5      # str: source-project root — both row kinds (grouping/averaging)
_IS_PROJECT_ROLE = Qt.UserRole + 6  # bool: True on a project (parent) row
_IS_CUSTOM_ROLE = Qt.UserRole + 7   # bool: True on a custom-bar (top-level) row
_CUSTOM_ID_ROLE = Qt.UserRole + 8   # str: the _CustomBar.id — custom rows and their member children

# Session-name key prefix for an averaged-project bar (distinguishes it from a bundle path key).
_AVG_KEY_PREFIX = "avg::"

# Id prefix for a custom bar (used as its _Bar.key for colour/naming).
_CUSTOM_KEY_PREFIX = "custom::"


@dataclass
class _Bar:
    """One bar in every panel: a single model, a project's ticked models combined, or a custom bar."""

    key: str                  # stable id for naming/colour (bundle path, "avg::<root>", or "custom::N")
    label: str                # resolved display name (custom name or default)
    targets: tuple[str, ...]  # contributing target columns (for the cross-target warning)
    histories: tuple          # one TrainHistory (individual) or several (combined over its members)
    is_average: bool          # True when ``histories`` must be reduced to one value
    combine: str = _COMBINE_AVG  # reducer for a multi-history bar (project averages stay "avg")


@dataclass
class _CustomBar:
    """A user-defined bar combining hand-picked models across projects (session-only)."""

    id: str                 # stable unique id (e.g. "custom::1") for tree/name/colour keying
    name: str               # display name (default "Custom bar N", user-renamable)
    members: list[str]      # normalised bundle paths of contributing models
    mode: str               # combine mode: one of the _COMBINE_* constants
    checked: bool = True    # whether this bar's row is ticked (drawn)


class _PanelExport:
    """Export provider for one comparison panel (held-out / training / delta / overall)."""

    def __init__(self, owner: "ModelComparison", kind: str) -> None:
        self._owner = owner
        self._kind = kind

    def draw_into(self, ax) -> None:
        # Export on the same shared y-range as the live 2×2 grid, so exported panels
        # compare on one scale (the largest panel's) rather than auto-scaling individually.
        ylim = self._owner.shared_ylim()
        self._owner.draw_kind_into(ax, self._kind, ylim=ylim)

    def export_title(self) -> str:
        return f"model_comparison_{self._kind}"

    def export_values(self) -> ChartValues:
        return self._owner.chart_values(self._kind)


class ModelComparison(QWidget):
    """The model-comparison sub-tab: bar charts of any chosen models across the four metric columns."""

    def __init__(self, session: Session) -> None:
        super().__init__()
        self._session = session
        self._selection = VariantSelection.legacy()
        self._histories: dict[str, object] = {}  # path str -> TrainHistory (lazy cache)
        self._sort_mode = _SORT_NONE  # bar ordering within each panel (cycled by the sort button)
        self._imported_roots: list[str] = []     # other projects added via "Import project…" (session-only)
        self._averaged_projects: set[str] = set()  # project roots collapsed to one averaged bar (session-only)
        self._session_names: dict[str, str] = {}   # session-only names: bundle path / "avg::<root>" -> name
        self._custom_bars: list[_CustomBar] = []   # user-defined cross-project bars (session-only)
        self._info_cache: dict[str, object] = {}   # path str -> bundle_info result (cheap metadata cache)

        root = QVBoxLayout(self)

        controls = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(self._bold("Models to compare:"))
        # A collapsible tree: each imported/active project is a parent row whose models are children.
        # Column 0 carries the include checkbox + label (parent rows are tri-state over their models);
        # column 1 carries the parent-only "Avg" checkbox (collapse this project's ticked models into
        # one averaged bar). Children stay tickable when averaged — their ticks pick what feeds the mean.
        self.model_tree = QTreeWidget()
        self.model_tree.setColumnCount(2)
        self.model_tree.setHeaderLabels(["Models to compare", "Avg"])
        self.model_tree.header().setStretchLastSection(False)
        self.model_tree.header().setSectionResizeMode(_COL_MAIN, QHeaderView.Stretch)
        self.model_tree.header().setSectionResizeMode(_COL_AVG, QHeaderView.Fixed)
        self.model_tree.setColumnWidth(_COL_AVG, 92)
        self.model_tree.setToolTip(
            "Tick models to include them. Tick a project's 'Avg' box to show its ticked models as a\n"
            "single averaged bar. Double-click a model — or a project (for its averaged bar) — to rename it."
        )
        self.model_tree.itemChanged.connect(self._on_tree_item_changed)
        self.model_tree.itemDoubleClicked.connect(self._on_tree_double_clicked)
        self.model_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.model_tree.customContextMenuRequested.connect(self._on_tree_context_menu)
        left.addWidget(self.model_tree, 1)
        left.addWidget(QLabel("Double-click a model (or a project, for its averaged bar) to rename it."))
        controls.addLayout(left, 1)

        right = QVBoxLayout()
        right.addWidget(self._bold("Metric:"))
        self.metric_combo = QComboBox()
        for key, label, _ in _METRICS:
            self.metric_combo.addItem(label, key)
        self.metric_combo.currentIndexChanged.connect(self._redraw)
        right.addWidget(self.metric_combo)
        right.addWidget(self._bold("Split aggregation:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Best split (lowest held-out rRMSE)", _BEST)
        self.mode_combo.addItem("Average across splits", _AVERAGE)
        self.mode_combo.setCurrentIndex(self.mode_combo.findData(_AVERAGE))
        self.mode_combo.setToolTip(
            "Best: each model's single best split, used for every panel.\n"
            "Average: the mean of each model's metric across all its splits."
        )
        self.mode_combo.currentIndexChanged.connect(self._redraw)
        right.addWidget(self.mode_combo)
        self.abbreviate_button = QPushButton("Rename all to abbreviations")
        self.abbreviate_button.setToolTip(
            "Set every model's chart name to its capitalised abbreviation\n"
            "(e.g. PLS, Elastic, RF, kNN). Overwrites any custom names."
        )
        self.abbreviate_button.clicked.connect(self._rename_all_to_abbreviations)
        right.addWidget(self.abbreviate_button)
        self.import_button = QPushButton("Import project…")
        self.import_button.setToolTip(
            "Add every saved model from another project to the comparison, as a new group.\n"
            "Imported projects (and any custom names) last for this session only."
        )
        self.import_button.clicked.connect(self._import_project)
        right.addWidget(self.import_button)
        self.custom_bar_button = QPushButton("New custom bar from ticked models")
        self.custom_bar_button.setToolTip(
            "Combine the currently-ticked models — from any project — into one custom bar.\n"
            "Custom bars average their members by default (change per bar) and last this session only."
        )
        self.custom_bar_button.clicked.connect(self._new_custom_bar_from_ticked)
        right.addWidget(self.custom_bar_button)
        self.sort_button = QPushButton(_SORT_LABELS[_SORT_NONE])
        self.sort_button.setToolTip(
            "Sort each panel's bars by height. Click cycles:\n"
            "high → low, then low → high, then back to the original order."
        )
        self.sort_button.clicked.connect(self._cycle_sort)
        right.addWidget(self.sort_button)
        right.addStretch(1)
        controls.addLayout(right)
        root.addLayout(controls)

        self.warning_label = QLabel("")
        self.warning_label.setStyleSheet("color: #b8860b;")  # amber
        self.warning_label.setWordWrap(True)
        self.warning_label.setVisible(False)
        root.addWidget(self.warning_label)

        # Four bar panels in a 2×2 grid, each independently exportable.
        grid = QGridLayout()
        self._axes: dict[str, object] = {}
        self._canvases: dict[str, FigureCanvas] = {}
        for i, (kind, _title) in enumerate(_KINDS):
            figure = Figure(figsize=(4, 2.6), tight_layout=True)
            canvas = FigureCanvas(figure)
            ax = figure.add_subplot(111)
            grid.addWidget(canvas, i // 2, i % 2)
            attach_export_menu(canvas, _PanelExport(self, kind))
            self._axes[kind] = ax
            self._canvases[kind] = canvas
        root.addLayout(grid, 1)

        self._draw_empty()

    @staticmethod
    def _bold(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("font-weight: bold;")
        return label

    @staticmethod
    def _norm(path) -> str:
        """A bundle path in one canonical form, so local/imported identity comparisons are reliable."""
        return str(Path(path).resolve())

    @classmethod
    def _project_root(cls, path: str) -> str:
        """The source-project root of a bundle: ``<root>/model/<file>.joblib`` -> ``<root>``."""
        return str(Path(cls._norm(path)).parent.parent)

    @staticmethod
    def _project_name(root: str) -> str:
        """A project root's display name (its folder name, falling back to the full path)."""
        return Path(root).name or root

    # ------------------------------------------------------------------ #
    # Import                                                              #
    # ------------------------------------------------------------------ #
    def reset_imports(self) -> None:
        """Drop all session-only state (imports, averaging, custom bars, names). Called on project switch."""
        self._imported_roots.clear()
        self._averaged_projects.clear()
        self._session_names.clear()
        self._custom_bars.clear()
        self._info_cache.clear()

    def _import_project(self) -> None:
        """Add every saved model from another project to the comparison, as a new group (session-only)."""
        start = str(Path(self._session.project.root).parent) if self._session.project else ""
        folder = QFileDialog.getExistingDirectory(self, "Import project folder", start)
        if not folder:
            return
        root = self._norm(folder)
        # Accept either the project root or its model/ folder; resolve to the root either way.
        if Path(root).name == "model":
            root = str(Path(root).parent)
        if not list_bundles(Path(root) / "model"):
            QMessageBox.warning(
                self, "No models",
                "That folder has no saved model bundles in its model/ sub-folder.",
            )
            return
        active = self._active_root()
        if root == active or root in self._imported_roots:
            self.reload_models()  # already present — just refresh
            return
        self._imported_roots.append(root)
        self.reload_models(tick_root=root)

    def _active_root(self) -> str | None:
        """The active project's resolved root, or ``None`` when no project is open."""
        return self._norm(self._session.project.root) if self._session.project else None

    def _model_dir_for(self, root: str) -> Path:
        """The model/ folder of a project root."""
        return Path(root) / "model"

    # ------------------------------------------------------------------ #
    # Bundle metadata (cheap, cached — no full model load)               #
    # ------------------------------------------------------------------ #
    def _info_for(self, path: str):
        """Cached ``bundle_info`` for ``path`` (metadata only), or ``None`` if it no longer loads."""
        path = self._norm(path)
        if path not in self._info_cache:
            try:
                self._info_cache[path] = bundle_info(path)
            except Exception:  # noqa: BLE001 - a vanished/broken bundle just drops out
                self._info_cache[path] = None
        return self._info_cache[path]

    def _target_for(self, path: str) -> str:
        """The target column a bundle was trained on (``""`` if unknown)."""
        info = self._info_for(path)
        return str(getattr(info, "target_column", "") or "") if info else ""

    def _label_for(self, path: str) -> str:
        """A member model's provenance label for a custom-bar child row: ``label — project``."""
        info = self._info_for(path)
        label = str(getattr(info, "label", "") or Path(path).stem) if info else Path(path).stem
        return f"{label} — {self._project_name(self._project_root(path))}"

    def _custom_bar(self, cb_id: str) -> "_CustomBar | None":
        """The custom bar with ``cb_id``, or ``None``."""
        return next((cb for cb in self._custom_bars if cb.id == cb_id), None)

    # ------------------------------------------------------------------ #
    # Model tree                                                          #
    # ------------------------------------------------------------------ #
    def reload_models(self, tick_root: str | None = None) -> None:
        """Rebuild the project→models tree from the active project plus any imported projects.

        Per-model ticks and per-project Avg toggles are preserved across the rebuild. ``tick_root``
        (a project root) has all of its models checked — used when a project is freshly imported.
        """
        checked = set(self._checked_paths())
        self.model_tree.blockSignals(True)
        self.model_tree.clear()

        roots: list[str] = []
        active = self._active_root()
        if active is not None:
            roots.append(active)
        kept_imports = []
        for r in self._imported_roots:
            if r == active or r in roots:
                continue
            if list_bundles(self._model_dir_for(r)):  # prune imports whose models vanished
                roots.append(r)
                kept_imports.append(r)
        self._imported_roots = kept_imports
        self._averaged_projects &= set(roots)  # forget Avg toggles for projects no longer present

        live: set[str] = set()
        for root in roots:
            infos = list_bundles(self._model_dir_for(root))
            if not infos and root != active:
                continue
            parent = QTreeWidgetItem(self.model_tree)
            parent.setData(_COL_MAIN, _IS_PROJECT_ROLE, True)
            parent.setData(_COL_MAIN, _ROOT_ROLE, root)
            parent.setText(_COL_MAIN, self._project_name(root))
            # Column 0: tri-state include checkbox (auto-rolls up its children). The separate "Average
            # into one bar" toggle is a real QCheckBox widget in column 1 (clearer + easier to hit than
            # a bare item checkbox), wired straight to _set_averaged for that project root.
            parent.setFlags(parent.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
            parent.setCheckState(_COL_MAIN, Qt.Unchecked)
            avg_box = QCheckBox("Avg")
            avg_box.setToolTip("Average this project's ticked models into a single bar.")
            avg_box.setChecked(root in self._averaged_projects)
            avg_box.toggled.connect(lambda on, r=root: self._set_averaged(r, on))
            self.model_tree.setItemWidget(parent, _COL_AVG, avg_box)
            for info in infos:
                path = self._norm(info.path)
                live.add(path)
                child = QTreeWidgetItem(parent)
                child.setData(_COL_MAIN, _IS_PROJECT_ROLE, False)
                child.setData(_COL_MAIN, _PATH_ROLE, path)
                child.setData(_COL_MAIN, _TARGET_ROLE, info.target_column)
                child.setData(_COL_MAIN, _NAME_ROLE, info.path.name)
                child.setData(_COL_MAIN, _AUTO_ROLE, info.label)
                child.setData(_COL_MAIN, _KEY_ROLE, info.model_key)
                child.setData(_COL_MAIN, _ROOT_ROLE, root)
                child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
                want = (tick_root is not None and root == tick_root) or path in checked
                child.setCheckState(_COL_MAIN, Qt.Checked if want else Qt.Unchecked)
                self._apply_item_text(child)
            self.model_tree.expandItem(parent)

        self._build_custom_rows()

        self.model_tree.blockSignals(False)
        # Drop cached histories for bundles that no longer exist.
        self._histories = {p: h for p, h in self._histories.items() if p in live}
        self._redraw()

    def _build_custom_rows(self) -> None:
        """Append one top-level row per custom bar (with a member child each) after the project groups.

        Called inside ``reload_models`` while signals are blocked. Members whose bundle no longer loads
        are pruned; a bar left with no members is dropped entirely.
        """
        amber = QBrush(QColor("#b8860b"))
        kept: list[_CustomBar] = []
        for cb in self._custom_bars:
            members = [p for p in cb.members if self._info_for(p) is not None]
            if not members:
                continue  # every member vanished — drop the bar
            cb.members = members
            kept.append(cb)

            parent = QTreeWidgetItem(self.model_tree)
            parent.setData(_COL_MAIN, _IS_CUSTOM_ROLE, True)
            parent.setData(_COL_MAIN, _CUSTOM_ID_ROLE, cb.id)
            parent.setText(_COL_MAIN, f"* {cb.name}")
            parent.setForeground(_COL_MAIN, amber)
            parent.setToolTip(_COL_MAIN, "Custom bar (session-only). Double-click to rename; "
                              "right-click to delete or add ticked models.")
            parent.setFlags(parent.flags() | Qt.ItemIsUserCheckable)
            parent.setCheckState(_COL_MAIN, Qt.Checked if cb.checked else Qt.Unchecked)

            combo = QComboBox()
            for mode, label in _COMBINE_LABELS.items():
                combo.addItem(label, mode)
            combo.setCurrentIndex(combo.findData(cb.mode))
            combo.setToolTip("How to combine this bar's members into one value.")
            combo.currentIndexChanged.connect(
                lambda _i, c=combo, b=cb: self._set_custom_mode(b, c.currentData())
            )
            self.model_tree.setItemWidget(parent, _COL_AVG, combo)

            for path in members:
                child = QTreeWidgetItem(parent)
                child.setData(_COL_MAIN, _IS_CUSTOM_ROLE, False)
                child.setData(_COL_MAIN, _CUSTOM_ID_ROLE, cb.id)
                child.setData(_COL_MAIN, _PATH_ROLE, path)
                child.setText(_COL_MAIN, self._label_for(path))
                child.setForeground(_COL_MAIN, amber)
                child.setToolTip(_COL_MAIN, "Member of a custom bar. Right-click to remove.")
            self.model_tree.expandItem(parent)
        self._custom_bars = kept

    def _set_custom_mode(self, cb: "_CustomBar", mode: str) -> None:
        """A custom bar's combine picker changed."""
        cb.mode = mode
        self._redraw()

    def _iter_model_items(self):
        """Yield every project-model (child) item in the tree, in display order.

        Custom-bar top-level rows and their member children are skipped, so ``_checked_paths`` and the
        abbreviation/rename passes only ever see real project models.
        """
        for p in range(self.model_tree.topLevelItemCount()):
            parent = self.model_tree.topLevelItem(p)
            if bool(parent.data(_COL_MAIN, _IS_CUSTOM_ROLE)):
                continue
            for c in range(parent.childCount()):
                yield parent.child(c)

    def _checked_paths(self) -> set[str]:
        """The bundle paths of every currently-ticked model."""
        return {
            str(item.data(_COL_MAIN, _PATH_ROLE))
            for item in self._iter_model_items()
            if item.checkState(_COL_MAIN) == Qt.Checked
        }

    def _on_tree_item_changed(self, item: QTreeWidgetItem, _column: int) -> None:
        """An include checkbox toggled — redraw. Custom-bar rows also write back their checked state."""
        if bool(item.data(_COL_MAIN, _IS_CUSTOM_ROLE)):
            cb = self._custom_bar(str(item.data(_COL_MAIN, _CUSTOM_ID_ROLE)))
            if cb is not None:
                cb.checked = item.checkState(_COL_MAIN) == Qt.Checked
        self._redraw()

    def _set_averaged(self, root: str, on: bool) -> None:
        """The project's 'Avg' checkbox toggled: collapse/expand its bar and redraw."""
        if on:
            self._averaged_projects.add(root)
        else:
            self._averaged_projects.discard(root)
        self._redraw()

    def _on_tree_double_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        """Double-click renames: a model's bar, a project's averaged bar, or a custom bar."""
        if bool(item.data(_COL_MAIN, _IS_CUSTOM_ROLE)):
            self._rename_custom_bar(item)  # a custom-bar top-level row
        elif item.data(_COL_MAIN, _CUSTOM_ID_ROLE) is not None:
            pass  # a custom-bar member child — no rename (edit via right-click "Remove")
        elif bool(item.data(_COL_MAIN, _IS_PROJECT_ROLE)):
            self._rename_average(item)
        else:
            self._rename_model(item)

    # ------------------------------------------------------------------ #
    # Custom bars                                                        #
    # ------------------------------------------------------------------ #
    def _new_custom_bar_from_ticked(self) -> None:
        """Button/menu slot: create a custom bar from the currently-ticked models.

        A plain slot (no positional args) so ``QPushButton.clicked``'s ``checked`` bool can't leak in
        as a member list.
        """
        self._new_custom_bar(sorted(self._checked_paths()))

    def _new_custom_bar(self, members: list[str]) -> None:
        """Create a custom bar from the given bundle paths (warns and no-ops if empty)."""
        members = sorted(members)
        if not members:
            QMessageBox.information(
                self, "No models ticked",
                "Tick the models you want to combine into a bar, then click again.",
            )
            return
        n = len(self._custom_bars) + 1
        self._custom_bars.append(_CustomBar(
            id=f"{_CUSTOM_KEY_PREFIX}{n}", name=f"Custom bar {n}",
            members=members, mode=_COMBINE_AVG,
        ))
        self.reload_models()

    def _add_to_custom_bar(self, cb: "_CustomBar", paths: list[str]) -> None:
        """Union ``paths`` into ``cb``'s members (dedup, preserve order) and rebuild."""
        for p in paths:
            if p not in cb.members:
                cb.members.append(p)
        self.reload_models()

    def _rename_custom_bar(self, item: QTreeWidgetItem) -> None:
        """Prompt for a custom bar's name (session-only)."""
        cb = self._custom_bar(str(item.data(_COL_MAIN, _CUSTOM_ID_ROLE)))
        if cb is None:
            return
        text, ok = QInputDialog.getText(self, "Rename custom bar", "Name for the charts:", text=cb.name)
        if not ok or not text.strip():
            return
        cb.name = text.strip()
        self.model_tree.blockSignals(True)
        item.setText(_COL_MAIN, f"* {cb.name}")
        self.model_tree.blockSignals(False)
        self._redraw()

    def _delete_custom_bar(self, cb_id: str) -> None:
        """Remove a custom bar entirely."""
        self._custom_bars = [cb for cb in self._custom_bars if cb.id != cb_id]
        self.reload_models()

    def _remove_custom_member(self, cb_id: str, path: str) -> None:
        """Drop one member from a custom bar; if it empties, remove the bar."""
        cb = self._custom_bar(cb_id)
        if cb is None:
            return
        cb.members = [p for p in cb.members if p != path]
        if not cb.members:
            self._custom_bars = [c for c in self._custom_bars if c.id != cb_id]
        self.reload_models()

    def _on_tree_context_menu(self, pos) -> None:
        """Right-click menu: create/grow custom bars, or edit an existing custom bar / member."""
        item = self.model_tree.itemAt(pos)
        menu = QMenu(self.model_tree)
        ticked = sorted(self._checked_paths())

        if item is not None and bool(item.data(_COL_MAIN, _IS_CUSTOM_ROLE)):
            cb_id = str(item.data(_COL_MAIN, _CUSTOM_ID_ROLE))
            menu.addAction("Rename custom bar…", lambda: self._rename_custom_bar(item))
            menu.addAction("Delete custom bar", lambda: self._delete_custom_bar(cb_id))
        elif item is not None and item.data(_COL_MAIN, _CUSTOM_ID_ROLE) is not None:
            cb_id = str(item.data(_COL_MAIN, _CUSTOM_ID_ROLE))
            path = str(item.data(_COL_MAIN, _PATH_ROLE))
            menu.addAction("Remove from custom bar", lambda: self._remove_custom_member(cb_id, path))
        else:
            new = menu.addAction("New custom bar from ticked models",
                                 self._new_custom_bar_from_ticked)
            new.setEnabled(bool(ticked))
            if self._custom_bars:
                sub = menu.addMenu("Add ticked models to custom bar")
                sub.setEnabled(bool(ticked))
                for cb in self._custom_bars:
                    sub.addAction(cb.name, lambda _=False, b=cb: self._add_to_custom_bar(b, ticked))

        if not menu.isEmpty():
            menu.exec(self.model_tree.viewport().mapToGlobal(pos))

    def _cycle_sort(self) -> None:
        """Advance the bar-ordering cycle (none → desc → asc → none) and redraw."""
        self._sort_mode = _SORT_CYCLE[self._sort_mode]
        self.sort_button.setText(_SORT_LABELS[self._sort_mode])
        self._redraw()

    # ------------------------------------------------------------------ #
    # Custom names                                                       #
    # ------------------------------------------------------------------ #
    def _is_local(self, path: str) -> bool:
        """True if ``path`` is one of the active project's own bundles (vs an imported one)."""
        project = self._session.project
        return project is not None and self._project_root(path) == self._norm(project.root)

    def _custom_name(self, item: QTreeWidgetItem) -> str:
        """A model's user-given chart name, or ``""`` if none.

        Local models read the persisted ``project.model_names`` (keyed by filename, as before);
        imported models read the session-only override keyed by their bundle path.
        """
        path = str(item.data(_COL_MAIN, _PATH_ROLE))
        if self._is_local(path):
            project = self._session.project
            return project.model_names.get(str(item.data(_COL_MAIN, _NAME_ROLE)), "") if project else ""
        return self._session_names.get(path, "")

    def _set_custom_name(self, item: QTreeWidgetItem, text: str) -> None:
        """Persist a model's custom name to the right place (project json for local, session for imported)."""
        path = str(item.data(_COL_MAIN, _PATH_ROLE))
        if self._is_local(path):
            if self._session.project is not None:
                self._session.project.set_model_name(str(item.data(_COL_MAIN, _NAME_ROLE)), text)
        elif text.strip():
            self._session_names[path] = text.strip()
        else:
            self._session_names.pop(path, None)

    def _apply_item_text(self, item: QTreeWidgetItem) -> None:
        """Render a model row: ``custom name — auto label`` when renamed, else just the auto label."""
        auto = str(item.data(_COL_MAIN, _AUTO_ROLE))
        custom = self._custom_name(item)
        item.setText(_COL_MAIN, f"{custom} — {auto}" if custom else auto)

    def _rename_model(self, item: QTreeWidgetItem) -> None:
        """Prompt for a custom chart name for the double-clicked model and persist it."""
        current = self._custom_name(item)
        text, ok = QInputDialog.getText(
            self,
            "Rename model",
            "Custom name for the charts (leave blank to use the default):",
            text=current,
        )
        if not ok:
            return
        self._set_custom_name(item, text)
        self.model_tree.blockSignals(True)
        self._apply_item_text(item)
        self.model_tree.blockSignals(False)
        self._redraw()

    def _rename_all_to_abbreviations(self) -> None:
        """Set every model's chart name to its capitalised abbreviation (e.g. PLS, Elastic, RF)."""
        self.model_tree.blockSignals(True)
        for item in self._iter_model_items():
            self._set_custom_name(item, _abbreviation(str(item.data(_COL_MAIN, _KEY_ROLE))))
            self._apply_item_text(item)
        self.model_tree.blockSignals(False)
        self._redraw()

    # ---- averaged-bar names (session-only) ---------------------------- #
    def _avg_default_name(self, root: str) -> str:
        """The default chart name for a project's averaged bar."""
        return f"{self._project_name(root)} (avg)"

    def _avg_name(self, root: str) -> str:
        """The averaged bar's chart name for ``root``: the session override, else the default."""
        return self._session_names.get(_AVG_KEY_PREFIX + root, "") or self._avg_default_name(root)

    def _rename_average(self, item: QTreeWidgetItem) -> None:
        """Prompt for a session-only name for a project's averaged bar (the project row was clicked)."""
        root = str(item.data(_COL_MAIN, _ROOT_ROLE))
        current = self._session_names.get(_AVG_KEY_PREFIX + root, "")
        text, ok = QInputDialog.getText(
            self,
            "Rename averaged bar",
            f"Custom name for {self._project_name(root)}'s averaged bar "
            "(leave blank to use the default):",
            text=current,
        )
        if not ok:
            return
        if text.strip():
            self._session_names[_AVG_KEY_PREFIX + root] = text.strip()
        else:
            self._session_names.pop(_AVG_KEY_PREFIX + root, None)
        self._redraw()

    def _history_for(self, path: str):
        """The cached :class:`TrainHistory` for ``path``, loading the bundle on first use."""
        if path not in self._histories:
            try:
                self._histories[path] = load_bundle(path).history
            except Exception:  # noqa: BLE001 - a bad bundle just drops out of the comparison
                self._histories[path] = None
        return self._histories[path]

    # ------------------------------------------------------------------ #
    # Data in                                                            #
    # ------------------------------------------------------------------ #
    def refresh(self, selection: VariantSelection | None = None) -> None:
        """Receive the augmented-data choice from the host and redraw the bars."""
        self._selection = selection or VariantSelection.legacy()
        self._redraw()

    # ------------------------------------------------------------------ #
    # Value extraction                                                   #
    # ------------------------------------------------------------------ #
    def _model_value(self, history, kind: str, key: str) -> float:
        """One bar height: ``key`` for ``kind``, over the chosen split aggregation."""
        splits = getattr(history, "splits", None) or []
        if not splits:
            return float("nan")
        sel = self._selection

        def cell(sm) -> float:
            if kind == "held_out":
                return variant_metrics(sm, "held_out", sel.held_out).get(key, float("nan"))
            if kind == "train":
                return variant_metrics(sm, "train", sel.train).get(key, float("nan"))
            if kind == "overall":
                return variant_metrics(sm, "overall", sel.overall).get(key, float("nan"))
            held = variant_metrics(sm, "held_out", sel.held_out).get(key, float("nan"))
            train = variant_metrics(sm, "train", sel.train).get(key, float("nan"))
            return float(held) - float(train)

        vals = np.asarray([cell(sm) for sm in splits], dtype=float)
        if self.mode_combo.currentData() == _BEST:
            return float(vals[self._best_index(history)])
        return float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")

    @staticmethod
    def _best_index(history) -> int:
        """Position in ``history.splits`` of the best split (lowest held-out rRMSE)."""
        splits = history.splits
        want = getattr(history, "best_split", 0)
        for i, sm in enumerate(splits):
            if getattr(sm, "split", None) == want:
                return i
        return 0

    # ------------------------------------------------------------------ #
    # Drawing                                                            #
    # ------------------------------------------------------------------ #
    def _draw_empty(self) -> None:
        for kind, title in _KINDS:
            ax = self._axes[kind]
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title)
            ax.text(0.5, 0.5, "Tick models to compare.", ha="center", va="center",
                    color="gray", transform=ax.transAxes)
            self._canvases[kind].draw_idle()

    def _bars(self) -> list[_Bar]:
        """The ordered bars: ticked individual models, averaged projects, then ticked custom bars.

        Ticked models whose source project is in ``_averaged_projects`` are collapsed into a single
        :class:`_Bar` (their histories averaged later), positioned where that project first appears;
        every other ticked model is its own bar. Each ticked custom bar is appended after the project
        bars, combining its member histories under its own combine mode. Models that fail to load are
        skipped.
        """
        bars: list[_Bar] = []
        avg_index: dict[str, int] = {}  # project root -> position of its average bar in ``bars``
        for item in self._iter_model_items():
            if item.checkState(_COL_MAIN) != Qt.Checked:
                continue
            path = str(item.data(_COL_MAIN, _PATH_ROLE))
            history = self._history_for(path)
            if history is None:
                continue
            target = str(item.data(_COL_MAIN, _TARGET_ROLE) or "")
            root = str(item.data(_COL_MAIN, _ROOT_ROLE))
            if root in self._averaged_projects:
                if root not in avg_index:
                    avg_index[root] = len(bars)
                    bars.append(_Bar(
                        key=_AVG_KEY_PREFIX + root,
                        label=self._avg_name(root),
                        targets=(),
                        histories=(),
                        is_average=True,
                    ))
                bar = bars[avg_index[root]]
                bars[avg_index[root]] = _Bar(
                    key=bar.key, label=bar.label,
                    targets=bar.targets + (target,),
                    histories=bar.histories + (history,),
                    is_average=True,
                )
            else:
                bars.append(_Bar(
                    key=path,
                    label=self._chart_label(item),
                    targets=(target,),
                    histories=(history,),
                    is_average=False,
                ))

        for cb in self._custom_bars:
            if not cb.checked:
                continue
            hists: list[object] = []
            targets: list[str] = []
            for path in cb.members:
                history = self._history_for(path)
                if history is None:
                    continue
                hists.append(history)
                targets.append(self._target_for(path))
            if not hists:
                continue
            bars.append(_Bar(
                key=cb.id,
                label=cb.name,
                targets=tuple(targets),
                histories=tuple(hists),
                is_average=True,
                combine=cb.mode,
            ))
        return bars

    def _chart_label(self, item: QTreeWidgetItem) -> str:
        """The x-axis tick for a model: its custom name if set, else the leading ``[ML] model_key`` part."""
        custom = self._custom_name(item)
        if custom:
            return custom
        return str(item.data(_COL_MAIN, _AUTO_ROLE)).split(" — ", 1)[0]

    def _bar_value(self, bar: _Bar, kind: str, key: str) -> float:
        """One bar height: an individual model's value, or its members combined under ``bar.combine``.

        NaN-safe: combining only non-finite contributions returns NaN (rendered as ``n/a``).
        """
        if not bar.is_average:
            return self._model_value(bar.histories[0], kind, key)
        vals = np.asarray([self._model_value(h, kind, key) for h in bar.histories], dtype=float)
        if not np.isfinite(vals).any():
            return float("nan")
        reduce = _COMBINE_FUNCS.get(bar.combine, np.nanmean)
        return float(reduce(vals))

    def _redraw(self) -> None:
        bars = self._bars()
        # Warn when the comparison spans more than one target (scale-free metrics still compare).
        targets = {t for bar in bars for t in bar.targets if t}
        if len(targets) > 1:
            self.warning_label.setText(
                "Comparing models trained on different targets (" + ", ".join(sorted(targets))
                + "): rRMSE, R and R² are scale-free, but MAPE and raw deltas may not be comparable."
            )
            self.warning_label.setVisible(True)
        else:
            self.warning_label.setVisible(False)
        ylim = self.shared_ylim()
        for kind, _ in _KINDS:
            self.draw_kind_into(self._axes[kind], kind, ylim=ylim)
            self._canvases[kind].draw_idle()

    def shared_ylim(self) -> tuple[float, float] | None:
        """A common (low, high) y-range spanning all four panels, so the bars compare on one scale.

        Taken from the panel with the largest range (every panel's finite values feed one min/max),
        so the live grid and any exported panel all use that same scale. Returns ``None`` when there
        is nothing finite to plot (panels keep their auto-scale). Always includes 0 (the bar
        baseline) and pads the top a little so the tallest bar isn't clipped.
        """
        bars = self._bars()
        key = self.metric_combo.currentData()
        vals = np.asarray(
            [self._bar_value(bar, kind, key) for bar in bars for kind, _ in _KINDS],
            dtype=float,
        )
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None
        low = min(0.0, float(vals.min()))
        high = max(0.0, float(vals.max()))
        pad = (high - low) * 0.05 or 1.0
        return (low - (pad if low < 0 else 0.0), high + pad)

    def draw_kind_into(self, ax, kind: str, ylim: tuple[float, float] | None = None) -> None:
        """Draw the ``kind`` panel's bar chart into ``ax`` (live canvas or export figure).

        Pure render: reads the ticked models, metric picker and aggregation mode but mutates no
        widgets, so the export layer can call it on a fresh figure. ``ylim`` forces a shared
        y-range across the live 2×2 grid; exports pass ``None`` and keep per-panel auto-scaling.
        """
        key = self.metric_combo.currentData()
        label = next(lbl for k, lbl, _ in _METRICS if k == key)
        suffix = next(s for k, _, s in _METRICS if k == key)
        title = next(t for k, t in _KINDS if k == kind)
        bars = self._bars()

        ax.clear()
        ax.set_title(title)
        if not bars:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.5, 0.5, "Tick models to compare.", ha="center", va="center",
                    color="gray", transform=ax.transAxes)
            return

        labels = [bar.label for bar in bars]
        values = np.asarray(
            [self._bar_value(bar, kind, key) for bar in bars], dtype=float
        )
        # Colour follows the model's original tick order, so each model keeps its colour when sorted.
        colours = [_COLOURS[i % len(_COLOURS)] for i in range(len(values))]
        order = self._bar_order(values)
        labels = [labels[i] for i in order]
        values = values[order]
        colours = [colours[i] for i in order]
        xs = np.arange(len(values))
        finite = np.isfinite(values)
        ax.bar(xs[finite], values[finite], color=[c for c, f in zip(colours, finite) if f])
        # Mark unavailable (e.g. legacy bundles, overall on a pre-cube model) as an n/a tick.
        for i in np.where(~finite)[0]:
            ax.text(i, 0, "n/a", ha="center", va="bottom", color="gray", fontsize="small")
        if kind == "delta":
            ax.axhline(0.0, color="black", linestyle=":", linewidth=1)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel(f"{label}{f' ({suffix})' if suffix else ''}")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, axis="y", which="major", color="0.9", linewidth=0.6)
        ax.set_axisbelow(True)

    def _bar_order(self, values: np.ndarray) -> np.ndarray:
        """Index order for the bars under the current sort mode (original order if unsorted).

        Non-finite (n/a) values always trail the finite ones so they don't break the ranking.
        """
        n = len(values)
        if self._sort_mode == _SORT_NONE or n == 0:
            return np.arange(n)
        finite = np.isfinite(values)
        keys = np.where(finite, values, -np.inf if self._sort_mode == _SORT_DESC else np.inf)
        order = np.argsort(keys, kind="stable")
        if self._sort_mode == _SORT_DESC:
            order = order[::-1]
        # Stable n/a placement: keep them after the finite bars, in original order.
        finite_order = [i for i in order if finite[i]]
        na_order = [i for i in range(n) if not finite[i]]
        return np.asarray(finite_order + na_order, dtype=int)

    def chart_values(self, kind: str) -> ChartValues:
        """The ``kind`` panel's raw (model label, value) rows for the "Copy values" menu."""
        key = self.metric_combo.currentData()
        label = next(lbl for k, lbl, _ in _METRICS if k == key)
        suffix = next(s for k, _, s in _METRICS if k == key)
        unit = f"{label}{f' ({suffix})' if suffix else ''}"
        rows = [
            (bar.label, self._bar_value(bar, kind, key)) for bar in self._bars()
        ]
        # Match the on-screen / exported bar order so copied values line up with the chart.
        order = self._bar_order(np.asarray([v for _, v in rows], dtype=float))
        rows = [rows[i] for i in order]
        return ChartValues(rows=rows, unit=unit)

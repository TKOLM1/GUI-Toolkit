"""Module 1 page: data augmentation.

Builds its augmentation controls straight from the :data:`augment.AUGMENTATIONS`
registry (one checkbox per method, with min/max spin boxes per parameter), runs the
batch on an :class:`AugmentWorker`, and on success fills the shared :class:`Session` and
emits :attr:`request_next` so the shell can move to feature generation.

Inputs and outputs share the active project's ``plots/`` sub-folder: augmentation reads the
``plot(N).laz`` originals there and writes the ``plot(N)_aug(k).laz`` copies back into the same
folder (no duplicate originals) — there are no file pickers here.
"""

from __future__ import annotations

from itertools import groupby
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from augment import AUGMENTATIONS, AugmentConfig
from common.config import CONFIG
from common.naming import LAS_SUFFIXES, aug_number_from_name
from common.session import Session
from .widgets import (
    ClearSection,
    cascade_clear_dialog,
    downstream_feature_files,
    downstream_model_files,
    make_console,
    make_status_indicator,
    open_folder,
    set_status_indicator,
    style_button,
)
from .workers import AugmentWorker


class AugmentPage(QWidget):
    """The augmentation module UI."""

    request_next = Signal()  # emitted to ask the shell to switch to feature generation
    data_invalidated = Signal()  # clouds changed (Clear or re-run): shell drops stale cached datasets

    def __init__(self, session: Session) -> None:
        super().__init__()
        self._session = session
        self._worker: AugmentWorker | None = None
        # key -> {"check": QCheckBox, "params": {param_name: (min_spin, max_spin)}}
        self._aug_controls: dict[str, dict] = {}
        self._output_dir_value: Path | None = None

        root = QHBoxLayout(self)
        root.addWidget(self._build_augment_panel(), 6)
        root.addWidget(self._build_run_panel(), 4)

    # ------------------------------------------------------------------ #
    # Panels                                                             #
    # ------------------------------------------------------------------ #
    def _input_files(self) -> list[Path]:
        """The plots to augment: the original ``plot(N).laz`` files in the project's plots/ folder.

        Augmented copies (``..._aug(k).laz``) already in the folder are excluded so re-running never
        augments an augmented file.
        """
        project = self._session.project
        if project is None:
            return []
        return sorted(
            p for p in project.plots_dir.glob("*")
            if p.suffix.lower() in LAS_SUFFIXES and aug_number_from_name(p.name) is None
        )

    def _manifest_name(self) -> str:
        """The manifest CSV file name from the active preset (e.g. 'Data augmentation.csv')."""
        return f"{CONFIG.augmentation_table_name}.csv"

    def _build_augment_panel(self) -> QWidget:
        box = QGroupBox("Augmentations")
        layout = QVBoxLayout(box)

        toggles = QHBoxLayout()
        select_all = QPushButton("Select all")
        select_all.clicked.connect(lambda: self._set_all(True))
        select_none = QPushButton("Select none")
        select_none.clicked.connect(lambda: self._set_all(False))
        toggles.addWidget(select_all)
        toggles.addWidget(select_none)
        toggles.addStretch(1)
        layout.addLayout(toggles)

        container = QWidget()
        inner = QVBoxLayout(container)
        for group, defs in groupby(AUGMENTATIONS, key=lambda a: a.group):
            group_box = QGroupBox(group)
            group_layout = QVBoxLayout(group_box)
            for aug in defs:
                check = QCheckBox(aug.label)
                check.setToolTip(aug.tooltip)
                group_layout.addWidget(check)
                controls = {"check": check, "params": {}}
                # One min/max row per parameter (preset range overrides the registry default).
                preset_ranges = CONFIG.augment_ranges.get(aug.key, {})
                for param in aug.params:
                    lo_default, hi_default = preset_ranges.get(
                        param.name, (param.default_min, param.default_max)
                    )
                    row = QHBoxLayout()
                    row.addWidget(QLabel(f"   {param.label}:"))
                    min_spin = self._make_param_spin(param, lo_default)
                    max_spin = self._make_param_spin(param, hi_default)
                    min_spin.setToolTip(param.tooltip)
                    max_spin.setToolTip(param.tooltip)
                    row.addWidget(QLabel("min"))
                    row.addWidget(min_spin)
                    row.addWidget(QLabel("max"))
                    row.addWidget(max_spin)
                    row.addStretch(1)
                    group_layout.addLayout(row)
                    controls["params"][param.name] = (min_spin, max_spin)
                self._aug_controls[aug.key] = controls
            inner.addWidget(group_box)

        # How many methods per sample + samples per original + seed (defaults from the preset).
        settings = QGroupBox("How to combine")
        form = QFormLayout(settings)
        self.min_methods = QSpinBox()
        self.min_methods.setRange(1, len(AUGMENTATIONS))
        self.min_methods.setToolTip("Minimum number of distinct methods applied per augmented sample.")
        self.max_methods = QSpinBox()
        self.max_methods.setRange(1, len(AUGMENTATIONS))
        self.max_methods.setToolTip("Maximum number of distinct methods applied per augmented sample.")
        self.n_per_sample = QSpinBox()
        self.n_per_sample.setRange(0, 1000)
        self.n_per_sample.setToolTip(
            "How many augmented copies to create per original plot. The original "
            "(unaugmented) plot is always kept too, so N here yields N+1 files per plot, "
            "numbered aug(1)..aug(N)."
        )
        # The seed is the one project-wide seed set on the Setup tab (shown here read-only).
        self.seed_label = QLabel("0")
        self.seed_label.setToolTip(
            "The project-wide random seed (set on the Setup tab). The same seed + settings "
            "reproduce the run."
        )
        form.addRow("Min methods / sample", self.min_methods)
        form.addRow("Max methods / sample", self.max_methods)
        form.addRow("Augmented copies / plot", self.n_per_sample)
        form.addRow("Random seed (project)", self.seed_label)
        inner.addWidget(settings)
        inner.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        layout.addWidget(scroll, 1)
        self.apply_preset_defaults()
        return box

    def apply_preset_defaults(self) -> None:
        """Set every augmentation control back to the active preset's values.

        Called when the panel is built and again whenever the preset changes, so the ticked
        methods, their min/max ranges and the combination settings always describe the preset that
        is actually active rather than the one that happened to be selected at start-up.
        """
        for aug in AUGMENTATIONS:
            controls = self._aug_controls.get(aug.key)
            if controls is None:
                continue
            controls["check"].setChecked(aug.key in CONFIG.augment_selected)
            preset_ranges = CONFIG.augment_ranges.get(aug.key, {})
            for param in aug.params:
                lo, hi = preset_ranges.get(param.name, (param.default_min, param.default_max))
                min_spin, max_spin = controls["params"][param.name]
                min_spin.setValue(lo)
                max_spin.setValue(hi)
        self.min_methods.setValue(CONFIG.augment_min_methods)
        self.max_methods.setValue(CONFIG.augment_max_methods)
        self.n_per_sample.setValue(CONFIG.augment_copies)

    def _build_run_panel(self) -> QWidget:
        box = QGroupBox("Output & run")
        layout = QVBoxLayout(box)

        self.indicator = make_status_indicator()
        layout.addWidget(self.indicator)

        self.io_label = QLabel("Input / output: (no project open)")
        self.io_label.setWordWrap(True)
        self.io_label.setStyleSheet("color: gray;")
        layout.addWidget(self.io_label)

        self.run_button = QPushButton("Run augmentation")
        self.run_button.clicked.connect(self._on_run)
        style_button(self.run_button, "primary")
        layout.addWidget(self.run_button)

        self.clear_button = QPushButton("Clear augmented output")
        self.clear_button.setToolTip(
            "Delete the augmented .laz copies, the manifest, and any generated feature/targets "
            ".csv tables (they are derived from the augmented plots). Originals are kept."
        )
        self.clear_button.clicked.connect(self._on_clear)
        style_button(self.clear_button, "clear")
        layout.addWidget(self.clear_button)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        console_box, self.log = make_console(self)
        layout.addWidget(console_box, 1)

        self.open_folder_button = QPushButton("Open output folder")
        self.open_folder_button.setEnabled(False)
        self.open_folder_button.clicked.connect(self._on_open_folder)
        layout.addWidget(self.open_folder_button)

        self.next_button = QPushButton("Go to feature generation →")
        self.next_button.setEnabled(False)
        self.next_button.clicked.connect(self._on_go_next)
        style_button(self.next_button, "next")
        layout.addWidget(self.next_button)
        return box

    # ------------------------------------------------------------------ #
    # Small builders / helpers                                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _make_param_spin(param, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 100000.0)
        spin.setDecimals(4)
        spin.setSingleStep(0.01)
        spin.setValue(value)
        return spin

    def _set_all(self, checked: bool) -> None:
        for controls in self._aug_controls.values():
            controls["check"].setChecked(checked)

    def _build_config(self) -> AugmentConfig | None:
        selected = [k for k, c in self._aug_controls.items() if c["check"].isChecked()]
        if not selected:
            QMessageBox.warning(self, "No augmentations", "Select at least one augmentation.")
            return None
        if self.min_methods.value() > self.max_methods.value():
            QMessageBox.warning(self, "Method range", "Min methods cannot exceed max methods.")
            return None
        ranges: dict[str, dict[str, tuple[float, float]]] = {}
        for key in selected:
            ranges[key] = {
                name: (lo.value(), hi.value())
                for name, (lo, hi) in self._aug_controls[key]["params"].items()
            }
        return AugmentConfig(
            selected=selected,
            ranges=ranges,
            min_methods=self.min_methods.value(),
            max_methods=self.max_methods.value(),
            n_per_sample=self.n_per_sample.value(),
            seed=self._project_seed(),
        )

    # ------------------------------------------------------------------ #
    # Actions                                                            #
    # ------------------------------------------------------------------ #
    def _on_run(self) -> None:
        if self._session.project is None:
            QMessageBox.warning(self, "No project", "Open or create a project on the Setup tab first.")
            return
        files = self._input_files()
        if not files:
            QMessageBox.warning(
                self, "No input plots",
                "No original plots found in the project's plots/ folder. Run the clip tab "
                "(or import already-clipped plots there) first.",
            )
            return
        config = self._build_config()
        if config is None:
            return

        # Augmented copies go into the project's shared plots/ folder, beside the originals.
        self._output_dir_value = self._session.project.plots_dir
        self._set_running(True)
        self.log.clear()
        total = len(files) * config.n_per_sample
        self.progress.setRange(0, total)
        self.progress.setValue(0)
        self._log(f"Augmenting {len(files)} plot(s) -> {self._output_dir_value}")

        self._worker = AugmentWorker(
            files, self._output_dir_value, config, manifest_name=self._manifest_name()
        )
        self._worker.progressed.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_progress(self, done: int, total: int, name: str, error) -> None:
        self.progress.setValue(done)
        self._log(f"  [{'skipped' if error else 'ok'}] {name}" + (f": {error}" if error else ""))

    def _on_finished(self, result) -> None:
        self._set_running(False)
        self._log(
            f"Done. {result.n_augmented} augmented, "
            f"{len(result.failed)} failed. Output: {result.output_dir}"
        )
        # Feed the next module: every file in the output folder (originals + augmented copies).
        all_files = sorted(result.output_dir.glob("*.la[sz]"))
        self._session.input_files = all_files
        self._session.augment_output_dir = result.output_dir
        self._session.manifest_path = result.manifest_path
        self.open_folder_button.setEnabled(True)
        self.next_button.setEnabled(True)
        self._refresh_io_label()
        # The clouds just changed on disk — drop any cached learning dataset built from the old set.
        self.data_invalidated.emit()
        QMessageBox.information(
            self, "Augmentation finished",
            f"{result.n_augmented} augmented file(s) written to:\n{result.output_dir}",
        )

    def _on_failed(self, message: str) -> None:
        self._set_running(False)
        self._log(f"ERROR: {message}")
        QMessageBox.critical(self, "Augmentation failed", message)

    def _on_open_folder(self) -> None:
        if self._output_dir_value:
            open_folder(self._output_dir_value)

    # -- clear ------------------------------------------------------------- #
    def _on_clear(self) -> None:
        if self._session.project is None:
            QMessageBox.warning(self, "No project", "Open or create a project first.")
            return
        project = self._session.project
        folder = project.plots_dir
        # The augmented copies (primary) plus everything derived from them — the feature/targets
        # tables and any saved models — so re-augmenting can't leave a stale targets table behind
        # (the silent-subset bug). Only the augmented copies and the manifest are cleared here; the
        # original plot(N).laz files are left in place. All downstream sections default to checked.
        aug_clouds = [
            p for p in folder.glob("*")
            if p.suffix.lower() in LAS_SUFFIXES and aug_number_from_name(p.name) is not None
        ]
        manifest = [p for p in folder.glob("*.csv") if p.name == self._manifest_name()]
        sections = [
            ClearSection("augmented", "Augmented copies + manifest", aug_clouds + manifest, primary=True),
            ClearSection("features", "Generated features + targets",
                         downstream_feature_files(project.features_dir)),
            ClearSection("models", "Saved models", downstream_model_files(project.model_dir)),
        ]
        removed = cascade_clear_dialog(self, "Clear augmented output", sections)
        if not removed:
            return
        # Reset the hand-off state for whatever was actually cleared.
        if removed.get("augmented"):
            self._session.manifest_path = None
        if removed.get("features"):
            self._session.feature_table_path = None
            self._session.targets_table_path = None
        # _refresh_io_label re-enables the next button while any input plots remain (augmentation
        # is optional), so clearing the augmented copies never blocks advancing with originals present.
        self._refresh_io_label()
        total = sum(removed.values())
        self._log(
            f"Cleared {total} file(s): "
            + ", ".join(f"{k}={n}" for k, n in removed.items() if n)
            + f". ({folder})"
        )
        # Tell the shell to drop the learning tabs' cached datasets — they may have been built from
        # the files we just deleted (the in-memory desync that let the optimizer read vanished clouds).
        self.data_invalidated.emit()

    def _on_go_next(self) -> None:
        self.request_next.emit()

    # ------------------------------------------------------------------ #
    def _set_running(self, running: bool) -> None:
        self.run_button.setEnabled(not running)
        self.run_button.setText("Working…" if running else "Run augmentation")

    def _log(self, message: str) -> None:
        self.log.appendPlainText(message)

    def _project_seed(self) -> int:
        return int(self._session.project.seed) if self._session.project else 0

    def reset_for_project(self) -> None:
        """Drop the previous project's cached output dir so a new one starts clean.

        Called by the shell when a project is created/opened. on_enter (via _refresh_io_label)
        re-derives the I/O label, next-button state and indicator from the now-active project's
        disk, so the cached output path is the only per-project state to clear here.
        """
        self._output_dir_value = None

    def on_enter(self) -> None:  # called by the shell when this page is shown
        self.seed_label.setText(str(self._project_seed()))
        self._refresh_io_label()

    def _refresh_io_label(self) -> None:
        project = self._session.project
        if project is None:
            self.io_label.setText("Input / output: (no project open)")
            self.next_button.setEnabled(False)
            set_status_indicator(self.indicator, False, "No project open.")
            return
        n = len(self._input_files())
        self.io_label.setText(
            f"Input / output ← → {project.plots_dir}  ({n} original plot file(s))"
        )
        # Augmentation is optional: as soon as there are input plots the user may skip straight
        # to feature generation, so the (green) next button is enabled whenever plots are present —
        # whether or not augmented copies already exist from a previous run.
        n_augmented = sum(
            1 for p in project.plots_dir.glob("*")
            if p.suffix.lower() in LAS_SUFFIXES and aug_number_from_name(p.name) is not None
        )
        self.next_button.setEnabled(bool(n or n_augmented))
        # Same short "N plots loaded" phrasing every tab uses, with the augmented count appended
        # when there are any (this tab's own output).
        if n_augmented:
            set_status_indicator(
                self.indicator, True, f"{n} plots loaded ({n_augmented} augmented)"
            )
        elif n:
            set_status_indicator(self.indicator, True, f"{n} plots loaded")
        else:
            set_status_indicator(
                self.indicator, False, "No clipped plots found — run the Clipping tab first."
            )

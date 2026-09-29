"""Module 2 page: feature generation (the original tool, as a page).

This is the former standalone window refactored into a :class:`QWidget` page. Inputs and
outputs are now derived from the active project: the plots to process are read from the
project's shared ``plots/`` sub-folder (clipped originals plus any augmented copies) and the
``.csv`` tables are written to ``features/``. There is no file picker left at all: the reference
sheet is chosen (and linked to the plots) on the **Import** tab, and this page reads the loaded
table off the shared :class:`~common.session.Session`. That split keeps one question per tab —
Import answers "did my ground truth reach my plots?", this page answers "what do I compute from
it?".

* :meth:`on_enter` re-reads the project (and offers to reuse an already-present features table).
* a **ground-truth selector** lets the user mark one reference column as the target; it
  is grayed out in the reference list, excluded from the features export, and written to
  a separate targets table.
* a "Go to ML" button hands the features + targets paths to the next module.
"""

from __future__ import annotations

from itertools import groupby
from pathlib import Path

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
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Signal

from common.config import CONFIG
from common.naming import LAS_SUFFIXES
from common.session import Session
from featuregen.external import ExternalData
from featuregen.features import FEATURES
from featuregen.io_las import Config, list_dimensions
from featuregen.pipeline import MAX_ONEHOT_COLUMNS
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
    wire_parent_toggle,
)
from .workers import BatchWorker

_NONE_TARGET = "(none)"


# The "no channel" sentinel for the RGB dropdowns (an unset colour channel is legitimate: many
# clouds carry no colour at all, and the colour features then simply come out NaN).
_NO_CHANNEL = "(none)"
# Channel names auto-detection accepts for each colour, lower-cased.
_RED_ALIASES = {"red", "r"}
_GREEN_ALIASES = {"green", "g"}
_BLUE_ALIASES = {"blue", "b"}
# Channel names that mean "height", best first. Used only as a fallback when neither the user nor
# the active preset has picked one: the dropdown lists the file's dimensions in storage order, so
# falling back to the first one silently made X the height on any cloud without a height channel.
_HEIGHT_ALIASES = ("relativeheight", "heightaboveground", "height", "hag", "normalizedz", "z")


class FeaturePage(QWidget):
    """The feature-generation module UI."""

    request_next = Signal()  # ask the shell to switch to the ML page
    data_invalidated = Signal()  # workbooks changed (Clear or re-run): shell drops stale caches

    def __init__(self, session: Session) -> None:
        super().__init__()
        self._session = session
        self._worker: BatchWorker | None = None
        self._feature_checks: dict[str, QCheckBox] = {}
        # The loaded reference sheet the column checkboxes were last built from, so on_enter can tell
        # a reload on Import (rebuild, losing the ticks) from the same sheet (leave the ticks alone).
        self._external: ExternalData | None = None
        self._external_checks: dict[str, QCheckBox] = {}
        self._external_encode_checks: dict[str, QCheckBox] = {}  # "split into binary" per column
        self._output_path: Path | None = None
        self._offered_root: Path | None = None  # project we've already offered existing features for

        root = QHBoxLayout(self)
        root.addWidget(self._build_feature_panel(), 4)
        root.addWidget(self._build_external_panel(), 4)
        root.addWidget(self._build_run_panel(), 4)

    # ------------------------------------------------------------------ #
    # Panels                                                             #
    # ------------------------------------------------------------------ #
    def _input_files(self) -> list[Path]:
        """The plots to feature-generate: every .las/.laz in the project's plots/ folder.

        That folder holds the clipped originals plus any augmented copies, so feature generation
        processes both whether or not augmentation was run.
        """
        project = self._session.project
        if project is None:
            return []
        return sorted(p for p in project.plots_dir.glob("*") if p.suffix.lower() in LAS_SUFFIXES)

    def _build_feature_panel(self) -> QWidget:
        box = QGroupBox("Features")
        layout = QVBoxLayout(box)
        toggles = QHBoxLayout()
        select_all = QPushButton("Select all")
        select_all.clicked.connect(lambda: self._set_all_features(True))
        select_none = QPushButton("Select none")
        select_none.clicked.connect(lambda: self._set_all_features(False))
        toggles.addWidget(select_all)
        toggles.addWidget(select_none)
        toggles.addStretch(1)
        layout.addLayout(toggles)

        # Two-level grouping: one checkable QGroupBox per class (Height / Color / 2D),
        # and inside it a bold togglable "type" header per type — no third box, to keep
        # the dense panel readable. The class box and each type header toggle all of their
        # feature checkboxes; both stay in sync as children change. A class with a single
        # type drops the redundant header (the class box already names it).
        container = QWidget()
        inner = QVBoxLayout(container)
        for cls, cls_defs in groupby(FEATURES, key=lambda f: f.cls):
            types = [(g, list(defs)) for g, defs in groupby(cls_defs, key=lambda f: f.group)]
            cls_box = QGroupBox(cls)
            cls_box.setCheckable(True)
            cls_box.setChecked(True)
            cls_layout = QVBoxLayout(cls_box)
            cls_children: list[QCheckBox] = []
            single_type = len(types) == 1
            for group, defs in types:
                type_children: list[QCheckBox] = []
                if not single_type:
                    type_header = QCheckBox(group)
                    type_header.setChecked(True)
                    type_header.setStyleSheet("font-weight: bold;")
                    cls_layout.addWidget(type_header)
                for feature in defs:
                    check = QCheckBox(feature.label)
                    check.setChecked(True)
                    check.setToolTip(feature.tooltip)
                    self._feature_checks[feature.key] = check
                    # Indent the leaf checkboxes under a type header (none when single-type).
                    row = QHBoxLayout()
                    row.setContentsMargins(16 if not single_type else 0, 0, 0, 0)
                    row.addWidget(check)
                    cls_layout.addLayout(row)
                    type_children.append(check)
                    cls_children.append(check)
                if not single_type:
                    wire_parent_toggle(type_header, type_children)
            wire_parent_toggle(cls_box, cls_children)
            inner.addWidget(cls_box)
        inner.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        layout.addWidget(scroll, 1)
        self.apply_preset_defaults()
        return box

    def apply_preset_defaults(self) -> None:
        """Tick the feature checkboxes per the active preset's enabled/disabled lists.

        Called when the panel is built and again whenever the preset changes, so the ticked
        features describe the preset that is actually active rather than the start-up one.
        """
        on = set(CONFIG.enabled_features([f.key for f in FEATURES]))
        for key, check in self._feature_checks.items():
            check.setChecked(key in on)
        if hasattr(self, "target_combo"):  # first call runs before the reference panel is built
            self._apply_preset_target()

    def _build_external_panel(self) -> QWidget:
        box = QGroupBox("Imported reference features")
        layout = QVBoxLayout(box)

        hint = QLabel("Loaded on the Import tab. Pick which of its columns to use below.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        layout.addWidget(hint)

        # Ground-truth (target) selector.
        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Ground truth (target):"))
        self.target_combo = QComboBox()
        self.target_combo.addItem(_NONE_TARGET)
        self.target_combo.setToolTip(
            "Pick a reference column to use as the ML target (y). It is excluded from the "
            "features export and written to a separate targets workbook."
        )
        self.target_combo.currentTextChanged.connect(self._on_target_changed)
        target_row.addWidget(self.target_combo, 1)
        layout.addLayout(target_row)

        toggles = QHBoxLayout()
        select_all = QPushButton("Select all")
        select_all.clicked.connect(lambda: self._set_all_external(True))
        select_none = QPushButton("Select none")
        select_none.clicked.connect(lambda: self._set_all_external(False))
        toggles.addWidget(select_all)
        toggles.addWidget(select_none)
        toggles.addStretch(1)
        layout.addLayout(toggles)

        self.external_status = QLabel("No reference sheet — load one on the Import tab.")
        self.external_status.setWordWrap(True)
        layout.addWidget(self.external_status)

        self._external_container = QWidget()
        self._external_layout = QVBoxLayout(self._external_container)
        self._external_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._external_container)
        layout.addWidget(scroll, 1)
        return box

    def _build_run_panel(self) -> QWidget:
        box = QGroupBox("Output & run")
        layout = QVBoxLayout(box)

        self.indicator = make_status_indicator()
        layout.addWidget(self.indicator)

        self.io_label = QLabel("Input / output: (no project open)")
        self.io_label.setWordWrap(True)
        self.io_label.setStyleSheet("color: gray;")
        layout.addWidget(self.io_label)

        layout.addWidget(QLabel("Output file name"))
        self.output_name = QLineEdit(self._default_output_name())
        layout.addWidget(self.output_name)

        layout.addWidget(self._build_advanced_box())

        self.run_button = QPushButton("Generate features")
        self.run_button.clicked.connect(self._on_run)
        style_button(self.run_button, "primary")
        layout.addWidget(self.run_button)

        self.clear_button = QPushButton("Clear generated features")
        self.clear_button.setToolTip(
            "Delete the feature and targets .csv tables from the output folder."
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

        self.next_button = QPushButton("Go to ML →")
        self.next_button.setEnabled(False)
        self.next_button.clicked.connect(lambda: self.request_next.emit())
        style_button(self.next_button, "next")
        layout.addWidget(self.next_button)
        return box

    def _build_advanced_box(self) -> QWidget:
        # Not a checkable group box any more: the height channel (and now the RGB channels) must
        # always be visible, because generation fails or silently drops colour features when they
        # are wrong — hiding them behind an "Advanced" tick made that easy to miss.
        box = QGroupBox("Channels & parameters")
        form = QFormLayout(box)
        defaults = Config()
        # Height channel is a dropdown of the channels actually detected in the project's plots
        # (filled by _refresh_channel_combo on enter), not a free-text field; there is no hardcoded
        # default — the active preset's height_channel pre-selects a channel when present.
        self.channel_combo = QComboBox()
        self.channel_combo.setToolTip(
            "Point dimension holding height-above-ground. The list is the channels detected in "
            "this project's plots; the active preset's height_channel pre-selects one when present."
        )
        # RGB channels: auto-detected from the plots (see _refresh_channel_combo), each overridable
        # exactly like the height channel. "(none)" turns the colour features off for that channel.
        self.red_combo = QComboBox()
        self.green_combo = QComboBox()
        self.blue_combo = QComboBox()
        for combo, colour in (
            (self.red_combo, "red"), (self.green_combo, "green"), (self.blue_combo, "blue")
        ):
            combo.setToolTip(
                f"Point dimension holding the {colour} channel. Auto-detected from this project's "
                f"plots; pick another dimension to override it, or '(none)' to skip colour features."
            )
        self.veg_spin = QSpinBox()
        self.veg_spin.setRange(0, 255)
        self.veg_spin.setValue(defaults.veg_code)
        self.ground_spin = QSpinBox()
        self.ground_spin.setRange(0, 255)
        self.ground_spin.setValue(defaults.ground_code)
        self.bin_spin = QDoubleSpinBox()
        self.bin_spin.setRange(0.01, 100.0)
        self.bin_spin.setSingleStep(0.1)
        self.bin_spin.setDecimals(2)
        self.bin_spin.setValue(defaults.entropy_bin_size)
        self.slope_pct_spin = QDoubleSpinBox()
        self.slope_pct_spin.setRange(1.0, 49.0)
        self.slope_pct_spin.setSingleStep(1.0)
        self.slope_pct_spin.setDecimals(1)
        self.slope_pct_spin.setValue(defaults.slope_strata_pct)
        self.slope_pct_spin.setToolTip(
            "Top/bottom height percentage used by the hand-crafted top-bottom angle "
            "features. Widen if the thin strata prove noisy on sparse plots."
        )
        form.addRow("Height channel", self.channel_combo)
        form.addRow("Red channel", self.red_combo)
        form.addRow("Green channel", self.green_combo)
        form.addRow("Blue channel", self.blue_combo)
        form.addRow("Vegetation class code", self.veg_spin)
        form.addRow("Ground class code", self.ground_spin)
        form.addRow("Entropy bin size (m)", self.bin_spin)
        form.addRow("Slope strata (%)", self.slope_pct_spin)
        return box

    # ------------------------------------------------------------------ #
    # Actions / helpers                                                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _default_output_name() -> str:
        """The features-table file name from the active preset (default 'features')."""
        return f"{CONFIG.feature_table_name}.csv"

    def _set_all_features(self, checked: bool) -> None:
        for check in self._feature_checks.values():
            check.setChecked(checked)

    def _selected_feature_keys(self) -> list[str]:
        return [f.key for f in FEATURES if self._feature_checks[f.key].isChecked()]

    # -- reference / target ------------------------------------------------ #
    def _sync_external(self) -> None:
        """Adopt whatever reference sheet the Import tab has loaded into the session.

        Called from :meth:`on_enter`, so switching to this tab always reflects the sheet (and key
        column) chosen on Import. The column checkboxes are rebuilt only when the Import tab has
        *reloaded* the sheet: re-entering the tab with the same loaded sheet must not silently reset
        the user's ticks or their chosen target. Identity, not the file path, decides — a reload of
        the same file with a different key or row filter (e.g. stage Z31 -> Z65) holds different
        rows, and keeping the old copy would write the wrong targets.
        """
        external: ExternalData | None = self._session.external
        if external is not None and external is self._external:
            return
        self._external = external
        if external is None:
            self._rebuild_external_checks([])
            self.external_status.setText("No reference sheet — load one on the Import tab.")
            return
        self._rebuild_external_checks(external.columns)
        self.external_status.setText(
            f"{len(external.frame)} plot(s) and {len(external.columns)} column(s) "
            f"from {external.source_path.name}."
        )

    def _rebuild_external_checks(self, columns: list[str]) -> None:
        while self._external_layout.count() > 1:
            item = self._external_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._external_checks = {}
        self._external_encode_checks = {}
        # Pre-tick from the active preset: which reference columns to use, and which to one-hot.
        on = set(CONFIG.enabled_reference(list(columns)))
        split_on = set(CONFIG.reference_features_split)
        for col in columns:
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            check = QCheckBox(col)
            check.setChecked(col in on)
            check.setToolTip(f"Imported reference column '{col}', matched by plot number")
            self._external_checks[col] = check
            split = QCheckBox("split")
            split.setChecked(col in split_on)
            split.setToolTip(
                "Treat this as a coded/categorical column and split it into one binary feature "
                "per unique value (one-hot). Refused if it would create over 100 binary features."
            )
            self._external_encode_checks[col] = split
            h.addWidget(check, 1)
            h.addWidget(split)
            self._external_layout.insertWidget(self._external_layout.count() - 1, row)
        # Repopulate the target dropdown.
        self.target_combo.blockSignals(True)
        self.target_combo.clear()
        self.target_combo.addItem(_NONE_TARGET)
        self.target_combo.addItems(columns)
        self.target_combo.setCurrentText(_NONE_TARGET)
        self.target_combo.blockSignals(False)
        self._apply_preset_target()

    def _apply_preset_target(self) -> None:
        """Select the active preset's target column when the loaded sheet has it (else none)."""
        preferred = CONFIG.target_column.strip()
        target = preferred if preferred and self.target_combo.findText(preferred) >= 0 else _NONE_TARGET
        self.target_combo.blockSignals(True)
        self.target_combo.setCurrentText(target)
        self.target_combo.blockSignals(False)
        self._on_target_changed(target)

    def _on_target_changed(self, target: str) -> None:
        """Gray out the chosen target in the reference list so it can't double as a feature."""
        # Share the chosen target with the rest of the app (the top-right status readout, ML/DL).
        self._session.target_column = None if target == _NONE_TARGET else target
        for col, check in self._external_checks.items():
            is_target = col == target and target != _NONE_TARGET
            if is_target:
                check.setChecked(False)
            check.setEnabled(not is_target)
            split = self._external_encode_checks.get(col)
            if split is not None:  # the target is a regression label; never one-hot it
                if is_target:
                    split.setChecked(False)
                split.setEnabled(not is_target)

    def _set_all_external(self, checked: bool) -> None:
        for col, check in self._external_checks.items():
            if check.isEnabled():
                check.setChecked(checked)

    def _selected_external_columns(self) -> list[str]:
        if self._external is None:
            return []
        target = self._selected_target()
        return [
            c for c in self._external.columns
            if self._external_checks[c].isChecked() and c != target
        ]

    def _encoded_columns(self) -> list[str]:
        """Selected reference columns whose 'split' (one-hot) toggle is on (target excluded)."""
        selected = set(self._selected_external_columns())
        return [
            c for c, chk in self._external_encode_checks.items()
            if chk.isChecked() and c in selected
        ]

    def _selected_target(self) -> str | None:
        text = self.target_combo.currentText()
        return None if text == _NONE_TARGET else text

    def _current_config(self) -> Config:
        # The dropdown holds the channels detected in the plots; fall back to the preset value and
        # finally the Config default so a missing/empty selection never yields a blank channel.
        channel = self.channel_combo.currentText().strip() or CONFIG.height_channel.strip()
        return Config(
            height_channel=channel or Config().height_channel,
            red_channel=self._combo_channel(self.red_combo),
            green_channel=self._combo_channel(self.green_combo),
            blue_channel=self._combo_channel(self.blue_combo),
            veg_code=self.veg_spin.value(),
            ground_code=self.ground_spin.value(),
            entropy_bin_size=self.bin_spin.value(),
            slope_strata_pct=self.slope_pct_spin.value(),
        )

    # -- run --------------------------------------------------------------- #
    def _on_run(self) -> None:
        if self._session.project is None:
            QMessageBox.warning(self, "No project", "Open or create a project on the Setup tab first.")
            return
        files = self._input_files()
        if not files:
            QMessageBox.warning(
                self, "No input plots",
                "No plots found in the project's plots/ folder. Run the clip (and optionally "
                "augment) tabs first.",
            )
            return
        # No height channel means every height-derived feature (nearly all of them) would fail on
        # the first file, so say so up front rather than after a batch of "channel not found".
        if not self.channel_combo.currentText().strip():
            QMessageBox.warning(
                self, "No height channel",
                "No height channel is selected.\n\nPick the point dimension holding "
                "height-above-ground (e.g. RelativeHeight) under 'Channels & parameters' — nearly "
                "every feature is computed from it.",
            )
            return
        feature_keys = self._selected_feature_keys()
        external_columns = self._selected_external_columns()
        encode_columns = self._encoded_columns()
        target = self._selected_target()
        if not feature_keys and not external_columns:
            QMessageBox.warning(self, "No features", "Select at least one feature.")
            return
        # Refuse a one-hot split that would explode into too many binary features (early, clear).
        if encode_columns and self._external is not None:
            total_binary = sum(len(self._external.unique_values(c)) for c in encode_columns)
            if total_binary > MAX_ONEHOT_COLUMNS:
                QMessageBox.warning(
                    self, "Too many binary features",
                    f"Splitting {len(encode_columns)} coded column(s) would create "
                    f"{total_binary} binary features, over the limit of {MAX_ONEHOT_COLUMNS}.\n\n"
                    "Untick some 'split' columns or pick lower-cardinality ones.",
                )
                return
        # Warn up-front when no ground-truth target is set (the result can't be used for ML).
        if target is None:
            if QMessageBox.question(
                self, "No target selected",
                "No ground-truth (target) column is selected, so the generated features can't be "
                "used to train an ML model.\n\nGenerate features anyway?",
            ) != QMessageBox.Yes:
                return
        out_dir = self._session.project.features_dir
        name = self.output_name.text().strip() or self._default_output_name()
        if not name.lower().endswith(".csv"):
            name += ".csv"
        self._output_path = out_dir / name
        targets_path = (
            out_dir / f"{CONFIG.targets_table_name}.csv" if target else None
        )

        config = self._current_config()
        # Persist the slope-strata % these features are generated with, so the Results viewer can
        # draw the hand-crafted slope planes at the same percentile the feature numbers used.
        self._session.project.set_slope_strata_pct(config.slope_strata_pct)
        # Persist the height channel too, so the Results viewer draws the cloud + feature geometry
        # from the same dimension the feature numbers used (e.g. RelativeHeight or Z).
        self._session.project.set_height_channel(config.height_channel)

        self._set_running(True)
        self.log.clear()
        self.progress.setRange(0, len(files))
        self.progress.setValue(0)
        self._log(f"Processing {len(files)} file(s) -> {self._output_path}")
        if target:
            self._log(f"Target '{target}' -> {targets_path}")

        self._worker = BatchWorker(
            files,
            self._output_path,
            feature_keys,
            config,
            external=self._external,
            external_columns=external_columns,
            encode_columns=encode_columns,
            target_column=target,
            targets_path=targets_path,
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
            f"Done. {result.n_succeeded} written, {result.n_failed} failed. "
            f"Saved to {result.output_path}"
        )
        if result.failed:
            self._log("Failed files:")
            for fname, reason in result.failed:
                self._log(f"  - {fname}: {reason}")
        self.open_folder_button.setEnabled(True)

        self._session.feature_table_path = result.output_path
        self._session.targets_table_path = result.targets_path
        # The workbooks just changed on disk — drop any cached learning dataset built from the old set.
        self.data_invalidated.emit()
        if result.targets_path is not None:
            self.next_button.setEnabled(True)
            self._log(f"Targets written to {result.targets_path}. Ready for ML.")
        else:
            self.next_button.setEnabled(False)
            self._log("No target selected - choose a ground-truth column to enable ML.")
        self._refresh_indicator()

        QMessageBox.information(
            self, "Finished",
            f"{result.n_succeeded} file(s) written to:\n{result.output_path}"
            + (f"\n\n{result.n_failed} failed (see log)." if result.failed else ""),
        )

    def _on_failed(self, message: str) -> None:
        self._set_running(False)
        self._log(f"ERROR: {message}")
        QMessageBox.critical(self, "Run failed", message)

    def _on_open_folder(self) -> None:
        if self._output_path:
            open_folder(self._output_path.parent)

    # -- clear ------------------------------------------------------------- #
    def _on_clear(self) -> None:
        if self._session.project is None:
            QMessageBox.warning(self, "No project", "Open or create a project first.")
            return
        project = self._session.project
        folder = project.features_dir
        # The feature stage (primary) plus the models derived from it: a saved model predicts
        # against these features, so regenerating them makes its field map stale. Models default
        # to checked.
        sections = [
            ClearSection("features", "Generated features + targets",
                         downstream_feature_files(folder), primary=True),
            ClearSection("models", "Saved models", downstream_model_files(project.model_dir)),
        ]
        removed = cascade_clear_dialog(self, "Clear generated features", sections)
        if not removed:
            return
        if removed.get("features"):
            self._session.feature_table_path = None
            self._session.targets_table_path = None
            self._output_path = None
            self.next_button.setEnabled(False)
            self._refresh_indicator()
        total = sum(removed.values())
        self._log(
            f"Cleared {total} file(s): "
            + ", ".join(f"{k}={n}" for k, n in removed.items() if n)
            + f". ({folder})"
        )
        # Tell the shell to drop the learning tabs' cached datasets — they may have been built from
        # the workbooks we just deleted (the in-memory desync that let training read vanished files).
        self.data_invalidated.emit()

    def _set_running(self, running: bool) -> None:
        self.run_button.setEnabled(not running)
        self.run_button.setText("Working…" if running else "Generate features")

    def _log(self, message: str) -> None:
        self.log.appendPlainText(message)

    # ------------------------------------------------------------------ #
    # Hand-off                                                           #
    # ------------------------------------------------------------------ #
    def reset_for_project(self) -> None:
        """Wipe the previous project's reference columns, output name and log so a new one starts clean.

        Called by the shell when a project is created/opened, before on_enter re-seeds the page
        from the now-active project. on_enter only fills the output-name field when it is empty, so
        without this it'd keep the prior project's value; clearing ``_offered_root`` lets on_enter
        re-offer the new project's existing features.

        The sheet itself belongs to the Import tab (which clears the session's copy in its own
        ``reset_for_project``); here we only drop the column checkboxes built from it, so
        ``_sync_external`` rebuilds them from whatever the new project loads.
        """
        self._external = None
        self._rebuild_external_checks([])
        self.external_status.setText("No reference sheet — load one on the Import tab.")
        # The target dropdown and the channel dropdowns keep the user's previous pick whenever the
        # name still exists (see _refresh_channel_combo), which would carry a channel or a target
        # column across projects; empty them so the new project's plots decide.
        self.target_combo.blockSignals(True)
        self.target_combo.clear()
        self.target_combo.addItem(_NONE_TARGET)
        self.target_combo.blockSignals(False)
        for combo in (self.channel_combo, self.red_combo, self.green_combo, self.blue_combo):
            combo.blockSignals(True)
            combo.clear()
            combo.blockSignals(False)
        self.output_name.clear()
        self._output_path = None
        self._offered_root = None
        self.log.clear()
        self.open_folder_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self._refresh_indicator()

    def on_enter(self) -> None:
        """Re-read the active project: adopt the Import tab's reference sheet, refresh the I/O label.

        Reuses an already-present features workbook the first time a project with existing
        features is entered, so an opened project is ready for ML without re-running.
        """
        project = self._session.project
        if project is None:
            self.io_label.setText("Input / output: (no project open)")
            set_status_indicator(self.indicator, False, "No project open.")
            return
        self._sync_external()
        # Keep the default output name in sync with the active preset.
        if not self.output_name.text().strip():
            self.output_name.setText(self._default_output_name())

        n = len(self._input_files())
        self.io_label.setText(
            f"Input ← {project.plots_dir.name}/  ({n} plot file(s))\n"
            f"Output → {project.features_dir}"
        )
        self._refresh_channel_combo()
        if self._offered_root != project.root:
            self._offered_root = project.root
            self._reuse_existing_features(project.features_dir)
        self._refresh_indicator()

    def _refresh_indicator(self) -> None:
        """Show whether a features workbook (and a targets workbook) is present for ML.

        Leads with the same short "N plots loaded" count the earlier tabs show, so the indicator
        reads consistently across the pipeline.
        """
        has_features = self._session.feature_table_path is not None
        has_targets = self._session.targets_table_path is not None
        n = len(self._input_files())
        if has_features and has_targets:
            set_status_indicator(
                self.indicator, True, f"{n} plots loaded — features + targets ready for ML."
            )
        elif has_features:
            set_status_indicator(
                self.indicator, False, "Features ready, but no target — pick a ground-truth column."
            )
        else:
            set_status_indicator(
                self.indicator, False, "No features yet — generate them, or reuse an existing set."
            )

    def _refresh_channel_combo(self) -> None:
        """Fill the Height-channel dropdown with the dimensions detected in the project's plots.

        Reads only the first plot's header (channels are uniform across a project's clouds) so this
        stays cheap. The user's current pick is preserved if it still exists; otherwise the active
        preset's ``height_channel`` is pre-selected when present, then the first height-like channel
        name (``RelativeHeight``/``Z``/…), and only then the first channel in the file.
        There is no hardcoded default — an empty plots/ folder simply leaves the dropdown empty.
        """
        files = self._input_files()
        try:
            channels = list_dimensions(files[0]) if files else []
        except Exception:  # noqa: BLE001 - a bad header must not break entering the tab
            channels = []
        previous = self.channel_combo.currentText().strip()
        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        self.channel_combo.addItems(channels)
        if previous in channels:
            self.channel_combo.setCurrentText(previous)
        elif CONFIG.height_channel.strip() in channels:
            self.channel_combo.setCurrentText(CONFIG.height_channel.strip())
        else:
            fallback = next(
                (c for alias in _HEIGHT_ALIASES for c in channels if c.strip().lower() == alias),
                None,
            )
            if fallback is not None:
                self.channel_combo.setCurrentText(fallback)
            # else: leave on the first detected channel (index 0), or empty if none were found.
        self.channel_combo.blockSignals(False)
        self._refresh_rgb_combos(channels)

    def _refresh_rgb_combos(self, channels: list[str]) -> None:
        """Fill (and auto-detect) the three RGB dropdowns from the detected ``channels``.

        Auto-detection matches a channel name case-insensitively against the usual spellings for
        each colour (``red``/``r``, ``green``/``g``, ``blue``/``b``); anything the file does not
        carry falls back to ``(none)``. A pick the user already made is kept whenever that channel
        still exists, so re-entering the tab never overrides a manual override.
        """
        for combo, aliases in (
            (self.red_combo, _RED_ALIASES),
            (self.green_combo, _GREEN_ALIASES),
            (self.blue_combo, _BLUE_ALIASES),
        ):
            previous = combo.currentText().strip()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(_NO_CHANNEL)
            combo.addItems(channels)
            if previous and previous != _NO_CHANNEL and previous in channels:
                combo.setCurrentText(previous)
            else:
                detected = next((c for c in channels if c.strip().lower() in aliases), None)
                combo.setCurrentText(detected or _NO_CHANNEL)
            combo.blockSignals(False)

    @staticmethod
    def _combo_channel(combo) -> str:
        """A channel dropdown's value, with the ``(none)`` sentinel mapped to an empty string."""
        text = combo.currentText().strip()
        return "" if text == _NO_CHANNEL else text

    def _reuse_existing_features(self, folder: Path) -> None:
        """If a features table is already in ``folder``, reuse it silently (skip running)."""
        candidates = [
            p for p in folder.glob("*.csv")
            if "feature" in p.name.lower()
            and "target" not in p.name.lower()
            and "manifest" not in p.name.lower()
            and "interaction" not in p.name.lower()
        ]
        if not candidates:
            return
        features = candidates[0]
        targets = next((p for p in folder.glob("*.csv") if "target" in p.name.lower()), None)
        self._session.feature_table_path = features
        self._session.targets_table_path = targets
        self._output_path = features
        self.open_folder_button.setEnabled(True)
        self.next_button.setEnabled(targets is not None)
        self._log(f"Using existing features: {features}")
        if targets is None:
            self._log("No target table found - generate with a target, or browse in ML.")

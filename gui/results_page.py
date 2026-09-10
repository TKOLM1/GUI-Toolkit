"""The results page — a tabbed view over a *saved model's* per-plot predictions.

Models are saved as bundles in the project's ``model/`` folder by the ML tab. This page
lists those bundles in a dropdown; picking one loads it and computes a prediction for every plot,
which it hands to the sub-tabs:

* **Split consistency** — per-split metric spread (held-out / training / delta) and the
  generalisation gap, read straight from the saved splits.
* **Map & 2-D/3-D view** — the field map coloured per plot (left-click z-score distribution,
  right-click "open in polyscope"), with multi-select for a combined 3-D view.
* **Model performance** — predicted-vs-actual for the loaded model's active split.

A **split dropdown** lets any saved split (not only the best) be made active; all sub-tabs follow it.

The page is a thin host: it owns the one shared polyscope viewer, loads the chosen bundle, builds
its data source (the feature workbook) and computes the table once per selection; the sub-tabs are
pure views over that table.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from common.naming import LAS_SUFFIXES, aug_number_from_name, plot_number_from_name
from common.session import Session
from ml import (
    aug_toggles_meaningful,
    bundle_info,
    has_variant_cube,
    list_bundles,
    load_bundle,
    load_dataset,
    prune_orphan_info_sidecars,
)
from .results import ViewerProcess
from .results.feature_importance import FeatureImportance
from .results.map_view import MapView
from .results.model_comparison import ModelComparison
from .results.model_performance import ModelPerformance
from .results.nested_cv_stability import NestedCvStability, summary_meta, with_target_note
from .results.split_consistency import SplitConsistency
from .results.variants import VariantSelection
from .widgets import make_status_indicator, select_and_delete, set_status_indicator
from .workers import ImportanceWorker, ResultsWorker


def effective_aug_state(setting: bool | None, had_augmented: bool | None) -> bool | None:
    """The augmented-data state to *show* for a per-model badge, given the recorded ``setting``.

    The Setup/learning tabs record whether "use augmented data" was ticked (``setting``), but ticking
    it over a folder that has no augmented plots changes nothing — no augmented copies exist to fit or
    score. So when the run had no augmented rows (``had_augmented`` False) a True setting is reported
    as False (orig-only), matching what the model actually did. ``setting`` None (legacy bundle, flag
    not recorded) stays None so it still renders the faint "unknown" badge. ``had_augmented`` None
    (legacy, unknown) leaves the setting untouched, never regressing older bundles.
    """
    if setting and had_augmented is False:
        return False
    return setting


class ResultsPage(QWidget):
    """The results module (the last tab): a QTabWidget over the results sub-tabs."""

    def __init__(self, session: Session) -> None:
        super().__init__()
        self._session = session
        self._worker: ResultsWorker | None = None
        self._importance_worker: ImportanceWorker | None = None
        self._loaded = None                               # the LoadedModel currently shown
        self._source = None                               # data source the loaded model predicts on
        self._table = None                                # cached per-plot prediction table (instant toggle refresh)
        self._features_frame = None                       # frame handed to the map view
        self._file_map: dict[tuple[int, int], Path] = {}  # (plot, aug) -> representative file
        self._viewer = ViewerProcess()                    # the one shared polyscope window

        root = QVBoxLayout(self)
        bar = QHBoxLayout()
        bar.addWidget(self._label("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.setToolTip("Saved models in this project's model/ folder.")
        self.model_combo.currentIndexChanged.connect(self._on_model_chosen)
        bar.addWidget(self.model_combo, 1)
        self.browse_button = QPushButton("Browse…")
        self.browse_button.setToolTip("Load a model bundle from anywhere on disk.")
        self.browse_button.clicked.connect(self._browse_model)
        bar.addWidget(self.browse_button)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setToolTip("Re-scan the project for saved models and recompute the current one.")
        self.refresh_button.clicked.connect(self._refresh)
        bar.addWidget(self.refresh_button)
        self.manage_button = QPushButton("Delete models…")
        self.manage_button.setToolTip("Delete specific saved model bundles from this project's model/ folder.")
        self.manage_button.clicked.connect(self._manage_models)
        bar.addWidget(self.manage_button)
        self.status = make_status_indicator()
        set_status_indicator(self.status, False, "Save a model on a learning tab, then pick it here.")
        bar.addWidget(self.status, 1)

        # Two paired columns, each an indicator stacked above its display toggle so it's obvious which
        # toggle each badge describes. The *indicator* reports a per-model fact (what the saved model
        # actually fitted / scored at train time): filled diamond = augmented copies were used, hollow
        # = originals only. Coloured blue/gray rather than green so it reads as a *setting*, not a
        # pass/fail. The *toggle* below is an independent display choice over the precomputed metric
        # cube: whether augmented copies count toward the metrics the sub-tabs show. Toggling re-picks
        # a stored cell — no recomputation — so it applies instantly across Split Consistency, Model
        # comparison and Model performance (but not the Map & 2-D/3-D view). Default: training on,
        # validation off (an honest test on real plots), independent of how the model was trained.
        self.train_aug_toggle = QCheckBox("Aug in training metrics")
        self.train_aug_toggle.setChecked(True)
        self.train_aug_toggle.setToolTip(
            "Count augmented plot copies in the training metrics shown by the results sub-tabs "
            "(Split Consistency, Model comparison, Model performance). Does not affect the map."
        )
        self.train_aug_toggle.toggled.connect(self._refresh_variant_subtabs)
        self.train_aug_indicator = self._label("")
        self._train_aug_tip = (
            "Whether this model was actually fitted on augmented plot copies "
            "(✓) or originals only (✗). Reads ✗ when the run had no augmented plots, "
            "even if 'use augmented data' was ticked — there were none to fit on."
        )
        self.train_aug_indicator.setToolTip(self._train_aug_tip)
        bar.addLayout(self._toggle_column(self.train_aug_indicator, self.train_aug_toggle))

        self.val_aug_toggle = QCheckBox("Aug in validation metrics")
        self.val_aug_toggle.setChecked(False)
        self.val_aug_toggle.setToolTip(
            "Count augmented plot copies in the held-out (validation) metrics shown by the results "
            "sub-tabs. Off by default so the score reflects real plots. Does not affect the map."
        )
        self.val_aug_toggle.toggled.connect(self._refresh_variant_subtabs)
        self.val_aug_indicator = self._label("")
        self._val_aug_tip = (
            "Whether this model's held-out validation actually scored augmented plot copies "
            "(✓) or originals only (✗). Reads ✗ when the run had no augmented plots, "
            "even if 'use augmented data' was ticked — there were none to score."
        )
        self.val_aug_indicator.setToolTip(self._val_aug_tip)
        bar.addLayout(self._toggle_column(self.val_aug_indicator, self.val_aug_toggle))
        root.addLayout(bar)

        # Split selector: every split is saved in the bundle, so the user can inspect any of them
        # (not only the best). Picking one re-runs the per-plot prediction with that split's model.
        split_bar = QHBoxLayout()
        split_bar.addWidget(self._label("Split:"))
        self.split_combo = QComboBox()
        self.split_combo.setToolTip(
            "Which training split to show — its held-out rRMSE is listed. The map and the "
            "model-performance scatter both follow this choice."
        )
        self.split_combo.currentIndexChanged.connect(self._on_split_chosen)
        split_bar.addWidget(self.split_combo, 1)
        root.addLayout(split_bar)

        self.tabs = QTabWidget()
        # The map view and the old point-cloud view are merged into one selectable map (select a
        # plot to add it to a combined 3-D view); Model performance stays its own tab.
        self.split_consistency = SplitConsistency()
        self.map_view = MapView(self._viewer)
        self.model_performance = ModelPerformance()
        # Model comparison loads its own set of bundles (it compares many models, not the one the
        # host has selected), so it takes the session to reach the project's model/ folder.
        self.model_comparison = ModelComparison(self._session)
        # Feature importance is the one analysis not served from the precomputed bundle: it re-scores
        # the active split's model, so it computes on demand off-thread (classical models only).
        self.feature_importance = FeatureImportance()
        self.feature_importance.compute_requested.connect(self._on_importance_requested)
        # Selection stability — per-fold spread for a nested-CV bundle, or the single feature/
        # hyperparameter set a plain fitted model ended up using (the degenerate one-"fold" case).
        self.nested_cv_stability = NestedCvStability()
        self._add_tab(self.split_consistency, "Split Consistency")
        self._add_tab(self.map_view, "Map & 2-D/3-D view")
        self._add_tab(self.model_performance, "Model performance")
        self._add_tab(self.model_comparison, "Model comparison")
        self._add_tab(self.feature_importance, "Feature importance")
        self._stability_tab_index = self._add_tab(self.nested_cv_stability, "Selection stability")
        self.tabs.setTabVisible(self._stability_tab_index, False)
        # Showing the map tab pulls the classical feature overlay in on demand (silently), replacing
        # the old "Load feature values…" button — feature columns appear as soon as the map is opened.
        self.tabs.currentChanged.connect(self._on_sub_tab_changed)
        root.addWidget(self.tabs, 1)

    @staticmethod
    def _label(text: str):
        from PySide6.QtWidgets import QLabel
        return QLabel(text)

    @staticmethod
    def _toggle_column(indicator, toggle) -> QVBoxLayout:
        """Stack a model-property ``indicator`` directly above its display ``toggle``.

        Keeps each badge visually glued to the checkbox it describes (training over training,
        validation over validation), with the indicator centred over the box.
        """
        from PySide6.QtCore import Qt
        column = QVBoxLayout()
        column.setSpacing(0)
        indicator.setAlignment(Qt.AlignHCenter)
        column.addWidget(indicator)
        column.addWidget(toggle)
        return column

    def _set_aug_indicator(self, label, on: bool | None, scope: str, base_tip: str) -> None:
        """Show a loaded model's per-scope augmented-data setting on ``label`` as a standalone badge.

        ``scope`` is ``"training"`` or ``"validation"``. ``on`` True -> ✓ (augmented copies were
        used), False -> ✗ (originals only), None -> a faint "unknown" badge for a legacy bundle that
        predates the per-model flag. Both badges always read the same noun (``augmented {scope}``) so
        they're consistent; the ✓/✗ symbol and blue/gray colour carry the on/off state. Coloured
        blue/gray rather than green so it reads as a *setting*, not a pass/fail. Callers pass the
        *effective* state — a run with no augmented rows reports False even if the setting was on,
        because augmented copies couldn't have been used. Use :meth:`_clear_aug_indicators` for "no
        model".
        """
        if on is None:
            label.setText(f"? augmented {scope}")
            label.setStyleSheet("color: #adb5bd; font-weight: bold;")  # faint - setting not recorded
            label.setToolTip(
                f"This model was saved before the per-model augmented-{scope} setting was recorded."
            )
            return
        if on:
            text, colour = f"✓ augmented {scope}", "#1971c2"  # blue - augmented copies used
        else:
            text, colour = f"✗ augmented {scope}", "#868e96"  # gray - originals only
        label.setText(text)
        label.setStyleSheet(f"color: {colour}; font-weight: bold;")
        label.setToolTip(base_tip)

    def _set_aug_indicators(self, train_on: bool | None, val_on: bool | None) -> None:
        """Refresh both per-model badges (training above its toggle, validation above its toggle)."""
        self._set_aug_indicator(self.train_aug_indicator, train_on, "training", self._train_aug_tip)
        self._set_aug_indicator(self.val_aug_indicator, val_on, "validation", self._val_aug_tip)

    def _clear_aug_indicators(self) -> None:
        """Blank both per-model badges (no model loaded)."""
        self.train_aug_indicator.setText("")
        self.val_aug_indicator.setText("")

    # ------------------------------------------------------------------ #
    # Augmented-data variant selection (the two display toggles)         #
    # ------------------------------------------------------------------ #
    def _current_selection(self) -> VariantSelection:
        """The augmented-data choice the variant sub-tabs read, from the two toggles.

        ``available`` is False for pre-cube bundles; the sub-tabs then show n/a and the toggles are
        greyed (see :meth:`_sync_toggle_state`).
        """
        history = self._loaded.history if self._loaded else None
        if not has_variant_cube(history):
            return VariantSelection.legacy()
        return VariantSelection.from_toggles(
            self.train_aug_toggle.isChecked(), self.val_aug_toggle.isChecked(), available=True
        )

    def _sync_toggle_state(self) -> None:
        """Enable the two aug toggles only when flipping them can actually change a metric.

        That needs a precomputed cube *and* the run having had augmented rows — an originals-only run
        builds a cube whose ``with_aug`` cells equal its ``orig_only`` cells, so the toggle would do
        nothing. ``aug_toggles_meaningful`` checks both (older bundles predate the flag and fall back to
        the cube check). When disabled we explain *why* so the greying isn't mysterious."""
        history = self._loaded.history if self._loaded else None
        meaningful = aug_toggles_meaningful(history)
        had_cube = has_variant_cube(history)
        # When the toggles can't change anything they're greyed; a greyed box that still shows a
        # checkmark reads as "augmented data is being counted" when there is none, so clear both
        # visually. While greyed the checked state can't affect any metric (no cube -> legacy(); cube
        # but no aug rows -> with_aug == orig_only), so this is purely cosmetic — re-enabling restores
        # the page defaults (training on, validation off). Block signals so the cosmetic flip doesn't
        # trigger a sub-tab refresh.
        for toggle, default in ((self.train_aug_toggle, True), (self.val_aug_toggle, False)):
            want = default if meaningful else False
            if toggle.isChecked() != want:
                blocked = toggle.blockSignals(True)
                toggle.setChecked(want)
                toggle.blockSignals(blocked)
            toggle.setEnabled(meaningful)
            if meaningful:
                toggle.setToolTip("")
            elif not had_cube:
                toggle.setToolTip(
                    "This model was saved before per-variant metrics existed — re-train and save "
                    "it to switch augmented-data choices here."
                )
            else:
                toggle.setToolTip(
                    "This run used no augmented data, so there is nothing to include or exclude — "
                    "the augmented-data choice can't change any metric here."
                )

    def _refresh_variant_subtabs(self) -> None:
        """Re-push the current selection to the variant sub-tabs without recomputing predictions.

        Reads the cached history/table only (the cube is precomputed), so toggling is instant. The
        Map & 2-D/3-D view is deliberately untouched — the toggles are about metrics, not the field map.
        """
        selection = self._current_selection()
        history = self._loaded.history if self._loaded else None
        self.split_consistency.refresh(history, selection)
        self.model_performance.refresh(history, self._table, selection)
        self.model_comparison.refresh(selection)
        # Importance isn't cube-served (it re-scores the model), so a toggle change can't be applied
        # in place — drop the stale chart and ask for a fresh Compute under the new aug choice.
        self.feature_importance.invalidate(
            "Augmented-data choice changed — press Compute to re-rank under the new setting."
        )

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #
    def close_viewer(self) -> None:
        """Shut the shared polyscope window down (called when the app closes)."""
        self._viewer.close()

    def on_enter(self) -> None:
        """Re-scan the project's saved models; auto-select and compute the newest if any."""
        self._populate_models()
        # The comparison tab keeps its own model list (it can include any saved bundle), so refresh
        # it against the same folder whenever the page is entered.
        self.model_comparison.reload_models()

    def reset_for_project(self) -> None:
        """Wipe the previous project's loaded model + cached predictions so a new one starts clean.

        Called by the shell when a project is created/opened, before on_enter re-scans the new
        project's model folder. Empties the model dropdown (so on_enter picks the new project's
        newest model, not the prior pick) and resets the loaded-model/source state via _show_empty,
        which also rebuilds the file map from the now-active project's clouds.
        """
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.blockSignals(False)
        # The comparison tab's imported models / averaging toggles / session names belong to the
        # previous project; clear them so the new project starts with only its own bundles.
        self.model_comparison.reset_imports()
        self._show_empty()

    # ------------------------------------------------------------------ #
    # Model list / selection                                             #
    # ------------------------------------------------------------------ #
    def _populate_models(self) -> None:
        """Fill the dropdown from the project's model/ folder (preserving the current pick)."""
        current = self.model_combo.currentData()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        infos = []
        if self._session.project is not None:
            infos = list_bundles(self._session.project.model_dir)
        for info in infos:
            self.model_combo.addItem(info.label, str(info.path))
        self.model_combo.blockSignals(False)

        if self.model_combo.count() == 0:
            self._show_empty()  # builds the file map; the map stays usable for inspection
            if self._file_map:
                set_status_indicator(
                    self.status, False,
                    "No saved models yet — the map is open for inspection (open plots in polyscope). "
                    "Train and save a model on a learning tab for predictions and metrics.",
                )
            else:
                set_status_indicator(
                    self.status, False, "No saved models yet — train and save one on a learning tab."
                )
            return
        # Re-select the previous pick if it survived; else the newest (index 0).
        idx = self.model_combo.findData(current) if current else -1
        self.model_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._on_model_chosen()

    def _on_model_chosen(self) -> None:
        path = self.model_combo.currentData()
        if path:
            self._load_and_compute(Path(path))

    def _browse_model(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        start = str(self._session.project.model_dir) if self._session.project else ""
        path, _ = QFileDialog.getOpenFileName(self, "Select model bundle", start, "Model bundle (*.joblib)")
        if not path:
            return
        # Add it to the combo (if not already there) and select it.
        if self.model_combo.findData(path) < 0:
            try:
                label = bundle_info(path).label
            except Exception:  # noqa: BLE001
                label = Path(path).name
            self.model_combo.addItem(label, path)
        self.model_combo.setCurrentIndex(self.model_combo.findData(path))

    def _refresh(self) -> None:
        self._populate_models()

    def _manage_models(self) -> None:
        """Delete specific saved model bundles from the project's model/ folder."""
        if self._session.project is None:
            QMessageBox.warning(self, "No project", "Open a project first.")
            return
        infos = list_bundles(self._session.project.model_dir)
        if not infos:
            QMessageBox.information(self, "Delete models", "No saved models in this project.")
            return
        removed = select_and_delete(
            self, "Delete saved models", [(info.path, info.label) for info in infos]
        )
        if removed:
            prune_orphan_info_sidecars(self._session.project.model_dir)  # drop now-orphaned .info.json
            self._populate_models()  # re-scan; re-selects a surviving model or shows empty

    # ------------------------------------------------------------------ #
    # Compute                                                            #
    # ------------------------------------------------------------------ #
    def _load_and_compute(self, path: Path) -> None:
        try:
            loaded = load_bundle(path)
        except Exception as exc:  # noqa: BLE001
            set_status_indicator(self.status, False, f"Could not load model: {exc}")
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        self._loaded = loaded
        # The data source (and the feature frame) are built lazily — only when a legacy bundle must
        # predict live, or when the 3-D viewer / distribution / feature overlay needs the files. A
        # format-4 bundle carries every split's predictions, so the map and metrics come straight
        # from the bundle with no source build and no re-prediction.
        self._source = None
        self._features_frame = None
        self._build_file_map()  # disk scan for (plot, aug) -> file; independent of the source
        self._populate_splits()
        self._sync_nested_cv_tab(loaded)
        if loaded.has_stored_predictions:
            self._compute_stored()
        else:
            self._compute_live()

    def _sync_nested_cv_tab(self, loaded) -> None:
        """Feed the selection-stability tab for any loaded model; show it whenever a model is open.

        A nested-CV bundle gets its true per-fold spread from ``vault_meta``; a plain fitted model gets
        a one-"fold" summary of the feature/hyperparameter set it ended up using (see
        :func:`gui.results.nested_cv_stability.summary_meta`). The tab is hidden only when nothing is
        loaded (handled by :meth:`clear`).
        """
        meta = getattr(loaded, "vault_meta", None) if loaded is not None else None
        is_nested = bool(getattr(loaded, "is_nested_cv", False)) and bool(meta)
        meta = dict(meta) if is_nested else summary_meta(loaded)
        # How the target was handled travels on the history (not on vault_meta), so attach it here for
        # both paths: it is the one thing about a model whose absence would let a reader mistake the
        # units of every other number on this tab.
        meta = with_target_note(meta, getattr(loaded, "history", None))
        self.nested_cv_stability.set_meta(meta)
        self.tabs.setTabVisible(self._stability_tab_index, loaded is not None)

    def _populate_splits(self) -> None:
        """Fill the split dropdown from the loaded bundle, each labelled with its rRMSE and R²."""
        self.split_combo.blockSignals(True)
        self.split_combo.clear()
        history = self._loaded.history if self._loaded else None
        splits = getattr(history, "splits", None) or []
        for sm in splits:
            rrmse = sm.metrics.get("rrmse", float("nan"))
            best = " (best)" if sm.split == history.best_split else ""
            self.split_combo.addItem(
                f"Split #{sm.split} — rRMSE {rrmse:.3g}%{best}", sm.split
            )
        # Default to whichever split the bundle marks active (its best split).
        active = getattr(history, "active_split", 0) or getattr(history, "best_split", 0)
        idx = self.split_combo.findData(active)
        self.split_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.split_combo.setEnabled(self.split_combo.count() > 1)
        self.split_combo.blockSignals(False)

    def _on_split_chosen(self) -> None:
        """Make the picked split active and rebuild its per-plot table.

        For a format-4 bundle this is an instant in-memory lookup (no worker, no source); only a
        legacy bundle re-runs the model off-thread.
        """
        if self._loaded is None:
            return
        split = self.split_combo.currentData()
        if split is None:
            return
        self._loaded.history.active_split = int(split)
        if self._loaded.has_stored_predictions:
            self._compute_stored()
        else:
            self._compute_live()

    def _compute_stored(self) -> None:
        """Build the active split's per-plot table from stored predictions — instant, on the UI thread."""
        if self._loaded is None:
            return
        try:
            table = self._loaded.stored_table()
        except Exception as exc:  # noqa: BLE001
            self._on_failed(str(exc))
            return
        self._on_table(table)

    def _compute_live(self) -> None:
        """Legacy path: build the source if needed and predict the active split off the UI thread."""
        if self._loaded is None:
            return
        try:
            self._ensure_source()
        except Exception as exc:  # noqa: BLE001
            set_status_indicator(self.status, False, f"Could not load model: {exc}")
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        set_status_indicator(
            self.status, False, f"Computing predictions with {self._loaded.model_key}…"
        )
        self.refresh_button.setEnabled(False)
        self._worker = ResultsWorker(self._loaded, self._source)
        self._worker.finished_ok.connect(self._on_table)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _ensure_source(self) -> None:
        """Build the data source on first need (idempotent). May raise FileNotFoundError.

        Sets ``_source`` and ``_features_frame``. For a format-4 bundle this is reached only when the
        user opens something that needs the actual files (it is never called just to show the map or
        the metrics).
        """
        if self._source is not None:
            return
        source = self._build_source(self._loaded)
        self._source = source
        self._features_frame = source.frame

    def _load_features_frame_quiet(self) -> bool:
        """Read the project's features workbook into ``self._features_frame`` if it's reachable.

        Model-independent and silent: the features workbook is produced by the **Features** stage, so
        it can (and should) be available for the map's colour-by / distribution / display fields even
        before any model is trained. Idempotent; returns whether a frame is now in memory. Failures
        (no path, missing/unreadable file) are swallowed — the map degrades to plot number only.
        """
        if self._features_frame is not None:
            return True
        features = self._session.feature_table_path
        if not (features and Path(features).exists()):
            return False
        from ml.dataset import _read_indexed

        try:
            self._features_frame = _read_indexed(features)
        except Exception:  # noqa: BLE001
            return False
        return True

    def _ensure_features_frame(self, *, quiet: bool = False) -> bool:
        """Load the features workbook for the classical map overlay (idempotent). Returns success.

        The per-plot table already carries actual/predicted/error, so the overlay needs only the
        feature *display* values — read straight from the features workbook (no targets needed),
        rather than building the full Dataset source. With ``quiet`` failures are swallowed; otherwise
        a dialog explains why the workbook couldn't be read.
        """
        if self._load_features_frame_quiet():
            return True
        if not quiet:
            features = self._session.feature_table_path
            if not (features and Path(features).exists()):
                QMessageBox.warning(
                    self, "Features unavailable",
                    "The project's features workbook isn't available; open the project that produced "
                    "this model to overlay feature values.",
                )
            else:
                QMessageBox.warning(self, "Features unavailable", "Could not read the features workbook.")
        return False

    def _auto_load_map_features(self) -> None:
        """Pull the feature overlay into the map when its tab is shown, if it isn't already.

        Best-effort and silent: the features workbook (when reachable) gives the map its feature /
        target columns to colour by, build distributions from and display — with or without a trained
        model. Re-pushes the current table (or ``None`` in inspection-only mode) once the frame is in.
        """
        if self._features_frame is not None:
            return
        if self._load_features_frame_quiet():
            self.map_view.refresh(self._table, self._features_frame, self._file_map,
                                  self._session.project, target_column=self._session.target_column)

    def _add_tab(self, page: QWidget, label: str) -> int:
        """Add a sub-tab, wrapped so it scrolls instead of forcing the window taller than the screen.

        A QTabWidget's layout makes every tab at least as large as the biggest one (the map view
        wants ~1165px of height), which would otherwise push the shell off the bottom of a laptop
        display. ``_tab_page`` unwraps this again for the identity checks.
        """
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(page)
        return self.tabs.addTab(scroll, label)

    def _tab_page(self, index: int) -> QWidget | None:
        """The page inside the sub-tab at ``index`` (see :meth:`_add_tab`)."""
        holder = self.tabs.widget(index)
        return holder.widget() if isinstance(holder, QScrollArea) else holder

    def _on_sub_tab_changed(self, index: int) -> None:
        """Auto-load the map's feature overlay the moment its tab becomes the visible one."""
        if self._tab_page(index) is self.map_view:
            self._auto_load_map_features()

    def _on_importance_requested(self, n_repeats: int, score_on: str, n_jobs: int) -> None:
        """Feature-importance tab asked to compute: build the dataset source, run the worker off-thread.

        Importance permutes the active split's fitted model over the feature workbook (X + y), so it
        needs the full :class:`~ml.Dataset` — the same source the legacy live-predict path builds —
        not just the display features frame. It reads the active split from ``history.active_split``
        (kept in sync by the split dropdown), so it always explains the split the rest of the page is
        showing. Seeded from the project so the ranking is reproducible. ``n_jobs`` only parallelises
        the shuffles (identical, reproducible result — see :func:`ml.permutation_importance`).
        """
        if self._loaded is None:
            return
        if self._importance_worker is not None and self._importance_worker.isRunning():
            return
        try:
            self._ensure_source()
        except Exception as exc:  # noqa: BLE001
            self.feature_importance.set_failed(str(exc))
            return
        seed = int(getattr(self._session.project, "seed", 0)) if self._session.project else 0
        include_aug = self._importance_include_aug(score_on)
        self.feature_importance.begin_compute()
        self._importance_worker = ImportanceWorker(
            self._source, self._loaded.history, n_repeats, score_on, include_aug, seed,
            n_jobs=n_jobs,
        )
        self._importance_worker.finished_ok.connect(self.feature_importance.set_result)
        self._importance_worker.failed.connect(self.feature_importance.set_failed)
        self._importance_worker.start()

    def _importance_include_aug(self, score_on: str) -> bool:
        """Whether to count augmented copies in the importance subset, per the Results-tab toggles.

        Held-out scoring follows the validation toggle, training scoring the training toggle, and the
        "all" subset (a mix of both sides) follows either toggle being on — the same originals-always,
        copies-on-toggle rule the other sub-tabs use.
        """
        val_aug = self.val_aug_toggle.isChecked()
        train_aug = self.train_aug_toggle.isChecked()
        if score_on == "held_out":
            return val_aug
        if score_on == "train":
            return train_aug
        return val_aug or train_aug  # "all"

    def _build_source(self, loaded):
        """Build the data source the loaded bundle predicts on (the feature table)."""
        features = self._session.feature_table_path
        targets = self._session.targets_table_path
        if not (features and targets and Path(features).exists() and Path(targets).exists()):
            raise FileNotFoundError(
                "This model needs the project's features + targets workbooks; "
                "open the project that produced it (or re-run feature generation)."
            )
        return load_dataset(features, targets)

    def _build_file_map(self) -> None:
        """Map (plot, aug) -> cloud file by scanning the project's cloud folders on disk.

        Built from disk rather than ``session.input_files`` (which is populated only by a live
        clip/augment run this session) so click-through and the distribution keep working after a
        project is merely re-opened. Prefers the folder used for predictions (so the map and the
        prediction source agree), falling back to the project's plots/.
        """
        self._file_map = {}
        candidates: list[Path] = []
        if self._session.augment_output_dir:
            candidates.append(Path(self._session.augment_output_dir))
        if self._session.project is not None:
            candidates.append(self._session.project.plots_dir)

        for folder in candidates:
            if not (folder and folder.is_dir()):
                continue
            for path in folder.iterdir():
                if path.suffix.lower() not in LAS_SUFFIXES:
                    continue
                plot = plot_number_from_name(path.name)
                if plot is None:
                    continue
                aug = aug_number_from_name(path.name) or 0
                self._file_map.setdefault((plot, aug), path)
            if self._file_map:
                break  # the first folder that yields any plot clouds wins

    def _on_table(self, table) -> None:
        self.refresh_button.setEnabled(True)
        n_plots = table["plot"].nunique() if len(table) else 0
        max_aug = int(table["aug"].max()) if len(table) else 0
        message = (
            f"{self._loaded.model_key}: {len(table)} predictions over {n_plots} plots. "
            f"Augmented levels: {max_aug}."
        )
        # Per-model badges: what the saved model actually fitted / scored at train time. A legacy
        # bundle predating either flag reports None -> a faint "unknown" badge (getattr default None,
        # distinct from a recorded False). The badge reports the *effective* state, not the raw
        # setting: when the run had no augmented rows (``had_augmented`` False) augmented copies
        # couldn't have been used, so both badges read off regardless of the recorded toggles —
        # ticking "use augmented data" over a folder with no augmented plots changes nothing.
        history_obj = self._loaded.history if self._loaded else None
        had_aug = getattr(history_obj, "had_augmented", None)
        train_aug = effective_aug_state(getattr(history_obj, "train_on_augmented", None), had_aug)
        val_aug = effective_aug_state(getattr(history_obj, "validate_on_augmented", None), had_aug)
        self._set_aug_indicators(train_aug, val_aug)
        set_status_indicator(self.status, True, message)
        self._table = table
        self._sync_toggle_state()
        selection = self._current_selection()
        history = self._loaded.history if self._loaded else None
        self.map_view.refresh(table, self._features_frame, self._file_map, self._session.project,
                               target_column=self._session.target_column)
        # The feature overlay needs the features workbook; pull it in automatically when the map tab
        # is already the visible one (a tab switch wouldn't fire), silently doing nothing for an
        # unreachable workbook.
        if self._tab_page(self.tabs.currentIndex()) is self.map_view:
            self._auto_load_map_features()
        self.model_performance.refresh(history, table, selection)
        self.split_consistency.refresh(history, selection)
        self.model_comparison.refresh(selection)
        # Feature importance re-runs on demand; a model/split change resets it (the previous result
        # belonged to a different model or split).
        self.feature_importance.set_model(
            is_classical=True,
            has_split_model=history is not None and bool(getattr(history, "splits", None)),
        )

    def _on_failed(self, message: str) -> None:
        self.refresh_button.setEnabled(True)
        set_status_indicator(self.status, False, f"Could not compute predictions: {message}")
        QMessageBox.critical(self, "Results failed", message)

    def _show_empty(self) -> None:
        self._loaded = None
        self._source = None
        self._table = None
        self._clear_aug_indicators()
        self._sync_toggle_state()
        self.split_combo.blockSignals(True)
        self.split_combo.clear()
        self.split_combo.blockSignals(False)
        self._features_frame = None
        selection = VariantSelection.legacy()
        # No model yet, but the map stays usable for inspection: feed it the project's clouds (built
        # from disk) with no predictions so the field map draws the plot footprints and the user can
        # still open any plot in polyscope. The features workbook is produced by the Features stage,
        # independent of any model, so load it too — its columns (and the target) are valid colour-by
        # / distribution / display fields before training. Only prediction-driven fields (model
        # error, metrics) need a model and are correctly absent here.
        self._build_file_map()
        self._load_features_frame_quiet()
        self.map_view.refresh(None, self._features_frame, self._file_map, self._session.project,
                              target_column=self._session.target_column)
        self.model_performance.refresh(None, None, selection)
        self.split_consistency.refresh(None, selection)
        self.nested_cv_stability.set_meta(None)
        self.tabs.setTabVisible(self._stability_tab_index, False)
        self.model_comparison.refresh(selection)
        self.feature_importance.set_model(is_classical=False, has_split_model=False)

"""Module 3 page: classical machine learning.

Loads the features (X) and targets (y) workbooks, lets the user pick features, a model (with a
hyperparameter form built dynamically from the model registry) and the outer/inner split settings,
then runs the whole thing as one nested cross-validation on a :class:`ValidateWorker`.

There is a single run: the **outer** folds are the honest train/held-out splits, and the **inner**
folds are where the optional hyperparameter search happens. Turning the search off makes the run a
plain cross-validation of the hyperparameters in the form; turning "try every model" on repeats the
whole procedure per model in the registry and keeps the best. Per-fold train/held-out rRMSEs are
printed to a console and plotted live on an embedded matplotlib canvas; the result is saved as a
bundle on demand.
"""

from __future__ import annotations

import os
from html import escape
from itertools import groupby
from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from PySide6.QtCore import Signal

from common.naming import plot_number_from_name
from common.session import Session
from ml import (
    MODELS,
    MODELS_BY_KEY,
    TrainConfig,
    average_split_metrics,
    average_split_train_metrics,
    load_dataset,
    save_bundle,
    training_split_stats,
)
from ml.dataset import Dataset
from .export import attach_export_menu
from .ml_help import MLHelpDialog
from .widgets import (
    make_console,
    make_status_indicator,
    open_folder,
    set_status_indicator,
    style_button,
    wire_parent_toggle,
)
from .workers import ValidateWorker

_GPR_WARN_ROWS = 1500  # GPR is O(n^3); warn above this many rows


def _split_mode_key(text: str) -> str:
    """Map a split-mode dropdown label to the engine's mode key.

    The combos show title-cased labels ("Random Systematic"); the splitting core keys them as
    lower-snake-case ("random_systematic"). Lower-case and underscore the spaces so "Random" /
    "Random Systematic" / "Sequential" / "Systematic" become the keys :data:`ml.splitting.SPLIT_MODES`
    expects.
    """
    return text.strip().lower().replace(" ", "_")


def _sync_split_count(mode_combo, count_spin, test_spin, n_groups: int) -> None:
    """Keep the split count and the split-ratio mutually consistent, per the split mode.

    The user always sets **both** the number of splits and the split ratio directly (both spinboxes
    stay enabled in every mode). How the two relate depends on the mode:

    * *Sequential* / *Systematic* / *Random Systematic* — without-replacement partitions: each fold
      holds out about ``round(ratio · n_groups)`` plots (Sequential: contiguous, Systematic: strided,
      Random Systematic: shuffled-then-contiguous), shrinking by ≤ 1 plot when needed so all ``splits``
      disjoint folds tile the field — so both the ratio AND the count are honoured, with the folds never
      overlapping (``splits · ratio ≤ 1``). The two are mutually capped (the count max is ``⌊1 / ratio⌋``
      and the ratio max ``1 / count``, via :func:`ml.splitting.max_splits_for_test_size` when the dataset
      size is known), so an invalid combination simply cannot be entered. With the product at 1 the folds
      tile the field exactly; below 1 some plots are never held out — both are valid.
    * *Random* — independent draws (overlap allowed), so the ratio is a free input and the count is
      unbounded by it; no cap is applied.

    ``n_groups`` is the distinct-plot count (0 when no dataset is loaded); the caps fall back to the
    plain ``1/ratio`` / ``1/count`` arithmetic when it is unknown.
    """
    # Remember each spinbox's design-time maximum once, so the caps below can be lifted back to the
    # widget's own intended bound (1000 for the training count, 20 for feature selection, 50 for the
    # vault, 0.9 for every ratio) rather than a single global — restoring to a wrong max would silently
    # widen or narrow a control.
    base_count_max = _base_maximum(count_spin)
    base_ratio_max = _base_maximum(test_spin)

    mode = mode_combo.currentText()
    # Both fields are always live inputs now; the only mode-dependent behaviour is whether the
    # splits·ratio ≤ 1 cap applies (every partition mode) or not (Random).
    count_spin.setEnabled(True)
    test_spin.setReadOnly(False)
    test_spin.setEnabled(True)

    if mode in ("Sequential", "Systematic", "Random Systematic"):
        # Without-replacement partition: the block size is the ratio and the folds must not overlap, so
        # cap each field against the other (the "block the input" guard the user asked for).
        _cap_split_count_and_ratio(count_spin, test_spin, n_groups, base_count_max, base_ratio_max)
    else:  # Random — independent draws may overlap, so any count/ratio combination is valid.
        count_spin.setMaximum(base_count_max)
        test_spin.setMaximum(base_ratio_max)


_BASE_MAX_PROPERTY = "_base_maximum"  # dynamic-property key caching a spinbox's design-time maximum


def _base_maximum(spin):
    """The spinbox's original (design-time) maximum, cached on first call as a dynamic property.

    ``_sync_split_count`` narrows these maxima to enforce the splits·ratio cap, so it needs the
    untouched bound to restore in the uncapped modes. Reading ``spin.maximum()`` directly would return
    an already-narrowed value on a later call; caching the first-seen maximum keeps the true bound.
    """
    cached = spin.property(_BASE_MAX_PROPERTY)
    if cached is None:
        cached = spin.maximum()
        spin.setProperty(_BASE_MAX_PROPERTY, cached)
    return cached


def _cap_split_count_and_ratio(count_spin, test_spin, n_groups, base_count_max, base_ratio_max) -> None:
    """Bound the count and ratio spinboxes so ``splits · ratio ≤ 1`` (a without-replacement partition).

    The cap is the **continuous** ``splits · ratio ≤ 1`` (``splits ≤ ⌊1/ratio⌋``, ``ratio ≤ 1/count``) —
    not the rounded ``⌊n_groups / round(ratio · n_groups)⌋``: the engine shrinks each hold-out block to
    ``⌊n_groups / splits⌋`` plots when the rounded ratio would overrun (see
    :func:`ml.splitting._block_size`), so the requested ratio AND count are always both honoured and the
    only real bound is the ratio itself. ``max_splits_for_test_size`` returns this continuous count cap;
    when the dataset isn't loaded the same ``⌊1/ratio⌋`` arithmetic is used directly. Both maxima are set
    (never above the widget's own ``base_*_max``), and a current value over its new max is clamped down,
    so the user can never hold an invalid combination. Signals are blocked while clamping so this doesn't
    recurse through the valueChanged handlers.
    """
    ratio = float(test_spin.value())
    if n_groups >= 2:
        from ml.splitting import max_splits_for_test_size
        count_max = max_splits_for_test_size(n_groups, ratio)
    else:
        count_max = max(1, int(1.0 / ratio)) if ratio > 0 else base_count_max
    count_max = min(base_count_max, max(count_spin.minimum(), count_max))

    # Cap the count first (clamping its value down if needed), THEN derive the ratio's max from the now
    # up-to-date count, so the two bounds can't fight each other on a single edit.
    count_spin.blockSignals(True)
    count_spin.setMaximum(count_max)
    count_spin.blockSignals(False)

    count = int(count_spin.value())
    # The largest ratio that still leaves room for ``count`` disjoint blocks under splits·ratio ≤ 1 is
    # the continuous ``1/count`` (the block shrinks to fit, so we don't bound by ⌊n/count⌋/n anymore).
    ratio_max = 1.0 / max(1, count)
    ratio_max = min(base_ratio_max, max(test_spin.minimum(), ratio_max))
    test_spin.blockSignals(True)
    test_spin.setMaximum(ratio_max)
    test_spin.blockSignals(False)


class MLPage(QWidget):
    """The classical-ML module UI."""

    request_next = Signal()  # ask the shell to switch to the results page

    def __init__(self, session: Session) -> None:
        super().__init__()
        self._session = session
        self._validator: ValidateWorker | None = None
        self._help_dialog = None  # held ref to the modeless help dialog while open
        # The run currently owning the Pause/Stop buttons; None when idle.
        self._active_worker = None
        self._vault_result = None  # last ml.VaultResult, for Save
        self._sweep_results = None  # last sweep's ranked list[ml.VaultResult] (None for a single run)
        # Which model of a sweep is running (1-based; 0 = not sweeping), so the live status line and
        # the graph can tell the models apart. Advanced by the worker's per-model signal.
        self._sweep_model_index = 0
        self._sweep_total = 0
        self._dataset: Dataset | None = None
        self._feature_checks: dict[str, QCheckBox] = {}
        self._normalize_checks: dict[str, QCheckBox] = {}
        self._stat_labels: dict[str, QLabel] = {}
        self._hparam_getters: dict[str, callable] = {}
        self._hparam_setters: dict[str, callable] = {}
        # Per-metric per-fold series for the tabbed graph: each maps a metric key (rrmse/r2/r/mape) to
        # the list of that metric's value per outer fold, for train and for held-out.
        self._train_series: dict[str, list[float]] = {}
        self._test_series: dict[str, list[float]] = {}

        root = QHBoxLayout(self)
        root.addWidget(self._build_data_panel(), 4)
        root.addWidget(self._build_model_panel(), 4)
        root.addWidget(self._build_run_panel(), 5)

        # The hyperparameter form now lives on the run panel, built after the model panel, so
        # populate it (and the search-space readout) once everything exists.
        self._on_model_changed()

    # ------------------------------------------------------------------ #
    # Panels                                                             #
    # ------------------------------------------------------------------ #
    def _build_data_panel(self) -> QWidget:
        box = QGroupBox("Data")
        layout = QVBoxLayout(box)

        # The features (X) and targets (y) tables are read from the project's features/ folder;
        # these read-only fields show what was loaded. Use "Reload" after re-running feature gen.
        self.features_path = QLineEdit()
        self.features_path.setReadOnly(True)
        self.targets_path = QLineEdit()
        self.targets_path.setReadOnly(True)
        layout.addWidget(QLabel("Features file (X, .csv)"))
        layout.addWidget(self.features_path)
        layout.addWidget(QLabel("Targets file (y, .csv)"))
        layout.addWidget(self.targets_path)

        reload_button = QPushButton("Reload data from project")
        reload_button.setToolTip(
            "Re-scan the project's features/ folder for the features and targets tables "
            "(use after re-running feature generation)."
        )
        reload_button.clicked.connect(self._reload_from_project)
        layout.addWidget(reload_button)

        self.data_status = QLabel("No data loaded.")
        self.data_status.setWordWrap(True)
        layout.addWidget(self.data_status)

        layout.addWidget(QLabel("Feature columns (X):"))

        # Two columns: selection (left) and standardisation (right), each "all" on
        # top of its "none".
        button_grid = QGridLayout()
        select_all = QPushButton("Select all")
        select_all.clicked.connect(lambda: self._set_all_features(True))
        select_none = QPushButton("Select none")
        select_none.clicked.connect(lambda: self._set_all_features(False))

        std_note = (
            "Standardisation only affects scale-sensitive models (linear/Lasso/Ridge/"
            "ElasticNet, SVR, KNN, GPR, PLS). Tree models (Random forest, HistGBT) are "
            "scale-invariant and never standardise, regardless of this choice."
        )
        std_all = QPushButton("Standardize all")
        std_all.setToolTip(std_note)
        std_all.clicked.connect(lambda: self._set_all_normalize(True))
        std_none = QPushButton("Standardize none")
        std_none.setToolTip(std_note)
        std_none.clicked.connect(lambda: self._set_all_normalize(False))

        button_grid.addWidget(select_all, 0, 0)
        button_grid.addWidget(select_none, 1, 0)
        button_grid.addWidget(std_all, 0, 1)
        button_grid.addWidget(std_none, 1, 1)
        layout.addLayout(button_grid)

        self._feature_container = QWidget()
        self._feature_layout = QVBoxLayout(self._feature_container)
        self._feature_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._feature_container)
        layout.addWidget(scroll, 1)
        return box

    @staticmethod
    def _scroll_tab(build) -> QWidget:
        """Wrap a tab body builder in a scroll area, so a tall form stays usable on small screens.

        ``build(layout)`` populates the given ``QVBoxLayout`` with the tab's content; a trailing
        stretch is added so the clusters sit at the top rather than spreading to fill the height.
        """
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        build(layout)
        layout.addStretch(1)
        scroll.setWidget(inner)
        return scroll

    @staticmethod
    def _cluster_header(layout: QVBoxLayout, text: str, *, top_gap: int = 10, return_label: bool = False):
        """Add a concise, bold cluster header with a little breathing room above it.

        Used to group related controls (e.g. the cross-validation split fields) under a short title
        with visible spacing between clusters — instead of nesting them in their own group boxes,
        which would clutter the dense forms. Pass ``return_label=True`` to get the :class:`QLabel`
        back so a caller can show/hide the whole cluster (used by the optional 'Iteration limits'
        cluster, which only some models have).
        """
        if top_gap:
            layout.addSpacing(top_gap)
        label = QLabel(text)
        label.setStyleSheet("font-weight: bold;")
        layout.addWidget(label)
        return label if return_label else None

    def _build_model_panel(self) -> QWidget:
        """The middle column: one panel for the whole run — model, hyperparameters, splits.

        There is a single procedure here (nested cross-validation), so there are no sub-tabs any more.
        The clusters read top-to-bottom in the order the run uses them: which model (or all of them),
        what hyperparameters (typed in, or searched for), how the outer folds and the inner search
        folds are cut, and whether augmented rows count. The console, graph and Save live on the right,
        shared as before.
        """
        return self._scroll_tab(self._add_run_controls)

    def _add_run_controls(self, layout: QVBoxLayout) -> None:
        intro = QLabel(
            "One run: the field is partitioned into OUTER folds, and for each one the model is built "
            "on that fold's training plots only and scored on the plots it never saw — an honest "
            "estimate plus a per-fold spread. When the hyperparameter search is on, it runs on the "
            "INNER folds inside each outer fold, so the tuning never sees the held-out plots either."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: gray;")
        layout.addWidget(intro)

        # --- model ---
        self._cluster_header(layout, "Model", top_gap=8)
        self.model_combo = QComboBox()
        for m in MODELS:
            self.model_combo.addItem(m.label, m.key)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        layout.addWidget(self.model_combo)

        # The sweep: run the *same* configured procedure once per model in the registry and keep the
        # best. It replaces the model choice (not the hyperparameters), so the combo greys out; each
        # model is scored from its own registry defaults, re-tuned per fold when the search is on.
        self.sweep_check = QCheckBox("Try every model in sequence and keep the best")
        self.sweep_check.setToolTip(
            f"Run the whole procedure below once for each of the {len(MODELS)} models, under the same "
            "features, splits and augmented-data settings, then keep the one with the lowest mean "
            "held-out rRMSE (a ranked comparison table is printed to the console). Each model starts "
            "from its own defaults — leave the hyperparameter search ON so every model is judged at "
            "its own best, not at whatever is typed in the form. This costs one full run per model."
        )
        self.sweep_check.toggled.connect(self._on_sweep_toggled)
        layout.addWidget(self.sweep_check)

        self.model_tooltip = QLabel("")
        self.model_tooltip.setWordWrap(True)
        layout.addWidget(self.model_tooltip)
        self.gpr_warning = QLabel("")
        self.gpr_warning.setWordWrap(True)
        self.gpr_warning.setStyleSheet("color: #b00;")
        layout.addWidget(self.gpr_warning)

        # --- hyperparameters: typed in, or searched for ---
        self._cluster_header(layout, "Hyperparameters")
        self.optimize_check = QCheckBox("Search for the best hyperparameters automatically")
        self.optimize_check.setChecked(True)  # the default: the honest, tuned path
        self.optimize_check.setToolTip(
            "On (default): an Optuna (Bayesian TPE) search re-tunes the hyperparameters inside every "
            "outer fold, over the inner cross-validation folds — so the tuning never sees that fold's "
            "held-out plots and the reported error stays honest. The form below is then ignored.\n"
            "Off: every fold uses exactly the values in the form, and the run is a plain honest "
            "cross-validation of that fixed pipeline."
        )
        self.optimize_check.toggled.connect(self._on_optimize_toggled)
        layout.addWidget(self.optimize_check)

        # Read-only summary of the auto-derived search space, so what the search tunes is never hidden.
        space_label = QLabel("")
        space_label.setWordWrap(True)
        space_label.setStyleSheet("color: gray;")
        self.search_space_label = space_label
        layout.addWidget(space_label)

        search_form = QFormLayout()
        self.n_trials = QSpinBox()
        self.n_trials.setRange(5, 2000)
        self.n_trials.setValue(50)
        self.n_trials.setToolTip(
            "How many hyperparameter combinations the search tries — per outer fold, so the total cost "
            "is trials × outer folds (× models, when sweeping)."
        )
        self.opt_threads = QSpinBox()
        cores = os.cpu_count() or 1
        self.opt_threads.setRange(1, cores)
        self.opt_threads.setValue(max(1, cores - 2))
        self.opt_threads.setToolTip(
            f"CPU threads used to score each trial's inner folds in parallel ({cores} available). The "
            "folds are independent fits, so this only changes speed, not the result."
        )
        search_form.addRow("Trials", self.n_trials)
        search_form.addRow("CPU threads", self.opt_threads)
        self._search_form = search_form
        layout.addLayout(search_form)

        # The manual form (greyed out while the search is on — it is what the search would overwrite).
        self._hparam_box = QWidget()
        self._hparam_form = QFormLayout(self._hparam_box)
        self._hparam_form.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._hparam_box)

        # --- iteration limits (only for models with an iterative solver cap, e.g. Lasso/Elastic Net).
        # The whole cluster (header + note + form) is hidden for models that have no cap, so it never
        # adds noise for Ridge/RF/etc. _rebuild_hparams populates and shows/hides it per model.
        self._iter_header = self._cluster_header(layout, "Iteration limits", return_label=True)
        self._iter_note = QLabel(
            "This model is fit by an iterative solver. If it can't settle within the cap below it stops "
            "early and logs a convergence warning to the console — raise the cap (or loosen the "
            "tolerance) if you see one."
        )
        self._iter_note.setWordWrap(True)
        self._iter_note.setStyleSheet("color: gray;")
        layout.addWidget(self._iter_note)
        self._iter_box = QWidget()
        self._iter_form = QFormLayout(self._iter_box)
        self._iter_form.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._iter_box)

        # --- outer folds (the honest held-out partition) ---
        # The OUTER loop is a without-replacement partition (no plot held out twice) — Sequential
        # (contiguous blocks), Systematic (every n-th plot, interleaved) or Random Systematic (shuffled
        # blocks); plain Random is excluded (its draws overlap, so it isn't a clean partition). The user
        # sets BOTH the fold count and the ratio (the block size); the two are mutually capped so
        # splits·ratio ≤ 1, and with the product at 1 every plot is held out exactly once.
        self._cluster_header(layout, "Outer folds (held-out partition)")
        outer_form = QFormLayout()
        self.outer_test_size = QDoubleSpinBox()
        self.outer_test_size.setRange(0.05, 0.9)
        self.outer_test_size.setSingleStep(0.05)
        self.outer_test_size.setValue(0.25)
        self.outer_test_size.setToolTip(
            "Fraction of plots held out in each OUTER fold (the block size, in plots). A real input in "
            "all three modes, bounded so splits · ratio ≤ 1. With the product at 1 the folds tile the "
            "field exactly (every plot held out once)."
        )
        self.outer_test_size.valueChanged.connect(self._refresh_feature_stats)
        self.outer_split_mode = QComboBox()
        self.outer_split_mode.addItems(["Sequential", "Systematic", "Random Systematic"])
        self.outer_split_mode.setToolTip(
            "How the outer loop partitions the field — all three are without-replacement partitions (no "
            "plot held out twice). Sequential: contiguous blocks. Systematic: every n-th plot "
            "(interleaved), so each held-out fold samples the whole field — the better choice under a "
            "spatial gradient along the plot order. Random Systematic: shuffled hold-out blocks. All "
            "bound the ratio so splits·ratio ≤ 1. Plain Random is not offered: its overlapping draws "
            "aren't a clean partition, so the estimate wouldn't be honest."
        )
        self.outer_n_splits = QSpinBox()
        self.outer_n_splits.setRange(1, 50)
        self.outer_n_splits.setValue(4)
        self.outer_n_splits.setToolTip(
            "How many OUTER folds: each holds out round(ratio · plots) plots without overlap, so "
            "splits · ratio ≤ 1. You set this and the ratio freely; with the product at 1 the folds "
            "tile the field exactly."
        )
        outer_form.addRow("Split ratio", self.outer_test_size)
        outer_form.addRow("Split mode", self.outer_split_mode)
        outer_form.addRow("Splits", self.outer_n_splits)
        layout.addLayout(outer_form)

        # --- inner cross-validation (where the hyperparameter search is scored) ---
        self._cluster_header(layout, "Inner cross-validation (per fold)")
        self.inner_note = QLabel("")
        self.inner_note.setWordWrap(True)
        self.inner_note.setStyleSheet("color: gray;")
        layout.addWidget(self.inner_note)
        inner_form = QFormLayout()
        self.inner_test_size = QDoubleSpinBox()
        self.inner_test_size.setRange(0.05, 0.9)
        self.inner_test_size.setSingleStep(0.05)
        self.inner_test_size.setValue(0.25)
        self.inner_test_size.setToolTip(
            "Fraction of (outer-train) plots held out per INNER cross-validation split — a free input "
            "in Random mode; in the three partition modes it is the block size, bounded so "
            "splits · ratio ≤ 1."
        )
        self.inner_split_mode = QComboBox()
        # Sequential first (the default): the inner CV then tiles each fold's outer-train plots too, so
        # the whole procedure is deterministic end to end. Random is allowed here (unlike the outer
        # loop): the inner loop only needs leakage-free grouped folds, not a strict partition, so its
        # noise-averaging is a legitimate defence.
        self.inner_split_mode.addItems(["Sequential", "Systematic", "Random Systematic", "Random"])
        self.inner_split_mode.setToolTip(
            "How the hyperparameter search's cross-validation splits the outer-train plots. Sequential "
            "(default): contiguous blocks. Systematic: every n-th plot (interleaved). Random "
            "Systematic: shuffled hold-out blocks without replacement. The three partition modes bound "
            "the ratio so splits·ratio ≤ 1. Random: fresh, independent grouped splits (may overlap — "
            "useful inner noise-averaging, ratio unbounded)."
        )
        self.inner_n_splits = QSpinBox()
        self.inner_n_splits.setRange(1, 50)
        self.inner_n_splits.setValue(5)
        self.inner_n_splits.setToolTip(
            "How many INNER cross-validation splits each trial is scored over, per outer fold."
        )
        inner_form.addRow("Split ratio", self.inner_test_size)
        inner_form.addRow("Split mode", self.inner_split_mode)
        inner_form.addRow("Splits", self.inner_n_splits)
        self._inner_form = inner_form
        layout.addLayout(inner_form)

        for w in (self.outer_n_splits, self.outer_test_size, self.inner_n_splits,
                  self.inner_test_size):
            w.valueChanged.connect(self._sync_split_controls)
        for w in (self.outer_split_mode, self.inner_split_mode):
            w.currentIndexChanged.connect(self._sync_split_controls)
        self.outer_split_mode.currentIndexChanged.connect(self._refresh_feature_stats)

        # --- target transform ---
        self._cluster_header(layout, "Target")
        self.log_target_check = QCheckBox("Fit on the log of the target")
        self.log_target_check.setChecked(False)
        self.log_target_check.setToolTip(
            "Fit every fold on log1p(target) instead of the raw target — the right shape for a "
            "multiplicative, right-skewed quantity like biomass, where the error grows with the plot. "
            "Predictions are back-transformed, so every reported metric (here and on the Results tab) "
            "stays in the target's own units. log1p, not log, so exact zeros are allowed."
        )
        self.log_bias_mode = QComboBox()
        self.log_bias_mode.addItems(["Smearing (Duan)", "Plain exponential"])
        self.log_bias_mode.setToolTip(
            "How the back-transform undoes the exponential's bias. Exponentiating a log-scale "
            "prediction gives the conditional MEDIAN, which under-predicts the mean by roughly "
            "exp(sigma^2/2) — a systematic shortfall that lands straight in rRMSE and MAPE. Smearing "
            "(Duan) multiplies by mean(exp(training residual)), a non-parametric estimate of exactly "
            "that factor, computed on each fold's training rows only. Plain exponential is the "
            "uncorrected, knowingly low version — for comparison."
        )
        bias_form = QFormLayout()
        bias_form.setContentsMargins(0, 0, 0, 0)
        bias_form.addRow("Back-transform", self.log_bias_mode)
        layout.addWidget(self.log_target_check)
        layout.addLayout(bias_form)
        self.log_target_check.toggled.connect(self.log_bias_mode.setEnabled)
        self.log_bias_mode.setEnabled(False)

        # --- augmented data ---
        self._cluster_header(layout, "Augmented data")
        self.use_aug_fit = QCheckBox("Use augmented data when fitting")
        self.use_aug_fit.setChecked(False)
        self.use_aug_fit.setToolTip(
            "When on, each fold is fit on augmented + original rows. When off, only original plots are "
            "fit. Applies to the held-out fits and to the inner search; toggling recomputes the feature "
            "ranges & R shown on the left."
        )
        self.use_aug_fit.toggled.connect(self._refresh_feature_stats)
        self.use_aug_val = QCheckBox("Use augmented data when scoring held-out plots")
        self.use_aug_val.setChecked(False)
        self.use_aug_val.setToolTip(
            "When off, the held-out plots are scored on original (non-augmented) rows only — an honest "
            "estimate on real data. When on, augmented held-out rows are scored too."
        )
        layout.addWidget(self.use_aug_fit)
        layout.addWidget(self.use_aug_val)

        layout.addSpacing(10)
        self.run_button = QPushButton("Run")
        self.run_button.setToolTip(
            "Run the procedure: for every outer fold, (optionally) search hyperparameters on the inner "
            "folds, fit, and score the held-out plots. The mean held-out rRMSE is the honest number; "
            "Save on the right ships it as a model bundle."
        )
        self.run_button.clicked.connect(self._on_run)
        style_button(self.run_button, "primary")
        layout.addWidget(self.run_button)

        # A clear "auto run in progress" banner — hidden unless a run is active.
        self.auto_run_banner = QLabel("● Auto run in progress")
        self.auto_run_banner.setStyleSheet(
            "color: white; background: #1f77b4; padding: 4px 8px; border-radius: 4px; font-weight: bold;"
        )
        self.auto_run_banner.setVisible(False)
        layout.addWidget(self.auto_run_banner)

        self._sync_split_controls()
        self._on_optimize_toggled(self.optimize_check.isChecked())

    def _on_sweep_toggled(self, on: bool) -> None:
        """Grey the model combo while the sweep owns the model choice."""
        self.model_combo.setEnabled(not on)
        self.run_button.setText("Run sweep" if on else "Run")

    def _on_optimize_toggled(self, on: bool) -> None:
        """Swap between the searched and the typed-in hyperparameter paths.

        With the search on, the manual form is disabled (it is exactly what the search would overwrite,
        so leaving it live would imply the typed values matter) and the trials/threads inputs appear.
        With it off, the form is the run's hyperparameters and the search cost inputs are hidden — as
        is the inner-CV cluster's reason for existing, so its note says so rather than hiding controls
        the engine still nominally accepts.
        """
        self._hparam_box.setEnabled(not on)
        self._iter_box.setEnabled(not on)
        self._set_form_row_visible(self._search_form, self.n_trials, on)
        self._set_form_row_visible(self._search_form, self.opt_threads, on)
        self.search_space_label.setVisible(on)
        for w in (self.inner_test_size, self.inner_split_mode, self.inner_n_splits):
            w.setEnabled(on)
        self.inner_note.setText(
            "Where each outer fold's hyperparameter search is scored."
            if on else
            "Not used: with the search off, every fold uses the hyperparameters in the form above."
        )

    def _on_show_help(self) -> None:
        """Open the consolidated ML help dialog (modeless, so it can sit beside the tab)."""
        dialog = MLHelpDialog(parent=self)
        self._help_dialog = dialog  # keep a ref so the modeless dialog isn't garbage-collected
        dialog.show()

    def _set_form_row_visible(self, form, field_widget, visible: bool) -> None:
        """Show/hide a whole ``QFormLayout`` row (its field widget AND its label) by the field."""
        field_widget.setVisible(visible)
        label = form.labelForField(field_widget)
        if label is not None:
            label.setVisible(visible)

    # The metrics the graph can show, as (key, label, y-label, y-limits). Each run fills the per-split
    # train/test series for these keys; the graph tabs switch between them. rRMSE/MAPE are percentages
    # (0–100), R² is 0–1, R is −1–1.
    _GRAPH_METRICS = (
        ("rrmse", "rRMSE", "rRMSE (%)", (0, 100)),
        ("r2", "R²", "R²", (0, 1)),
        ("r", "R", "R", (-1, 1)),
        ("mape", "MAPE", "MAPE (%)", (0, 100)),
    )

    def _build_run_panel(self) -> QWidget:
        box = QGroupBox("Run")
        layout = QVBoxLayout(box)

        # Header row: the readiness indicator on the left, the Info button on the right (it opens the
        # consolidated explanation of the whole tab — the run, the graph and the console).
        header = QHBoxLayout()
        self.indicator = make_status_indicator()
        header.addWidget(self.indicator, 1)
        info_button = QPushButton("Info")
        info_button.setToolTip(
            "Open a full explanation of the ML tab (the model choice, the hyperparameter "
            "search, the outer/inner splits and how they fit together)."
        )
        info_button.clicked.connect(self._on_show_help)
        header.addWidget(info_button)
        layout.addLayout(header)

        self.output_label = QLabel("Output: (no project open)")
        self.output_label.setWordWrap(True)
        self.output_label.setStyleSheet("color: gray;")
        layout.addWidget(self.output_label)

        # One matplotlib canvas shared across the four metric tabs (cheaper than four canvases): the
        # tab bar only chooses which metric is drawn into the single axes; the run fills the per-fold
        # series as its outer folds land.
        self.figure = Figure(figsize=(4, 2.6), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.graph_tabs = QTabWidget()
        for key, label, _ylabel, _ylim in self._GRAPH_METRICS:
            holder = QWidget()  # an empty placeholder tab; the shared canvas is re-parented on switch
            QVBoxLayout(holder).setContentsMargins(0, 0, 0, 0)
            self.graph_tabs.addTab(holder, label)
        self.graph_tabs.currentChanged.connect(self._on_graph_tab_changed)
        layout.addWidget(self.graph_tabs)
        # Mount the canvas into the first tab and draw the empty axes.
        self._mount_canvas(0)
        attach_export_menu(self.canvas, self)
        self._reset_plot()

        # --- run controls. Pause/Stop dispatch to the active worker; the progress bar is filled by the
        # per-fold handler. They live here in the Run box, just above the console, so the controls and
        # the live output sit together.
        controls = QHBoxLayout()
        self.pause_button = QPushButton("Pause")
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self._on_pause)
        controls.addWidget(self.pause_button)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._on_stop)
        style_button(self.stop_button, "clear")
        controls.addWidget(self.stop_button)
        layout.addLayout(controls)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        # The console. Its header carries a "Show warnings" toggle and a small red bin that clears it
        # (with a warning), top-right.
        self.show_warnings_check = QCheckBox("Show warnings")
        self.show_warnings_check.setChecked(True)
        self.show_warnings_check.setToolTip(
            "When on, warnings raised during a run (e.g. solver convergence, degenerate folds) are "
            "shown in the console, de-duplicated. Turn off to silence them — the run is unaffected, "
            "only the console output. Takes effect for warnings raised after you toggle it."
        )

        # A single live status line above the console: it shows the run's progress (model / fold / trial
        # counts) and is overwritten in place, so it gives quick feedback while a run is hammering —
        # without flooding the scrolling console. Muted grey + italic so it reads as
        # an unobtrusive status, not a log line; cleared when a run ends.
        self.status_line = QLabel("")
        self.status_line.setStyleSheet("color: #888; font-style: italic;")
        self.status_line.setToolTip(
            "Live status of the run (which model / fold / trial it is on). Updates in place."
        )
        layout.addWidget(self.status_line)

        # The shared console block (bold header, the warnings toggle, a red bin) wrapping this
        # tab's rich-text browser — the sweep table is written as HTML, so it can't be a plain log.
        console_box, self.console = make_console(
            self, QTextBrowser(), extra_header=[self.show_warnings_check]
        )
        self.console.setOpenLinks(False)
        self.console.setOpenExternalLinks(False)
        layout.addWidget(console_box, 1)

        # Save: a name field (pre-filled with a suggested name when a run finishes) plus the button
        # that writes the finished run's model (a sweep's winner) as a bundle.
        save_row = QHBoxLayout()
        self.save_name = QLineEdit()
        self.save_name.setPlaceholderText("Run the procedure to enable saving…")
        save_row.addWidget(self.save_name, 1)
        self.save_button = QPushButton("Save model")
        self.save_button.setEnabled(False)
        self.save_button.setToolTip(
            "Save the finished run's model — a sweep saves its winner — as a bundle in the project's "
            "model/ folder, under the name on the left. Choose it on the Results tab."
        )
        self.save_button.clicked.connect(self._on_save)
        style_button(self.save_button, "primary")
        save_row.addWidget(self.save_button)
        layout.addLayout(save_row)

        self.open_folder_button = QPushButton("Open output folder")
        self.open_folder_button.setEnabled(False)
        self.open_folder_button.clicked.connect(self._on_open_folder)
        layout.addWidget(self.open_folder_button)

        self.results_button = QPushButton("Go to results →")
        self.results_button.setEnabled(False)
        self.results_button.setToolTip(
            "Open the field map of per-plot prediction errors for the model just built."
        )
        self.results_button.clicked.connect(lambda: self.request_next.emit())
        style_button(self.results_button, "next")
        layout.addWidget(self.results_button)
        return box

    def _mount_canvas(self, index: int) -> None:
        """Re-parent the single shared canvas into the graph tab at ``index`` (so it shows there)."""
        holder = self.graph_tabs.widget(index)
        if holder is not None and self.canvas.parent() is not holder:
            holder.layout().addWidget(self.canvas)

    def _on_graph_tab_changed(self, index: int) -> None:
        """Move the shared canvas to the newly-selected metric tab and redraw that metric."""
        self._mount_canvas(index)
        self._redraw_plot()

    def _current_graph_metric(self) -> str:
        """The metric key (rrmse/r2/r/mape) for the currently-selected graph tab."""
        idx = self.graph_tabs.currentIndex() if hasattr(self, "graph_tabs") else 0
        return self._GRAPH_METRICS[max(0, idx)][0]

    # ------------------------------------------------------------------ #
    # Model / hyperparameter form                                        #
    # ------------------------------------------------------------------ #
    def _current_model_key(self) -> str:
        return self.model_combo.currentData()

    def _n_groups(self) -> int:
        """Distinct plot count in the loaded dataset (for the derived sequential split count)."""
        if self._dataset is None:
            return 0
        import numpy as np
        return int(len(np.unique(self._dataset.frame.index.map(plot_number_from_name))))

    def _sync_split_controls(self) -> None:
        """Keep the outer and inner split count/ratio pairs mutually consistent with their modes.

        The outer partition divides the field into the user's chosen number of folds — its mode combo
        offers Sequential (contiguous blocks), Systematic (every n-th plot) or Random Systematic
        (shuffled blocks); plain Random is not offered, since the outer loop must hold every plot out at
        most once. The inner CV additionally allows Random. In every partition mode the count and ratio
        are mutually capped so splits·ratio ≤ 1; in Random they are both free. ``_sync_split_count``
        enforces all of these.
        """
        _sync_split_count(
            self.outer_split_mode, self.outer_n_splits, self.outer_test_size, self._n_groups()
        )
        _sync_split_count(
            self.inner_split_mode, self.inner_n_splits, self.inner_test_size, self._n_groups()
        )

    def _inner_split_mode_value(self) -> str:
        # sequential / systematic / random_systematic / random
        return _split_mode_key(self.inner_split_mode.currentText())

    def _outer_split_mode_value(self) -> str:
        # sequential / systematic / random_systematic
        return _split_mode_key(self.outer_split_mode.currentText())

    def _target_transform_value(self) -> str:
        """The engine key for the target-transform choice ("none" / "log")."""
        from ml.target_transform import LOG, NONE

        return LOG if self.log_target_check.isChecked() else NONE

    def _target_bias_value(self) -> str:
        """The engine key for the log back-transform ("smearing" / "naive")."""
        from ml.target_transform import NAIVE, SMEARING

        return SMEARING if self.log_bias_mode.currentIndex() == 0 else NAIVE

    def _project_seed(self) -> int:
        """The one project-wide random seed (set on the Setup tab); 0 when no project is open."""
        return int(self._session.project.seed) if self._session.project else 0

    def _on_model_changed(self) -> None:
        """Rebuild the hyperparameter form and refresh the search-space readout for the new model."""
        self._rebuild_hparams()
        self._refresh_search_space()

    def _refresh_search_space(self) -> None:
        """Show the search's (auto-derived) space for the current model + feature count."""
        from ml.optimize import classical_search_space_text

        n_features = len(self._selected_features()) if self._dataset is not None else None
        text = classical_search_space_text(self._current_model_key(), n_features)
        self.search_space_label.setText(text or "This model has no tunable hyperparameters.")

    def _rebuild_hparams(self) -> None:
        while self._hparam_form.rowCount():
            self._hparam_form.removeRow(0)
        while self._iter_form.rowCount():
            self._iter_form.removeRow(0)
        self._hparam_getters = {}
        model = MODELS_BY_KEY[self._current_model_key()]
        self.model_tooltip.setText(model.tooltip)
        self.gpr_warning.setText(
            "Note: Gaussian Process is O(n^3) - slow/heavy on large augmented datasets."
            if model.key == "gpr" else ""
        )
        self._hparam_setters = {}
        # Core hyperparameters go under the "Hyperparameters" header; iteration limits (max_iter/tol,
        # present only on iterative-solver models like Lasso/Elastic Net) go under their own header
        # below. Both register in the same getter/setter maps, so _collect_params() still returns the
        # full param dict — the split is purely visual.
        for h in model.core_hparams():
            widget, getter, setter = self._make_hparam_widget(h)
            widget.setToolTip(h.tooltip)
            self._hparam_form.addRow(h.label, widget)
            self._hparam_getters[h.name] = getter
            self._hparam_setters[h.name] = setter
        iter_hparams = model.iteration_limit_hparams()
        for h in iter_hparams:
            widget, getter, setter = self._make_hparam_widget(h)
            widget.setToolTip(h.tooltip)
            self._iter_form.addRow(h.label, widget)
            self._hparam_getters[h.name] = getter
            self._hparam_setters[h.name] = setter
        # Hide the whole "Iteration limits" cluster for models that have no iterative cap.
        has_iter = bool(iter_hparams)
        self._iter_header.setVisible(has_iter)
        self._iter_note.setVisible(has_iter)
        self._iter_box.setVisible(has_iter)

    @staticmethod
    def _make_hparam_widget(h):
        """Return ``(widget, getter, setter)`` for one hyperparameter control."""
        if h.kind == "int":
            w = QSpinBox()
            w.setRange(int(h.min if h.min is not None else 0),
                       int(h.max if h.max is not None else 1_000_000))
            if h.step:
                w.setSingleStep(int(h.step))
            w.setValue(int(h.default))
            return w, w.value, (lambda v, w=w: w.setValue(int(round(float(v)))))
        if h.kind == "float":
            w = QDoubleSpinBox()
            lo = float(h.min) if h.min is not None else 0.0
            hi = float(h.max) if h.max is not None else 1e9
            decimals = 10 if abs(float(h.default)) < 1e-2 and h.default != 0 else 4
            w.setDecimals(decimals)
            w.setRange(lo, hi)
            w.setSingleStep(float(h.step) if h.step else max(10 ** -decimals, (hi - lo) / 100))
            w.setValue(float(h.default))
            return w, w.value, (lambda v, w=w: w.setValue(float(v)))
        if h.kind == "choice":
            w = QComboBox()
            w.addItems([str(c) for c in h.choices])
            w.setCurrentText(str(h.default))
            return w, w.currentText, (lambda v, w=w: w.setCurrentText(str(v)))
        # bool
        w = QCheckBox()
        w.setChecked(bool(h.default))
        return w, w.isChecked, (lambda v, w=w: w.setChecked(bool(v)))

    def _collect_params(self) -> dict:
        return {name: getter() for name, getter in self._hparam_getters.items()}

    def _apply_params(self, params: dict) -> None:
        """Write optimised hyperparameters back into the form widgets."""
        for name, setter in self._hparam_setters.items():
            if name in params:
                setter(params[name])

    # ------------------------------------------------------------------ #
    # Data                                                               #
    # ------------------------------------------------------------------ #
    def _resolve_workbooks(self) -> tuple[Path | None, Path | None]:
        """Find the features (X) and targets (y) tables for the active project.

        Prefers the paths the feature tab handed over in the session; otherwise scans the
        project's features/ folder for a ``*feature*.csv`` and a matching ``*target*.csv``.
        """
        features = self._session.feature_table_path
        targets = self._session.targets_table_path
        if features and Path(features).exists():
            return Path(features), (Path(targets) if targets and Path(targets).exists() else None)
        project = self._session.project
        if project is None:
            return None, None
        folder = project.features_dir
        candidates = [
            p for p in sorted(folder.glob("*.csv"))
            if "feature" in p.name.lower()
            and "target" not in p.name.lower()
            and "manifest" not in p.name.lower()
        ]
        if not candidates:
            return None, None
        feat = candidates[0]
        tgt = next((p for p in sorted(folder.glob("*.csv")) if "target" in p.name.lower()), None)
        return feat, tgt

    def _reload_from_project(self) -> None:
        """Re-scan the project for workbooks and load the dataset."""
        features, targets = self._resolve_workbooks()
        if features is None or targets is None:
            self.data_status.setText(
                "No features + targets workbooks found in the project's features/ folder. "
                "Generate features with a target column on the Feature tab first."
            )
            self.features_path.setText(str(features) if features else "")
            self.targets_path.setText("")
            return
        self.features_path.setText(str(features))
        self.targets_path.setText(str(targets))
        self._on_load()

    def _on_load(self) -> None:
        fpath = self.features_path.text().strip()
        tpath = self.targets_path.text().strip()
        if not fpath or not tpath:
            return
        try:
            dataset = load_dataset(fpath, tpath)
        except Exception as exc:  # noqa: BLE001
            self.data_status.setText(f"Load failed: {exc}")
            self._refresh_indicator()
            return
        self._dataset = dataset
        self._session.target_column = dataset.target_column
        self._rebuild_feature_checks(dataset.feature_columns)
        self.data_status.setText(
            f"Loaded {len(dataset.frame)} row(s). Target: '{dataset.target_column}'. "
            f"{len(dataset.feature_columns)} feature column(s)."
        )
        self._refresh_search_space()
        self._sync_split_controls()
        self._refresh_indicator()

    def _rebuild_feature_checks(self, columns: list[str]) -> None:
        while self._feature_layout.count() > 1:
            item = self._feature_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._feature_checks = {}
        self._normalize_checks = {}
        self._stat_labels = {}

        # Group the loaded columns by the same class -> type structure the feature tab uses,
        # so the ML feature list is browsable and class/type headers can toggle selection in
        # bulk. Columns not in the feature registry (imported reference / one-hot columns) fall
        # into a trailing "Imported / other" class so nothing is ever dropped.
        for cls, types in self._grouped_columns(columns):
            cls_box = QGroupBox(cls)
            cls_box.setCheckable(True)
            cls_box.setChecked(True)
            cls_layout = QVBoxLayout(cls_box)
            cls_children: list[QCheckBox] = []
            single_type = len(types) == 1
            for type_name, cols in types:
                type_children: list[QCheckBox] = []
                # Show a type header only when the class has >1 type and the type is named
                # (the "Imported / other" bucket has an empty type name).
                show_header = bool(type_name) and not single_type
                if show_header:
                    header = QCheckBox(type_name)
                    header.setChecked(True)
                    header.setStyleSheet("font-weight: bold;")
                    cls_layout.addWidget(header)
                for col in cols:
                    row = QWidget()
                    h = QHBoxLayout(row)
                    h.setContentsMargins(16 if show_header else 0, 0, 0, 0)

                    select = QCheckBox(col)
                    select.setChecked(True)
                    select.toggled.connect(self._refresh_search_space)  # PLS range tracks count
                    self._feature_checks[col] = select

                    stat = QLabel("")
                    stat.setStyleSheet("color: gray;")
                    stat.setToolTip(
                        "[min – max] and direct R (signed Pearson) on the training split "
                        "(display only)."
                    )
                    self._stat_labels[col] = stat

                    normalize = QCheckBox("standardize")
                    normalize.setChecked(True)
                    normalize.setToolTip(
                        "Standardise this feature (mean 0, unit variance), re-fit on the "
                        "training rows of each fold. Only affects scale-sensitive models "
                        "(linear/Lasso/Ridge/ElasticNet, SVR, KNN, GPR, PLS). Tree models "
                        "(Random forest, HistGBT) are scale-invariant and never standardise, "
                        "regardless of this choice."
                    )
                    self._normalize_checks[col] = normalize

                    h.addWidget(select, 1)
                    h.addWidget(stat)
                    h.addWidget(normalize)
                    cls_layout.addWidget(row)
                    type_children.append(select)
                    cls_children.append(select)
                if show_header:
                    wire_parent_toggle(header, type_children)
            wire_parent_toggle(cls_box, cls_children)
            self._feature_layout.insertWidget(self._feature_layout.count() - 1, cls_box)

        self._refresh_feature_stats()

    @classmethod
    def _grouped_columns(cls, columns: list[str]):
        """Order the loaded columns into ``[(class, [(type, [col, ...]), ...]), ...]``.

        Registry columns keep their feature-tab class/type and the canonical registry order; any
        non-registry column (imported reference / one-hot) lands in a trailing "Imported / other"
        class so nothing is ever dropped."""
        from featuregen.features import FEATURES, FEATURES_BY_KEY

        present = set(columns)
        result: list[tuple[str, list[tuple[str, list[str]]]]] = []
        # Walk the registry in order, picking up only the columns actually loaded.
        for grp_cls, cls_defs in groupby(FEATURES, key=lambda f: f.cls):
            types: list[tuple[str, list[str]]] = []
            for type_name, defs in groupby(cls_defs, key=lambda f: f.group):
                cols = [f.key for f in defs if f.key in present]
                if cols:
                    types.append((type_name, cols))
            if types:
                result.append((grp_cls, types))
        # Anything else not in the registry (kept in the workbook's column order).
        extra = [c for c in columns if c not in FEATURES_BY_KEY]
        if extra:
            result.append(("Imported / other", [("", extra)]))
        return result

    def _refresh_feature_stats(self) -> None:
        """Recompute each feature's [min–max] + direct R on the training split and show them."""
        if self._dataset is None or not self._stat_labels:
            return
        cols = list(self._stat_labels.keys())
        try:
            stats = training_split_stats(
                self._dataset, cols, self.outer_test_size.value(), self._project_seed(),
                self.use_aug_fit.isChecked(),
            )
        except Exception:  # noqa: BLE001 - the preview must never break the page
            stats = {}
        for col, label in self._stat_labels.items():
            label.setText(self._format_stat(*stats.get(col, (None, None, None))))

    @staticmethod
    def _format_stat(mn, mx, direct_r) -> str:
        rng = f"[{mn:.3g} – {mx:.3g}]" if mn is not None and mx is not None else "[n/a]"
        rs = f"R={direct_r:.2f}" if direct_r is not None else "R=n/a"
        return f"{rng}  {rs}"

    def _set_all_features(self, checked: bool) -> None:
        for chk in self._feature_checks.values():
            chk.setChecked(checked)

    def _set_all_normalize(self, state: bool) -> None:
        for chk in self._normalize_checks.values():
            chk.setChecked(state)

    def _selected_features(self) -> list[str]:
        return [c for c, chk in self._feature_checks.items() if chk.isChecked()]

    def _normalize_features(self) -> list[str]:
        """Selected features whose 'normalize' toggle is on (subset of selected features)."""
        return [
            c
            for c in self._selected_features()
            if self._normalize_checks.get(c) is not None and self._normalize_checks[c].isChecked()
        ]

    # ------------------------------------------------------------------ #
    # Shared run controls                                                #
    # ------------------------------------------------------------------ #
    def _on_pause(self) -> None:
        """Toggle the active run paused/resumed."""
        worker = self._active_worker
        if worker is None:
            return
        if self.pause_button.text() == "Pause":
            worker.pause()
            self.pause_button.setText("Resume")
            self._log("Paused — will hold at the next checkpoint.")
        else:
            worker.resume()
            self.pause_button.setText("Pause")
            self._log("Resumed.")

    def _on_stop(self) -> None:
        """Ask for confirmation, then stop the active run early (work finished so far is kept)."""
        worker = self._active_worker
        if worker is None:
            return
        if self.sweep_check.isChecked():
            prompt = ("Stop the sweep now? Models already finished are kept and compared; the model "
                      "in progress is discarded.")
        else:
            prompt = ("Stop the run now? Outer folds already finished are kept and reported; the "
                      "in-progress fold is discarded so the estimate stays unbiased.")
        if QMessageBox.question(self, "Stop", prompt) != QMessageBox.Yes:
            return
        worker.stop()
        self.pause_button.setText("Pause")
        self.stop_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self._log("Stopping after the current step…")

    def _suggested_model_name(self, history) -> str:
        """Build the suggested Save name: ``<model_key>_nested-CV_rrmse<avg>_<timestamp>``.

        The rRMSE is the across-folds mean (so the name carries the headline number), with the decimal
        point written as ``p`` to keep a clean stem.
        """
        from datetime import datetime
        avg_rrmse = average_split_metrics(history).get("rrmse", float("nan"))
        rrmse_txt = f"rrmse{avg_rrmse:.2f}".replace(".", "p") if avg_rrmse == avg_rrmse else "rrmse_na"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self._current_model_key()}_nested-CV_{rrmse_txt}_{stamp}"

    def _enable_save(self, suggested_name: str) -> None:
        """Pre-fill the Save name field with ``suggested_name`` and enable Save + Go-to-results."""
        self.save_name.setText(suggested_name)
        self.save_button.setEnabled(True)
        self.results_button.setEnabled(True)

    def _on_save(self) -> None:
        """Save the finished run as a bundle under the name in the field.

        A sweep saves its *winner* — the result the page kept — so Save always ships the model whose
        number is on screen.
        """
        if self._session.project is None:
            QMessageBox.warning(self, "No project", "Open or create a project first.")
            return
        if self._vault_result is None:
            return
        vault = self._vault_result
        meta = {
            "mean_held_out_rrmse": vault.mean_held_out_rrmse,
            "std_held_out_rrmse": vault.std_held_out_rrmse,
            "n_outer": vault.n_outer,
            "did_optimize": vault.did_optimize,
            "fold_features": vault.fold_features,
            "fold_params": vault.fold_params,
        }
        try:
            path = save_bundle(
                vault.history, str(self._session.project.model_dir),
                model_key=vault.model_key or self._current_model_key(),
                vault_meta=meta, name=self.save_name.text().strip() or None,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self._log(f"Saved model bundle: {path.name}")
        self.open_folder_button.setEnabled(True)
        QMessageBox.information(
            self, "Saved",
            f"Saved to:\n{path}\n\nChoose it on the Results tab to compute the field map.",
        )

    def _on_open_folder(self) -> None:
        if self._session.project is not None:
            open_folder(self._session.project.model_dir)

    # ------------------------------------------------------------------ #
    # The run (nested CV, optionally swept over every model)             #
    # ------------------------------------------------------------------ #
    def _set_running(self, running: bool) -> None:
        sweeping = self.sweep_check.isChecked()
        self.run_button.setEnabled(not running)
        if running:
            self.run_button.setText("Sweeping…" if sweeping else "Running…")
        else:
            self.run_button.setText("Run sweep" if sweeping else "Run")
        self.pause_button.setEnabled(running)
        self.stop_button.setEnabled(running)
        if not running:
            self.pause_button.setText("Pause")
            self._clear_status()  # the run ended (done / failed / stopped) — blank the live status line
        self.auto_run_banner.setText(
            "● Auto run in progress — model sweep" if sweeping else "● Auto run in progress — nested CV"
        )
        self.auto_run_banner.setVisible(running)
        self._active_worker = self._validator if running else None

    def _on_run(self) -> None:
        if self._dataset is None:
            QMessageBox.warning(self, "No data", "Load a features and targets file first.")
            return
        features = self._selected_features()
        if not features:
            QMessageBox.warning(self, "No features", "Select at least one feature column.")
            return
        sweeping = self.sweep_check.isChecked()
        do_optimize = self.optimize_check.isChecked()
        # GPR is O(n^3): warn before a run that will fit it many times. A sweep always includes it.
        if len(self._dataset.frame) > _GPR_WARN_ROWS and (sweeping or self._current_model_key() == "gpr"):
            who = "The sweep includes Gaussian Process, which" if sweeping else "Gaussian Process"
            if QMessageBox.question(
                self, "Large dataset",
                f"{who} is O(n³) and may be very slow on {len(self._dataset.frame)} rows. Continue?",
            ) != QMessageBox.Yes:
                return

        inner_mode = self._inner_split_mode_value()
        outer_mode = self._outer_split_mode_value()
        n_outer = self.outer_n_splits.value()
        # ``config`` carries the model, the hyperparameters the form holds, the features and the outer
        # hold-out ratio; the kwargs carry the fold counts, the split modes and the inner search.
        config = TrainConfig(
            model_key=self._current_model_key(),
            params=self._collect_params(),
            feature_columns=features,
            normalize_columns=self._normalize_features(),
            test_size=self.outer_test_size.value(),
            seed=self._project_seed(),
            train_on_augmented=self.use_aug_fit.isChecked(),
            validate_on_augmented=self.use_aug_val.isChecked(),
            target_transform=self._target_transform_value(),
            target_bias_correction=self._target_bias_value(),
        )
        kwargs = dict(
            do_optimize=do_optimize,
            n_outer_splits=n_outer,
            outer_split_mode=outer_mode,
            inner_test_size=self.inner_test_size.value(),
            inner_split_mode=inner_mode,
            opt_n_cv_splits=self.inner_n_splits.value(),
            opt_trials=self.n_trials.value(),
            opt_fit_on_augmented=self.use_aug_fit.isChecked(),
            opt_validate_on_augmented=self.use_aug_val.isChecked(),
            seed=config.seed,
        )
        model_keys = [m.key for m in MODELS] if sweeping else None

        self._sweep_results = None
        self._sweep_model_index = 1 if sweeping else 0
        self._sweep_total = len(model_keys) if sweeping else 0
        self.save_button.setEnabled(False)  # the previous run's model is now stale
        self.progress.setRange(0, 0)  # busy: the fold loop reports its own per-fold progress
        self._clear_graph()
        tuning = "tuned per fold" if do_optimize else "fixed hyperparameters"
        if sweeping:
            self._set_status(f"Sweep — starting ({len(model_keys)} models)…")
            self._log(
                f"Sweeping {len(model_keys)} model(s) over {n_outer} {outer_mode} outer fold(s) "
                f"({tuning}, {inner_mode} inner CV @ test={self.inner_test_size.value():g}, "
                f"project seed {config.seed})…"
            )
        else:
            self._set_status(f"Running — starting ({n_outer} outer fold(s))…")
            self._log(
                f"Running {config.model_key} over {n_outer} {outer_mode} outer fold(s) ({tuning}, "
                f"{inner_mode} inner CV @ test={self.inner_test_size.value():g}, project seed "
                f"{config.seed})…"
            )

        if config.target_transform != "none":
            from ml.target_transform import describe

            self._log(describe(config.target_transform, config.target_bias_correction))

        self._validator = ValidateWorker(self._dataset, config, kwargs, model_keys=model_keys)
        self._validator.warnings_enabled = self.show_warnings_check.isChecked
        self._validator.fold_done.connect(self._on_fold)
        self._validator.inner_progressed.connect(self._on_inner_progress)
        self._validator.model_done.connect(self._on_sweep_model)
        self._validator.sweep_done.connect(self._on_sweep_finished)
        self._validator.noted.connect(self._log)
        self._validator.warned.connect(self._log)
        self._validator.finished_ok.connect(self._on_finished)
        self._validator.failed.connect(self._on_failed)
        self._set_running(True)  # set after the worker exists, so Pause/Stop bind to it
        self._validator.start()

    def _on_fold(self, fold: int, total: int, held_out: float, train: float,
                 train_metrics: dict, test_metrics: dict) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(fold)
        self._set_status(f"{self._stage_prefix()} — outer fold {fold}/{total} done")
        # During processing show only train + held-out rRMSE; the graph fills every metric per fold.
        self._log(
            f"  outer fold {fold}/{total}: held-out rRMSE={held_out:.4g}%  train rRMSE={train:.4g}%"
        )
        # A sweep plots only its first model's folds — mixing several models' curves on one axis would
        # read as one run's spread. The console's comparison table is the sweep's real report.
        if self._sweep_model_index <= 1:
            self._append_graph_point(train_metrics, test_metrics)

    def _on_inner_progress(self, fold: int, n_outer: int, stage: str, trial: int, total: int) -> None:
        """Tick the live status line through a fold's inner hyperparameter search."""
        self._set_status(
            f"{self._stage_prefix()} — outer fold {fold}/{n_outer} · {stage} trial {trial}/{total}"
        )

    def _stage_prefix(self) -> str:
        """The live status line's leading label — counting the sweep's models, when sweeping."""
        if self._sweep_model_index:
            return f"Sweep {self._sweep_model_index}/{self._sweep_total}"
        return "Running"

    def _on_sweep_model(self, index: int, total: int, key: str, result) -> None:
        """One model of a sweep finished (or failed, with ``result`` None) — log its headline number.

        The counters also drive :meth:`_stage_prefix`, so the status line names the model the *next*
        model's folds belong to.
        """
        label = MODELS_BY_KEY[key].label if key in MODELS_BY_KEY else key
        if result is None:
            self._log(f"  {label}: failed (see above) — skipped.")
        else:
            self._log(
                f"  {label}: held-out rRMSE = {result.mean_held_out_rrmse:.4g}% "
                f"± {result.std_held_out_rrmse:.3g}%"
            )
        # Point the counter at the model that is about to start, so the status line stays truthful.
        self._sweep_model_index = index + 1 if index < total else 0
        self._sweep_total = total

    def _on_sweep_finished(self, ranked) -> None:
        """A sweep finished: keep the ranked list and print the comparison table."""
        self._sweep_results = list(ranked)
        self._log("")
        self._log(f"{'#':<3} {'Model':<28} {'held-out rRMSE':>15} {'R²':>10} {'R':>10} "
                  f"{'MAPE':>11} {'train rRMSE':>14}")
        for i, res in enumerate(ranked, start=1):
            label = MODELS_BY_KEY[res.model_key].label if res.model_key in MODELS_BY_KEY else res.model_key
            test = average_split_metrics(res.history)
            train = average_split_train_metrics(res.history)
            # Format each number to its own string first, then pad the string — padding the float
            # directly lets a wide value (a negative R², an exponent) run into the next column.
            cells = (
                f"{test.get('rrmse', float('nan')):.4g}%",
                f"{test.get('r2', float('nan')):.3g}",
                f"{test.get('r', float('nan')):.3g}",
                f"{test.get('mape', float('nan')):.4g}%",
                f"{train.get('rrmse', float('nan')):.4g}%",
            )
            widths = (15, 10, 10, 11, 14)
            row = " ".join(c.rjust(w) for c, w in zip(cells, widths))
            self._log(f"{i:<3} {label[:28]:<28} {row}")
        self._log("")

    def _on_finished(self, vault) -> None:
        """The run (or the sweep's winner) finished: report it and arm Save."""
        self._sweep_model_index = 0
        self._vault_result = vault
        mean, std = vault.mean_held_out_rrmse, vault.std_held_out_rrmse
        if self._sweep_results:
            label = (MODELS_BY_KEY[vault.model_key].label
                     if vault.model_key in MODELS_BY_KEY else vault.model_key)
            # Select the winner in the (re-enabled) combo, so the form and the kept result agree.
            idx = self.model_combo.findData(vault.model_key)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
            self._log(f"Sweep winner: {label} — held-out rRMSE = {mean:.4g}% ± {std:.3g}% "
                      f"across {len(vault.history.splits)} fold(s).")
        else:
            self._log(f"Done: held-out rRMSE = {mean:.4g}% ± {std:.3g}% "
                      f"across {len(vault.history.splits)} fold(s).")
        # If the run tuned per fold, the folds' hyperparameters differ — write the best fold's into the
        # form so the user can see (and re-use) what won, exactly as the old Optimize button did.
        if vault.did_optimize and vault.fold_params:
            best = vault.fold_params[vault.history.best_split - 1]
            self._apply_params(best)
            pretty = ", ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in best.items()
            )
            self._log(f"Best fold's hyperparameters applied to the form: {pretty}")
        # End-of-run summary: averaged train & held-out rRMSE / R² / R / MAPE across the outer folds.
        self._log_avg_metrics(vault.history)
        self._log("Edit the name on the right if you like, then Save to keep this model.")
        self._set_running(False)  # after the logging, so the buttons re-enable on a settled state
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self._enable_save(self._suggested_model_name(vault.history))

    def _on_failed(self, message: str) -> None:
        self._sweep_model_index = 0
        self._set_running(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        if "before any" in message.lower():
            # A user-initiated stop with nothing completed: report it plainly, not as an error.
            self._log(message)
            return
        if "optuna" in message.lower():
            message += "\n\nInstall it with:  .venv\\Scripts\\pip.exe install optuna"
        self._log(f"ERROR: {message}")
        QMessageBox.critical(self, "Run failed", message)

    # ------------------------------------------------------------------ #
    # Plot helpers                                                       #
    # ------------------------------------------------------------------ #
    def _clear_graph(self) -> None:
        """Empty the per-metric series so the graph reads blank (a new run is about to fill it)."""
        self._train_series = {}
        self._test_series = {}
        self._reset_plot()

    def _append_graph_point(self, train_metrics: dict, test_metrics: dict) -> None:
        """Append one fold's train + held-out value for every metric, then redraw the active tab."""
        for key, *_ in self._GRAPH_METRICS:
            self._train_series.setdefault(key, []).append(float(train_metrics.get(key, float("nan"))))
            self._test_series.setdefault(key, []).append(float(test_metrics.get(key, float("nan"))))
        self._redraw_plot()

    def _reset_plot(self) -> None:
        self.draw_into(self.ax)
        self.canvas.draw_idle()

    def _redraw_plot(self) -> None:
        self.draw_into(self.ax)
        self.canvas.draw_idle()

    def draw_into(self, ax, metric: str | None = None) -> None:
        """Draw the per-outer-fold train vs held-out curve for one metric into ``ax``.

        Pure render shared by the live canvas and the export layer; reads the accumulated per-metric
        series but touches no widgets. ``metric`` defaults to the currently-selected graph tab. With no
        data yet it draws just the labelled, empty axes.
        """
        metric = metric or self._current_graph_metric()
        meta = next((m for m in self._GRAPH_METRICS if m[0] == metric), self._GRAPH_METRICS[0])
        _key, label, ylabel, ylim = meta

        ax.clear()
        train = self._train_series.get(metric, [])
        test = self._test_series.get(metric, [])
        if train:
            xs = list(range(1, len(train) + 1))
            ax.plot(xs, train, label="train", marker="o", markersize=3)
            ax.plot(xs, test, label="held-out", marker="o", markersize=3)
            ax.legend(loc="best")
        ax.set_title(f"{label} per outer fold")
        ax.set_xlabel("outer fold")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)

    def export_title(self) -> str:
        return f"ml_{self._current_graph_metric()}_curve"

    def _set_status(self, message: str) -> None:
        """Overwrite the single live status line above the console (in-place quick feedback)."""
        self.status_line.setText(message)

    def _clear_status(self) -> None:
        """Blank the live status line (called when a run finishes, fails or is stopped)."""
        self.status_line.setText("")

    def _log(self, message: str) -> None:
        """Append a plain message to the console.

        ``quote=False`` keeps apostrophes/quotes literal (e.g. a model key in single quotes) — only
        ``<``/``>``/``&`` need escaping in a plain-text console; quoting them would print ``&#x27;``.
        """
        self.console.append(escape(message, quote=False))

    @staticmethod
    def _fmt_metrics(m: dict) -> str:
        """One-line ``rRMSE=… R²=… R=… MAPE=…`` summary of a standard-metrics dict."""
        return (f"rRMSE={m.get('rrmse', float('nan')):.4g}%  R²={m.get('r2', float('nan')):.4g}  "
                f"R={m.get('r', float('nan')):.4g}  MAPE={m.get('mape', float('nan')):.4g}%")

    def _log_avg_metrics(self, history) -> None:
        """Log the averaged train and held-out 4-metric summary across the run's outer folds."""
        train_avg = average_split_train_metrics(history)
        test_avg = average_split_metrics(history)
        n = len(history.splits)
        self._log(f"Average across {n} fold(s) — train:     {self._fmt_metrics(train_avg)}")
        self._log(f"Average across {n} fold(s) — held-out:  {self._fmt_metrics(test_avg)}")

    # ------------------------------------------------------------------ #
    # Hand-off                                                           #
    # ------------------------------------------------------------------ #
    def on_enter(self) -> None:
        project = self._session.project
        if project is None:
            self.output_label.setText("Output: (no project open)")
            set_status_indicator(self.indicator, False, "No project open.")
            return
        self.output_label.setText(f"Model output → {project.model_dir}")
        # Auto-load the dataset from the project the first time. After that the cache is dropped
        # explicitly whenever the workbooks change underneath it — a Clear or a re-run of feature-gen
        # — via invalidate_dataset(), so there's nothing to re-check on entry.
        if self._dataset is None:
            features, targets = self._resolve_workbooks()
            if features is not None and targets is not None:
                self.features_path.setText(str(features))
                self.targets_path.setText(str(targets))
                self._on_load()
        self._refresh_indicator()

    def invalidate_dataset(self) -> None:
        """Drop the cached dataset so the next tab entry rebuilds it from disk.

        Called by the shell when the workbooks this page was built from change — a Clear deletes
        them, or a re-run of feature-gen rewrites them — so a stale dataset can't be run against files
        that no longer exist.
        """
        self._dataset = None
        self._refresh_indicator()

    def reset_for_project(self) -> None:
        """Wipe every trace of the previous project so a new one starts clean.

        Called by the shell when a project is created/opened, before on_enter re-seeds the page
        from the now-active project. Drops the cached dataset and the last run (its result, ranking
        and graph), empties the console and path fields, and clears the feature-checkbox list so
        nothing from the old project carries over.
        """
        self._dataset = None
        self._vault_result = None
        self._sweep_results = None
        self._clear_graph()
        self.console.clear()
        self.features_path.clear()
        self.targets_path.clear()
        self._rebuild_feature_checks([])
        self._refresh_indicator()

    def _refresh_indicator(self) -> None:
        """Show whether the features + targets dataset is loaded and ready to run."""
        if self._dataset is not None:
            # Rows whose target is missing are dropped before training (make_xy), so the count that
            # matters is the usable one. Reporting only the table length made a partly-matched
            # reference sheet look like a full dataset.
            frame = self._dataset.frame
            total = len(frame)
            usable = int(pd.to_numeric(frame[self._dataset.target_column],
                                       errors="coerce").notna().sum())
            counts = (f"{total} row(s)" if usable == total
                      else f"{usable} of {total} row(s) usable ({total - usable} with no target)")
            set_status_indicator(
                self.indicator, True,
                f"Dataset loaded: {counts}, target '{self._dataset.target_column}'.",
            )
        else:
            set_status_indicator(
                self.indicator, False,
                "No dataset loaded — generate features with a target, then Reload.",
            )

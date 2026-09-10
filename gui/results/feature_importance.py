"""Results sub-tab: explain the loaded model's behaviour by **permutation feature importance**.

Shuffle one feature's column and measure how much the model's held-out rRMSE gets worse; the bigger
the rise, the more the model relied on that feature. It is model-agnostic — it only feeds the fitted
pipeline inputs and reads its score — so it works for *every* classical model in the registry, unlike
native importances that exist only for trees and linear models (see :mod:`ml.explain`).

This is the one results analysis that can't be served from the precomputed bundle: it re-scores the
active split's fitted model many times over, so it runs **on demand** (a Compute button) and **off the
UI thread** (an :class:`~gui.workers.ImportanceWorker`). The defaults are tuned for stability against
the randomness of a single shuffle — 30 repeats, scored on the held-out plots — and the per-feature
std is drawn as an error bar so the user can *see* how settled each ranking is.

A deep model has no tabular features, so the tab greys out for deep bundles (mirroring how the map's
feature overlay is classical-only).
"""

from __future__ import annotations

import os

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ml.explain import DEFAULT_N_REPEATS, SCORE_CHOICES

from ..export import ChartValues, attach_export_menu

# Human labels for the score-subset choices (ml.explain.SCORE_CHOICES order).
_SCORE_LABELS = {
    "held_out": "Held-out plots (leakage-free)",
    "train": "Training plots",
    "all": "All plots",
}

# Show at most this many bars by default so a 40-feature model stays readable; a toggle lifts it.
_TOP_N_DEFAULT = 15


class FeatureImportance(QWidget):
    """The feature-importance sub-tab: a permutation-importance bar chart for the active split.

    The host owns the data, so this widget asks for a (re)compute via :attr:`compute_requested`
    rather than reaching for the dataset itself — it has no source to permute until the host has
    built one (the same lazy-source contract the map's feature overlay uses).
    """

    # Emitted when the user presses Compute; the host runs the worker and calls back set_result().
    compute_requested = Signal(int, str, int)  # n_repeats, score_on, n_jobs

    def __init__(self) -> None:
        super().__init__()
        self._result = None           # ml.ImportanceResult | None
        self._enabled_kind = False    # is the loaded model classical (so importance is meaningful)?
        self._busy = False

        root = QVBoxLayout(self)

        controls = QHBoxLayout()
        controls.addWidget(self._bold("Repeats:"))
        self.repeats_spin = QSpinBox()
        self.repeats_spin.setRange(3, 200)
        self.repeats_spin.setValue(DEFAULT_N_REPEATS)
        self.repeats_spin.setToolTip(
            "How many times each feature is shuffled and re-scored. More repeats average out the "
            "randomness of a single shuffle (steadier bars, smaller error bars) but take longer.\n"
            f"Default {DEFAULT_N_REPEATS} is a strong guard against shuffle noise."
        )
        controls.addWidget(self.repeats_spin)

        controls.addWidget(self._bold("Score on:"))
        self.score_combo = QComboBox()
        for key in SCORE_CHOICES:
            self.score_combo.addItem(_SCORE_LABELS.get(key, key), key)
        self.score_combo.setToolTip(
            "Which plots to measure the score drop on. Held-out (the default) measures importance "
            "for generalisation — what actually transfers to unseen plots — and can't leak, since "
            "those plots weren't fitted. Training/All are in-sample and tend to over-rate features."
        )
        controls.addWidget(self.score_combo)

        controls.addWidget(self._bold("CPU threads:"))
        self.threads_spin = QSpinBox()
        cores = os.cpu_count() or 1
        self.threads_spin.setRange(1, cores)
        self.threads_spin.setValue(max(1, cores - 2))  # leave a couple of threads for the UI
        self.threads_spin.setToolTip(
            f"CPU threads used to score the shuffles in parallel ({cores} available). The shuffles are "
            "independent predictions, and all permutations are drawn up front, so this only changes "
            "speed — the ranking is identical (and reproducible) at any thread count."
        )
        controls.addWidget(self.threads_spin)

        self.top_only = QCheckBox(f"Top {_TOP_N_DEFAULT} only")
        self.top_only.setChecked(True)
        self.top_only.setToolTip("Show only the most important features (keeps a wide model readable).")
        self.top_only.toggled.connect(self._redraw)
        controls.addWidget(self.top_only)

        self.compute_button = QPushButton("Compute")
        self.compute_button.setToolTip("Run the permutation-importance analysis on the active split.")
        self.compute_button.clicked.connect(self._on_compute)
        controls.addWidget(self.compute_button)
        controls.addStretch(1)
        root.addLayout(controls)

        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: #555;")
        root.addWidget(self.info_label)

        self.figure = Figure(figsize=(6, 4), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        attach_export_menu(self.canvas, self)
        root.addWidget(self.canvas, 1)

        self._draw_placeholder("Pick a classical model, then press Compute.")

    @staticmethod
    def _bold(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("font-weight: bold;")
        return label

    # ------------------------------------------------------------------ #
    # Host interface                                                     #
    # ------------------------------------------------------------------ #
    def set_model(self, *, is_classical: bool, has_split_model: bool) -> None:
        """Tell the tab which kind of model is loaded; gate the controls accordingly.

        Called on every model/split change. A new model invalidates any previous result (it belonged
        to a different model), so the chart resets to a placeholder until the user computes again.
        """
        self._enabled_kind = bool(is_classical and has_split_model)
        self._result = None
        self._set_busy(False)
        if not has_split_model:
            self._gate("Save and select a model to explain its features.")
        elif not is_classical:
            self._gate("Feature importance applies to classical models only — a deep model has no "
                       "tabular features to permute.")
        else:
            self._enable(True)
            self._draw_placeholder("Press Compute to rank this model's features.")

    def set_result(self, result) -> None:
        """Receive a finished :class:`ml.ImportanceResult` from the host's worker and draw it."""
        self._result = result
        self._set_busy(False)
        self._redraw()

    def set_failed(self, message: str) -> None:
        """The host's worker failed; surface it and re-enable Compute."""
        self._result = None
        self._set_busy(False)
        self._draw_placeholder(f"Could not compute importance: {message}")

    def begin_compute(self) -> None:
        """The host accepted a compute request and started the worker — show the busy state."""
        self._set_busy(True)

    def invalidate(self, reason: str) -> None:
        """Drop the current ranking because an input it depends on (e.g. the aug toggle) changed.

        Unlike the precomputed sub-tabs, importance can't be re-derived instantly — it re-scores the
        model many times — so rather than auto-recomputing on every toggle flick we clear the stale
        chart and ask the user to press Compute again under the new choice.
        """
        if not self._enabled_kind or self._busy or self._result is None:
            return
        self._result = None
        self._draw_placeholder(reason)

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #
    def _gate(self, message: str) -> None:
        self._enable(False)
        self._draw_placeholder(message)

    def _enable(self, on: bool) -> None:
        for w in (self.repeats_spin, self.score_combo, self.threads_spin,
                  self.compute_button, self.top_only):
            w.setEnabled(on)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.compute_button.setEnabled(self._enabled_kind and not busy)
        self.compute_button.setText("Computing…" if busy else "Compute")

    def _on_compute(self) -> None:
        if not self._enabled_kind or self._busy:
            return
        self.compute_requested.emit(
            int(self.repeats_spin.value()),
            str(self.score_combo.currentData()),
            int(self.threads_spin.value()),
        )

    def _redraw(self) -> None:
        self.draw_into(self.ax)
        self.canvas.draw_idle()
        if self._result is not None:
            self.info_label.setText(self._summary(self._result))

    @staticmethod
    def _summary(result) -> str:
        if result.is_empty:
            return "  ".join(result.notes) if result.notes else "No features to show."
        subset = _SCORE_LABELS.get(result.score_on, result.score_on)
        aug = "incl. augmented copies" if result.include_aug else "originals only"
        text = (
            f"Baseline rRMSE {result.baseline_rrmse:.3g}% over {result.n_scored} rows "
            f"({subset}, {aug}); each bar is the mean rRMSE increase across {result.n_repeats} "
            f"shuffles (error bar = std)."
        )
        if result.notes:
            text += "  ⚠ " + "  ".join(result.notes)
        return text

    def _draw_placeholder(self, text: str) -> None:
        self.info_label.setText("")
        self.ax.clear()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.text(0.5, 0.5, text, ha="center", va="center", color="gray",
                     wrap=True, transform=self.ax.transAxes)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ #
    # Export provider (matplotlib draw_into + export_title)              #
    # ------------------------------------------------------------------ #
    def export_title(self) -> str:
        return "feature_importance"

    def draw_into(self, ax) -> None:
        """Draw the importance bar chart into ``ax`` (live canvas or a fresh export figure).

        Pure render over the cached result: a horizontal bar per feature (most important at the top),
        with the per-feature std as a symmetric error bar. Reads the Top-N toggle but mutates nothing.
        """
        ax.clear()
        result = self._result
        if result is None or result.is_empty:
            ax.set_xticks([])
            ax.set_yticks([])
            msg = "Press Compute to rank this model's features."
            if result is not None and result.notes:
                msg = "  ".join(result.notes)
            ax.text(0.5, 0.5, msg, ha="center", va="center", color="gray",
                    wrap=True, transform=ax.transAxes)
            return

        features = list(result.features)
        means = np.asarray(result.mean, dtype=float)
        stds = np.asarray(result.std, dtype=float)
        if self.top_only.isChecked() and len(features) > _TOP_N_DEFAULT:
            features = features[:_TOP_N_DEFAULT]
            means = means[:_TOP_N_DEFAULT]
            stds = stds[:_TOP_N_DEFAULT]

        # Most-important first looks best with the largest bar on top, so plot reversed on the y-axis.
        ys = np.arange(len(features))[::-1]
        colours = ["#2c7fb8" if m >= 0 else "#bbbbbb" for m in means]  # ~zero/negative greyed
        ax.barh(ys, means, xerr=stds, color=colours, error_kw={"ecolor": "#444", "elinewidth": 0.8})
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_yticks(ys)
        ax.set_yticklabels(features, fontsize="small")
        ax.set_xlabel("Importance — rRMSE increase when shuffled (%)")
        # Title drops the "(leakage-free)" qualifier the dropdown/caption keep — the chart heading reads cleaner without it.
        subset_label = _SCORE_LABELS.get(result.score_on, result.score_on).replace(" (leakage-free)", "")
        title = f"Permutation importance ({subset_label})"
        ax.set_title(title)
        ax.grid(True, axis="x", which="major", color="0.9", linewidth=0.6)
        ax.set_axisbelow(True)

    def export_values(self) -> ChartValues:
        """Per-feature (name → mean importance) rows, most important first, honouring the Top-N toggle."""
        result = self._result
        if result is None or result.is_empty:
            return ChartValues(rows=[], unit="")
        features = list(result.features)
        means = np.asarray(result.mean, dtype=float)
        if self.top_only.isChecked() and len(features) > _TOP_N_DEFAULT:
            features = features[:_TOP_N_DEFAULT]
            means = means[:_TOP_N_DEFAULT]
        rows = [(feat, float(m)) for feat, m in zip(features, means)]
        return ChartValues(rows=rows, unit="rRMSE increase when shuffled (%)")

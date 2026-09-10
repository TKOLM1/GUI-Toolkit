"""Results sub-tab: predicted-vs-actual performance for the loaded model's active split.

A scatter of ground-truth (x) against predicted (y) for the loaded model's active split. Two parts
of the split can be shown independently, each with its own three-way selector:

* **Validation (held-out)** plots — the plots the model never saw, drawn as crosses.
* **Training** plots — the in-sample plots the model was fit on, drawn as faint circles.

Each selector offers **Disabled / Show points / Fit line (and show points)**. When *both* are showing
points, a **Fit line to all** toggle additionally fits one least-squares line through the two sets
combined. A grey 1:1 reference line marks perfect predictions.

The points come from the host's per-plot prediction table (every plot predicted with the active
split's model, tagged ``role`` T/V and ``aug``), filtered to honour the host's two "Aug in … metrics"
toggles: an original plot always shows; an augmented copy shows only when the toggle for its side
(training / validation) is on. The top-left shows **one metric box per shown series** (held-out
and/or training), plus a combined **held-out + training** box when the combined fit is on — each
reporting the metrics of exactly its own points, so every box matches its dots. A **Show metric
labels** toggle (on by default) hides all the boxes at once.

An **Average over all splits** toggle replaces the single split with one pooled cross-validated
cloud: every split's stored held-out predictions are concatenated (each plot appears once per split
it was held out in), drawn as a single series with its own fit and metric box — all labelled
"(average)". The training/held-out selectors don't apply in that mode and are greyed out.

Separately, a **delta label** beneath the controls reports the per-metric generalisation gap
``held-out − training`` — for the active split, or (in average mode) the mean held-out minus the mean
training across splits — read from the precomputed metric cube under the current augmented-data choice
(``n/a`` for any metric an older model didn't record).

The splitting is a two-way grouped train/held-out split (no separate validation fold), so "test",
"validation" and "held-out" all name the same plots the model never saw. Which model and split are
shown is chosen by the host page's dropdowns.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from common.naming import aug_number_from_name
from ml.metrics import standard_metrics
from ml.trainer import active_split_model, variant_metrics

from ..export import ChartValues, attach_export_menu, draw_boxes
from .variants import VariantSelection

# The four standard metrics, as (key, label, formatter) — drives the in-graph box and the delta label.
_METRICS = (
    ("rrmse", "rRMSE", lambda v: f"{v:.4g}%"),
    ("r2", "R²", lambda v: f"{v:.4g}"),
    ("r", "R", lambda v: f"{v:.4g}"),
    ("mape", "MAPE", lambda v: f"{v:.4g}%"),
)

# Three-way display selector values, in order.
_MODE_OFF, _MODE_POINTS, _MODE_FIT = "off", "points", "fit"


def _polyfit_xy(actual: np.ndarray, predicted: np.ndarray):
    """Least-squares slope/intercept for ``predicted`` on ``actual``, or ``None`` if degenerate."""
    if actual.size >= 2 and float(np.var(actual, ddof=0)) > 0:
        slope, intercept = np.polyfit(actual, predicted, 1)
        return float(slope), float(intercept)
    return None


class ModelPerformance(QWidget):
    """The predicted-vs-actual sub-tab for the loaded model's active split."""

    def __init__(self) -> None:
        super().__init__()
        self._history = None
        self._table = None  # host's per-plot prediction table for the active split (role T/V, aug)
        self._selection = VariantSelection.legacy()

        root = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.val_mode = self._make_mode_combo(
            "Validation", "the held-out (out-of-sample) plots — the model never saw these"
        )
        bar.addWidget(self.val_mode)
        self.train_mode = self._make_mode_combo(
            "Training", "the in-sample plots the model was fit on (their fit is optimistic)"
        )
        bar.addWidget(self.train_mode)
        self.fit_all = QCheckBox("Fit line to all (combined)")
        self.fit_all.setToolTip(
            "Fit one least-squares line through the training and held-out points combined. "
            "Available only when both training and held-out points are shown."
        )
        self.fit_all.toggled.connect(self._redraw)
        bar.addWidget(self.fit_all)
        self.show_labels = QCheckBox("Show metric labels")
        self.show_labels.setChecked(True)
        self.show_labels.setToolTip(
            "Show the in-graph metric boxes (top-left): one per shown series, plus a combined box "
            "when 'Fit line to all' is on."
        )
        self.show_labels.toggled.connect(self._redraw)
        bar.addWidget(self.show_labels)
        self.average_splits = QCheckBox("Average over all splits")
        self.average_splits.setToolTip(
            "Instead of one split, pool every split's held-out predictions into a single "
            "cross-validated cloud (each plot appears once per split it was held out in). The "
            "training/held-out selectors don't apply in this mode — all points are shown together."
        )
        self.average_splits.toggled.connect(self._redraw)
        bar.addWidget(self.average_splits)
        bar.addStretch(1)
        root.addLayout(bar)

        # The generalisation gap, in its own label (not folded into the in-graph box).
        self.delta_label = QLabel("")
        self.delta_label.setStyleSheet("color: #555;")
        self.delta_label.setToolTip(
            "Per-metric held-out − training for this split (the generalisation gap), under the "
            "current augmented-data choice."
        )
        root.addWidget(self.delta_label)

        self.figure = Figure(figsize=(5, 5), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self._draw_empty("Train a model to see its predicted-vs-actual plot.")
        root.addWidget(self.canvas, 1)
        attach_export_menu(self.canvas, self)

    def _make_mode_combo(self, name: str, what: str) -> QComboBox:
        combo = QComboBox()
        combo.addItem(f"{name}: Disabled", _MODE_OFF)
        combo.addItem(f"{name}: Show points", _MODE_POINTS)
        combo.addItem(f"{name}: Fit line (and show points)", _MODE_FIT)
        combo.setToolTip(f"How to show {what}.")
        combo.currentIndexChanged.connect(self._redraw)
        return combo

    # ------------------------------------------------------------------ #
    # Data in / sync                                                     #
    # ------------------------------------------------------------------ #
    def refresh(self, history, table=None, selection: VariantSelection | None = None) -> None:
        """Receive the loaded model's :class:`TrainHistory` (or ``None``), the per-plot table and the
        augmented-data choice, then redraw.

        ``table`` is the host's per-plot prediction table for the active split (columns
        ``actual``/``predicted``/``role``/``aug`` indexed by file name); it supplies the points for
        both series. It may be ``None`` (the scatter is then unavailable).
        """
        self._history = history
        self._table = table
        self._selection = selection or VariantSelection.legacy()
        self._redraw()

    # ------------------------------------------------------------------ #
    # Point selection                                                    #
    # ------------------------------------------------------------------ #
    def _role_xy(self, role: str, variant: str):
        """(actual, predicted) for ``role`` ('T'/'V') rows of the table, honouring the aug choice.

        Originals (``aug == 0``) always count; augmented copies count only when ``variant`` is
        ``"with_aug"``. Returns ``None`` when no table is available, else a possibly-empty pair.
        """
        table = self._table
        if table is None or not len(table) or "role" not in table:
            return None
        rows = table[table["role"] == role]
        if variant != "with_aug" and "aug" in rows:
            rows = rows[rows["aug"] == 0]
        return (rows["actual"].to_numpy(dtype=float),
                rows["predicted"].to_numpy(dtype=float))

    def _validation_xy(self):
        return self._role_xy("V", self._selection.held_out)

    def _training_xy(self):
        return self._role_xy("T", self._selection.train)

    def _pooled_xy(self):
        """(actual, predicted) pooled over every split's stored held-out predictions.

        Each split contributes its own held-out rows, so a plot appears once per split it was held
        out in — a genuine cross-validated cloud. The rows are already filtered to the model's
        train-time validation choice; we additionally drop augmented copies unless the validation aug
        toggle is on. Returns ``None`` when no split carries predictions, else a (possibly-empty) pair.
        """
        splits = getattr(self._history, "splits", None) or []
        if not splits:
            return None
        actual_parts, pred_parts = [], []
        keep_aug = self._selection.held_out == "with_aug"
        for sm in splits:
            preds = getattr(sm, "predictions", None)
            if preds is None or preds.empty:
                continue
            rows = preds
            if not keep_aug:
                is_orig = [aug_number_from_name(str(i)) is None for i in rows.index]
                rows = rows[is_orig]
            if rows.empty:
                continue
            actual_parts.append(rows["actual"].to_numpy(dtype=float))
            pred_parts.append(rows["predicted"].to_numpy(dtype=float))
        if not actual_parts:
            return (np.empty(0), np.empty(0))
        return (np.concatenate(actual_parts), np.concatenate(pred_parts))

    # ------------------------------------------------------------------ #
    # Drawing                                                            #
    # ------------------------------------------------------------------ #
    def _draw_empty(self, message: str) -> None:
        self._draw_empty_into(self.ax, message)
        self.canvas.draw_idle()

    @staticmethod
    def _draw_empty_into(ax, message: str) -> None:
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.5, message, ha="center", va="center", color="gray",
                transform=ax.transAxes)

    def _redraw(self) -> None:
        # Widget-state side effects live here (not in draw_into, which stays pure for export).
        # In average mode there is no training/held-out split, so the per-series selectors and the
        # combined-fit toggle don't apply — grey them out.
        averaging = self.average_splits.isChecked()
        for w in (self.val_mode, self.train_mode, self.fit_all):
            w.setEnabled(not averaging)
        if not averaging:
            both_shown = (self.train_mode.currentData() != _MODE_OFF
                          and self.val_mode.currentData() != _MODE_OFF)
            self.fit_all.setEnabled(both_shown)
        self._update_delta_label()
        self.draw_into(self.ax)
        self.canvas.draw_idle()

    def _update_delta_label(self) -> None:
        averaging = self.average_splits.isChecked()
        if averaging:
            deltas = self._average_deltas()
            prefix = "Δ (held-out − training, average):"
        else:
            sm = active_split_model(self._history) if self._history is not None else None
            deltas = self._split_deltas(sm) if sm is not None else {}
            prefix = "Δ (held-out − training):"
        if not deltas:
            self.delta_label.setText(f"{prefix} n/a")
            return
        parts = []
        for key, label, fmt in _METRICS:
            d = deltas.get(key)
            parts.append(f"{label} {fmt(d)}" if d is not None and np.isfinite(d) else f"{label} n/a")
        self.delta_label.setText(f"{prefix}   " + "    ".join(parts))

    def draw_into(self, ax) -> None:
        """Draw the predicted-vs-actual scatter into ``ax`` (the live axes or an export figure's).

        Pure render: reads the current data + selector state but mutates no widgets, so the export
        layer can call it on a fresh figure. Font sizes follow the axes' rcParams.
        """
        if self._history is None or not getattr(self._history, "splits", None):
            self._draw_empty_into(ax, "Train a model to see its predicted-vs-actual plot.")
            return
        sm = active_split_model(self._history)
        if sm is None:
            self._draw_empty_into(ax, "This split has no predictions.")
            return

        if self.average_splits.isChecked():
            self._draw_average(ax)
            return

        val_mode = self.val_mode.currentData()
        train_mode = self.train_mode.currentData()
        val_xy = self._validation_xy() if val_mode != _MODE_OFF else None
        train_xy = self._training_xy() if train_mode != _MODE_OFF else None
        show_val = val_xy is not None and val_xy[0].size > 0
        show_train = train_xy is not None and train_xy[0].size > 0

        if not (show_val or show_train):
            self._draw_empty_into(
                ax, "Enable training and/or validation points to see the scatter."
            )
            return

        ax.clear()
        lo_vals: list[float] = []
        hi_vals: list[float] = []

        if show_train:
            ax.scatter(train_xy[0], train_xy[1], marker="o", s=24, facecolors="none",
                       edgecolors="#7f7f7f", linewidths=1.0, alpha=0.6,
                       label="training plots (in-sample)")
            lo_vals += [train_xy[0].min(), train_xy[1].min()]
            hi_vals += [train_xy[0].max(), train_xy[1].max()]
        if show_val:
            ax.scatter(val_xy[0], val_xy[1], marker="x", s=40, color="#1f77b4", linewidths=1.2,
                       label="held-out plots")
            lo_vals += [val_xy[0].min(), val_xy[1].min()]
            hi_vals += [val_xy[0].max(), val_xy[1].max()]

        # Per-series fit lines.
        if show_val and val_mode == _MODE_FIT:
            self._draw_fit(ax, val_xy[0], val_xy[1], color="#d73027", linestyle="-",
                           label_prefix="held-out fit")
        if show_train and train_mode == _MODE_FIT:
            self._draw_fit(ax, train_xy[0], train_xy[1], color="#7f7f7f", linestyle="-",
                           label_prefix="training fit")
        # Combined fit through both sets (only meaningful, and only enabled, when both are shown).
        if show_val and show_train and self.fit_all.isChecked():
            comb_a = np.concatenate([val_xy[0], train_xy[0]])
            comb_p = np.concatenate([val_xy[1], train_xy[1]])
            self._draw_fit(ax, comb_a, comb_p, color="#6a3d9a", linestyle="--",
                           label_prefix="combined fit")

        # 1:1 reference line over the shared data range.
        lo, hi = float(min(lo_vals)), float(max(hi_vals))
        ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1, label="1:1")

        ax.set_xlabel(f"actual {self._history.target_column}")
        ax.set_ylabel(f"predicted {self._history.target_column}")
        ax.set_title(f"Split #{sm.split} — predicted vs actual")
        ax.grid(True, which="major", color="0.85", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(loc="best")

        # One metric box per shown series (matching its dots), plus a combined box when the
        # combined fit is on. Each box has a stable id so the export dialog can size / move / toggle
        # it independently; the "Show metric labels" checkbox still hides them all on the live canvas.
        if self.show_labels.isChecked():
            boxes: list[tuple[str, str]] = []
            if show_val:
                boxes.append(("held_out", self._metrics_text("held-out", val_xy[0], val_xy[1])))
            if show_train:
                boxes.append(("training", self._metrics_text("training", train_xy[0], train_xy[1])))
            if show_val and show_train and self.fit_all.isChecked():
                comb_a = np.concatenate([val_xy[0], train_xy[0]])
                comb_p = np.concatenate([val_xy[1], train_xy[1]])
                boxes.append(("combined", self._metrics_text("held-out + training", comb_a, comb_p)))
            draw_boxes(ax, boxes)

    def _draw_average(self, ax) -> None:
        """The average-over-splits view: one pooled cross-validated cloud, no T/V distinction.

        Pools every split's held-out predictions, draws them as one series with a single fit line,
        and a single metric box — all labelled "(average)" so it is never mistaken for one split.
        """
        pooled = self._pooled_xy()
        if pooled is None or pooled[0].size == 0:
            self._draw_empty_into(ax, "No saved per-split predictions to average.")
            return
        actual, predicted = pooled

        ax.clear()
        ax.scatter(actual, predicted, marker="x", s=40, color="#1f77b4", linewidths=1.2,
                   label="all plots (pooled across splits)")
        # The pooled cloud has no per-series selector, so it always carries its own fit line.
        self._draw_fit(ax, actual, predicted, color="#d73027", linestyle="-",
                       label_prefix="fit (average)")
        lo = float(min(actual.min(), predicted.min()))
        hi = float(max(actual.max(), predicted.max()))
        ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1, label="1:1")

        ax.set_xlabel(f"actual {self._history.target_column}")
        ax.set_ylabel(f"predicted {self._history.target_column}")
        n_splits = len(getattr(self._history, "splits", []) or [])
        ax.set_title(f"Average over {n_splits} splits — predicted vs actual")
        ax.grid(True, which="major", color="0.85", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(loc="best")

        if self.show_labels.isChecked():
            draw_boxes(ax, [("average", self._metrics_text("all plots (average)", actual, predicted))])

    @staticmethod
    def export_labels() -> list[tuple[str, str]]:
        """The in-graph metric boxes the export dialog can size / move / toggle, as (id, name).

        Stable across the current selector state so the controls don't appear and disappear; a box
        the current view doesn't draw simply has no effect. Ids match those passed to ``draw_boxes``.
        """
        return [
            ("held_out", "Held-out box"),
            ("training", "Training box"),
            ("combined", "Combined box"),
            ("average", "Average box"),
        ]

    @staticmethod
    def _draw_fit(ax, actual, predicted, *, color, linestyle, label_prefix) -> None:
        fit = _polyfit_xy(actual, predicted)
        if fit is None:
            return
        slope, intercept = fit
        xs = np.array([actual.min(), actual.max()])
        ax.plot(xs, slope * xs + intercept, color=color, linestyle=linestyle, linewidth=1.8,
                label=f"{label_prefix} (slope {slope:.2f})")

    def export_title(self) -> str:
        return "predicted_vs_actual"

    def export_values(self) -> ChartValues:
        """The scatter's raw points as (``series · actual X`` → predicted Y) rows.

        Mirrors what the live plot shows: the pooled cloud in average mode, else the held-out and
        training series the selectors have enabled. Each row pairs a point's actual value (in the
        label) with its predicted value, so a copy stays unambiguous when both series are present.
        """
        target = getattr(self._history, "target_column", "") if self._history is not None else ""
        unit = f"predicted {target}".strip()
        if self._history is None or active_split_model(self._history) is None:
            return ChartValues(rows=[], unit=unit)

        series: list[tuple[str, object]] = []
        if self.average_splits.isChecked():
            pooled = self._pooled_xy()
            if pooled is not None:
                series.append(("pooled", pooled))
        else:
            if self.val_mode.currentData() != _MODE_OFF:
                series.append(("held-out", self._validation_xy()))
            if self.train_mode.currentData() != _MODE_OFF:
                series.append(("training", self._training_xy()))

        rows: list[tuple[str, float]] = []
        for name, xy in series:
            if xy is None:
                continue
            actual, predicted = xy
            for a, p in zip(np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)):
                rows.append((f"{name} · actual {a:g}", float(p)))
        return ChartValues(rows=rows, unit=unit)

    @staticmethod
    def _metrics_text(scope: str, actual, predicted) -> str:
        """One in-graph metrics box: its scope label + n + the four standard metrics of its points."""
        metrics = standard_metrics(actual, predicted)
        lines = [f"{scope}", f"n={metrics.get('n', 0)}"]
        for key, label, fmt in _METRICS:
            lines.append(f"{label}={fmt(metrics.get(key, float('nan')))}")
        return "\n".join(lines)

    def _split_deltas(self, sm) -> dict:
        """Per-metric held-out − training for ``sm`` under the current augmented-data choice."""
        sel = self._selection
        held = variant_metrics(sm, "held_out", sel.held_out)
        train = variant_metrics(sm, "train", sel.train)
        out: dict[str, float] = {}
        for key, _, _ in _METRICS:
            if key in held and key in train:
                out[key] = float(held[key]) - float(train[key])
        return out

    def _average_deltas(self) -> dict:
        """Per-metric (mean held-out − mean training) across all splits, under the current aug choice.

        Averages each side over the splits' cube cells first, then differences — the all-split
        counterpart of :meth:`_split_deltas`, for the average view's delta label.
        """
        splits = getattr(self._history, "splits", None) or []
        sel = self._selection
        out: dict[str, float] = {}
        for key, _, _ in _METRICS:
            held = [variant_metrics(sm, "held_out", sel.held_out).get(key) for sm in splits]
            train = [variant_metrics(sm, "train", sel.train).get(key) for sm in splits]
            held = [float(v) for v in held if v is not None and np.isfinite(v)]
            train = [float(v) for v in train if v is not None and np.isfinite(v)]
            if held and train:
                out[key] = float(np.mean(held)) - float(np.mean(train))
        return out

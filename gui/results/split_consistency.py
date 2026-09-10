"""Results sub-tab: how stable the model's metrics are across its training splits.

A model is trained over several grouped train/held-out splits; this tab summarises how consistent the
held-out performance is across them and how large the generalisation gap (held-out − training) is.

Four columns sit side by side:

* **held-out metrics:** the mean ± std across all splits for each of rRMSE, R², R, MAPE, and a graph
  of the chosen metric over the split index (with its mean line).
* **training metrics:** the same summary and graph for the per-split *training* metrics, so the term
  being subtracted in the delta is visible directly.
* **delta (held-out − training) metrics:** the mean ± std of the per-split delta for each metric, and
  a graph of the chosen delta over splits.
* **overall metrics:** metrics over *all* of each split's plots (training + held-out) — what the model
  scores when asked to predict every plot it has, in- and out-of-sample together.

Which augmented copies count toward each column is chosen by the host's two "Aug in … metrics"
toggles, passed in as a :class:`~gui.results.variants.VariantSelection`; the numbers come from the
precomputed cube each :class:`~ml.trainer.SplitModel` carries (``metric_variants``), so switching the
toggles is instant and never refits. Older bundles without a cube fall back to their single stored
held-out / training series and show ``n/a`` for overall (and the host greys the toggles out).

A single **y-axis scale** field (0–1) at the top syncs all four graphs: ``1`` opens each axis to its
metric's full range (``±100%`` for rRMSE/MAPE, ``±1`` for R²/R), ``0`` fits tightly to the data, and
values in between interpolate linearly between the two.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from ml.trainer import variant_metrics

from ..export import ChartValues, attach_export_menu
from .variants import VariantSelection

# (key, label, unit-suffix) for the four standard metrics, in display order.
_METRICS = (
    ("rrmse", "rRMSE", "%"),
    ("r2", "R²", ""),
    ("r", "R", ""),
    ("mape", "MAPE", "%"),
)

# The four columns: attribute-suffix → column title. Delta stays next to its two operands; overall
# (all of a split's plots together) sits last.
_KINDS = (
    ("held_out", "Held-out metrics"),
    ("train", "Training metrics"),
    ("delta", "Delta (held-out − training) metrics"),
    ("overall", "Overall metrics"),
)

# Full-range bound per metric unit, reached when the y-axis scale field is 1.
# Percentage metrics open to ±100%, unitless ones (R², R) to ±1.
def _full_bound(suffix: str) -> float:
    return 100.0 if suffix == "%" else 1.0


class _ColumnExport:
    """Export provider for one split-consistency column (held-out / training / delta).

    Each of the three columns is its own graph, so each gets its own provider that re-renders that
    column's metric into an export figure via the owner's pure ``draw_graph_into``.
    """

    def __init__(self, owner: "SplitConsistency", kind: str) -> None:
        self._owner = owner
        self._kind = kind

    def draw_into(self, ax) -> None:
        self._owner.draw_graph_into(ax, self._kind)

    def export_title(self) -> str:
        return f"split_consistency_{self._kind}"

    def export_values(self) -> ChartValues:
        return self._owner.chart_values(self._kind)


class SplitConsistency(QWidget):
    """The first Results sub-tab: per-split metric spread and the generalisation gap."""

    def __init__(self) -> None:
        super().__init__()
        self._history = None
        self._selection = VariantSelection.legacy()

        root = QVBoxLayout(self)
        root.addLayout(self._build_scale_control())
        columns = QHBoxLayout()
        for kind, title in _KINDS:
            columns.addLayout(self._build_column(kind, title), 1)
        root.addLayout(columns)

        self._draw_empty()

    def _build_scale_control(self) -> QHBoxLayout:
        """The top bar: a summary-mode toggle and a 0–1 field that scales every graph's y-axis."""
        row = QHBoxLayout()

        # Toggle for what the summary numbers above each graph report: the mean ± std across all
        # splits (default) or the single selected split's value (the host's Split dropdown).
        self.selected_only = QCheckBox("Show selected split only")
        self.selected_only.setToolTip(
            "Off: the summary above each graph is the mean ± std across all splits.\n"
            "On: it shows only the split picked in the Split dropdown (no spread)."
        )
        self.selected_only.toggled.connect(self._redraw)
        row.addWidget(self.selected_only)
        row.addSpacing(24)

        label = QLabel("Y-axis scale (0 = fit data, 1 = full range ±100% / ±1):")
        label.setStyleSheet("font-weight: bold;")
        field = QLineEdit("0.2")
        validator = QDoubleValidator(0.0, 1.0, 4)
        validator.setNotation(QDoubleValidator.StandardNotation)
        field.setValidator(validator)
        field.setFixedWidth(80)
        field.editingFinished.connect(self._redraw)
        row.addWidget(label)
        row.addWidget(field)
        row.addStretch(1)
        self._scale_field = field
        return row

    def _scale(self) -> float:
        """The y-axis scale field clamped to [0, 1]; 0 if the field is empty/invalid."""
        try:
            return float(min(1.0, max(0.0, float(self._scale_field.text()))))
        except (TypeError, ValueError):
            return 0.0

    def _build_column(self, kind: str, title_text: str) -> QVBoxLayout:
        """One column of the tab: a title, a mean±std summary grid, a metric picker and a graph."""
        col = QVBoxLayout()
        title = QLabel(title_text)
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        col.addWidget(title)

        grid = QGridLayout()
        labels: dict[str, QLabel] = {}
        for row, (key, label, _) in enumerate(_METRICS):
            name = QLabel(f"{label}:")
            name.setStyleSheet("font-weight: bold;")
            value = QLabel("—")
            grid.addWidget(name, row, 0)
            grid.addWidget(value, row, 1)
            labels[key] = value
        col.addLayout(grid)

        picker = QComboBox()
        for key, label, _ in _METRICS:
            picker.addItem(label, key)
        picker.currentIndexChanged.connect(self._redraw)
        col.addWidget(picker)

        figure = Figure(figsize=(4, 2.8), tight_layout=True)
        canvas = FigureCanvas(figure)
        ax = figure.add_subplot(111)
        col.addWidget(canvas, 1)
        attach_export_menu(canvas, _ColumnExport(self, kind))

        # Stash the widgets for this column so refresh/redraw can find them by kind.
        setattr(self, f"_{kind}_labels", labels)
        setattr(self, f"_{kind}_picker", picker)
        setattr(self, f"_{kind}_figure", figure)
        setattr(self, f"_{kind}_canvas", canvas)
        setattr(self, f"_{kind}_ax", ax)
        return col

    # ------------------------------------------------------------------ #
    # Data in                                                            #
    # ------------------------------------------------------------------ #
    def refresh(self, history, selection: VariantSelection | None = None) -> None:
        """Receive the loaded model's :class:`TrainHistory` (or ``None``) and the augmented-data
        choice, then redraw all four columns."""
        self._history = history
        self._selection = selection or VariantSelection.legacy()
        self._redraw()

    # ------------------------------------------------------------------ #
    # Series helpers                                                     #
    # ------------------------------------------------------------------ #
    def _series(self, kind: str, key: str) -> np.ndarray:
        """Per-split values of ``key`` — held-out, training, delta, or overall.

        Reads the precomputed metric cube under the current augmented-data choice. Delta uses the
        *selected* held-out and training variants, so it tracks the toggles too. Missing cells
        (older bundles, or overall on a pre-cube model) yield ``nan``, which the summary and graph
        render as ``n/a`` / gaps rather than failing.
        """
        splits = getattr(self._history, "splits", None) or []
        sel = self._selection
        vals: list[float] = []
        for sm in splits:
            if kind == "held_out":
                v = variant_metrics(sm, "held_out", sel.held_out).get(key, float("nan"))
            elif kind == "train":
                v = variant_metrics(sm, "train", sel.train).get(key, float("nan"))
            elif kind == "overall":
                v = variant_metrics(sm, "overall", sel.overall).get(key, float("nan"))
            else:  # delta = held-out − training, each under its selected variant
                held = variant_metrics(sm, "held_out", sel.held_out).get(key, float("nan"))
                train = variant_metrics(sm, "train", sel.train).get(key, float("nan"))
                v = float(held) - float(train)
            vals.append(float(v))
        return np.asarray(vals, dtype=float)

    def _active_index(self) -> int | None:
        """Position in ``history.splits`` of the selected split (``active_split``/``best_split``).

        Indexes the same ordered list :meth:`_series` builds from, so it picks the matching element
        out of a series. Returns ``None`` when there are no splits or none matches.
        """
        splits = getattr(self._history, "splits", None) or []
        if not splits:
            return None
        want = getattr(self._history, "active_split", 0) or getattr(self._history, "best_split", 0)
        for i, sm in enumerate(splits):
            if getattr(sm, "split", None) == want:
                return i
        return 0  # fall back to the first split if the active one isn't found

    # ------------------------------------------------------------------ #
    # Drawing                                                            #
    # ------------------------------------------------------------------ #
    def _draw_empty(self) -> None:
        for kind, _ in _KINDS:
            for label in getattr(self, f"_{kind}_labels").values():
                label.setText("—")
            ax = getattr(self, f"_{kind}_ax")
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.5, 0.5, "Train and load a model to see split consistency.",
                    ha="center", va="center", color="gray", transform=ax.transAxes)
            getattr(self, f"_{kind}_canvas").draw_idle()

    def _redraw(self) -> None:
        if self._history is None or not getattr(self._history, "splits", None):
            self._draw_empty()
            return
        for kind, _ in _KINDS:
            self._update_summary(kind)
            self._update_graph(kind)

    def _update_summary(self, kind: str) -> None:
        labels = getattr(self, f"_{kind}_labels")
        selected_only = self.selected_only.isChecked()
        idx = self._active_index() if selected_only else None
        for key, _, suffix in _METRICS:
            vals = self._series(kind, key)
            if selected_only:
                # One split's value (no spread) — the split picked in the host's Split dropdown.
                val = float(vals[idx]) if idx is not None and idx < vals.size else float("nan")
                labels[key].setText(f"{val:.4g}{suffix}" if np.isfinite(val) else "n/a")
                continue
            mean = float(np.nanmean(vals)) if vals.size else float("nan")
            std = float(np.nanstd(vals)) if vals.size else float("nan")
            if np.isfinite(mean):
                labels[key].setText(f"{mean:.4g}{suffix} ± {std:.3g}{suffix}")
            else:
                labels[key].setText("n/a")

    def _update_graph(self, kind: str) -> None:
        self.draw_graph_into(getattr(self, f"_{kind}_ax"), kind)
        getattr(self, f"_{kind}_canvas").draw_idle()

    def draw_graph_into(self, ax, kind: str) -> None:
        """Draw the ``kind`` column's per-split metric graph into ``ax`` (live or export figure).

        Pure render shared by the live canvas and the export layer: it reads the chosen metric from
        that column's picker and the current history, but touches no widgets. Font sizes follow the
        axes' rcParams so an export renders at the standardised point size.
        """
        picker = getattr(self, f"_{kind}_picker")
        key = picker.currentData()
        label = next(lbl for k, lbl, _ in _METRICS if k == key)
        suffix = next(s for k, _, s in _METRICS if k == key)
        vals = self._series(kind, key)
        xs = np.arange(1, vals.size + 1)

        ax.clear()
        finite = np.isfinite(vals)
        if finite.any():
            ax.plot(xs[finite], vals[finite], marker="o", markersize=4, color="#1f77b4")
            mean = float(np.nanmean(vals))
            ax.axhline(mean, color="#d73027", linestyle="--", linewidth=1,
                       label=f"mean {mean:.3g}{suffix}")
            # Reference line at y=0 so the sign of each value (and of the delta) reads at a glance.
            ax.axhline(0.0, color="black", linestyle=":", linewidth=1)
            ax.legend(loc="best")
            self._apply_yscale(ax, key, suffix)
        else:
            ax.text(0.5, 0.5, "n/a (not recorded for this model)", ha="center", va="center",
                    color="gray", transform=ax.transAxes)
        ax.set_xlabel("split")
        ax.set_ylabel(f"{'Δ ' if kind == 'delta' else ''}{label}{f' ({suffix})' if suffix else ''}")
        if vals.size:
            ax.set_xticks(xs)
        ax.grid(True, which="major", color="0.9", linewidth=0.6)
        ax.set_axisbelow(True)

    def chart_values(self, kind: str) -> ChartValues:
        """The ``kind`` column's per-split (``split N`` → metric value) rows for "Copy values"."""
        picker = getattr(self, f"_{kind}_picker")
        key = picker.currentData()
        label = next(lbl for k, lbl, _ in _METRICS if k == key)
        suffix = next(s for k, _, s in _METRICS if k == key)
        unit = f"{'Δ ' if kind == 'delta' else ''}{label}{f' ({suffix})' if suffix else ''}"
        vals = self._series(kind, key)
        rows = [(f"split {i + 1}", float(v)) for i, v in enumerate(vals)]
        return ChartValues(rows=rows, unit=unit)

    def _apply_yscale(self, ax, key: str, suffix: str) -> None:
        """Set y-limits by blending a *shared* auto-fit range toward the metric's full range.

        The auto-fit range is pooled across every panel that currently shows ``key`` (and y=0 is
        always included), so all same-metric panels get identical limits — they no longer scale
        differently just because their data spans differ. At scale ``0`` the axis is that shared
        data range; at ``1`` it opens to the symmetric full range (``±100`` for percentage metrics,
        ``±1`` for unitless ones); in between it interpolates linearly.
        """
        t = self._scale()
        lo_data, hi_data = self._shared_range(key)
        if not (np.isfinite(lo_data) and np.isfinite(hi_data)):
            return
        bound = _full_bound(suffix)
        lo = (1.0 - t) * lo_data + t * (-bound)
        hi = (1.0 - t) * hi_data + t * bound
        if hi <= lo:  # degenerate (e.g. a flat series at scale 0) — pad so matplotlib is happy
            pad = abs(hi) * 0.05 or 1.0
            lo, hi = lo - pad, hi + pad
        ax.set_ylim(lo, hi)

    def _shared_range(self, key: str) -> tuple[float, float]:
        """Common data range for ``key`` over every panel currently showing it, including y=0.

        Pooling the panels that display the same metric is what makes their axes line up; y=0 is
        folded in so the new zero reference line is always on-screen and panels stay comparable.
        """
        lo, hi = 0.0, 0.0  # always include the zero reference line
        for kind, _ in _KINDS:
            if getattr(self, f"_{kind}_picker").currentData() != key:
                continue
            vals = self._series(kind, key)
            finite = vals[np.isfinite(vals)]
            if finite.size:
                lo = min(lo, float(finite.min()))
                hi = max(hi, float(finite.max()))
        if hi == lo:
            return float("nan"), float("nan")
        margin = (hi - lo) * 0.05
        return lo - margin, hi + margin

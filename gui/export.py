"""One shared export layer for every results graph and the field map.

Each on-screen visualisation is sized for its dock panel, so an OS screenshot of one comes out at
a different size / font than another — useless for a thesis where every figure must look the same.
This module decouples the *export* render from the live widget: a graph re-draws itself into a
**fresh** figure at one standardised :class:`ExportSpec` (size / DPI / font), so two exports come
out pixel-consistent regardless of how big each panel happened to be on screen.

Two kinds of visualisation plug in through one menu:

* **matplotlib graphs** (loss curves, scatter, distributions, split-consistency) expose a
  ``draw_into(ax)`` callable; :func:`render_matplotlib` runs it on a throwaway figure sized by the
  spec, inside an :func:`matplotlib.rc_context` so the live figure's styling is never touched. The
  *style* (per-element fonts, element scaling, legend placement, padding) lives here, not in the
  provider — so one spec controls how every graph looks, and the graphs stay style-agnostic. A graph
  that draws in-graph annotation boxes routes them through :func:`draw_boxes` with a stable id per
  box and declares them via an optional ``export_labels()``; the export dialog then sizes / moves /
  toggles each box independently (per-box overrides on the spec), so multi-label graphs are handled
  the same universal way as fonts and the legend.
* **the Qt field map** (:class:`gui.results.field_canvas.FieldCanvas`) hands back a
  :class:`gui.results.field_canvas.FieldExport` from ``build_export(spec)``; :func:`render_field_image`
  renders its :class:`QGraphicsScene` and bakes the colour legend + title back in (those are
  viewport overlays the scene render can't see), to an image or a vector surface.

:func:`attach_export_menu` wires a right-click menu onto any canvas (Export… opens the merged
preview-and-settings :class:`ExportDialog`, plus quick Copy / Save as shortcuts), so adding export
to a graph is a single call. The :class:`ExportSpec` is seeded from and persisted to its own
``export_spec.json`` sidecar (see :func:`seed_spec_from_config`). PNG/SVG/PDF are chosen by the file
extension, and Copy puts a PNG on the clipboard.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Callable, Protocol

from PySide6.QtCore import QEvent, QRectF, Qt, QTimer
from PySide6.QtGui import QGuiApplication, QImage, QPainter
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

# A centimetre is this many inches; matplotlib figsize is in inches, but thesis figures are
# specified in cm (column widths), so the spec stores cm and converts here.
_CM_PER_INCH = 2.54

# Legend placements offered in the dialog. "outside …" anchors the legend beyond the axes and lets
# constrained-layout reserve room, so it can never overlap the data; the rest are normal in-axes
# matplotlib ``loc`` strings. Maps the dialog label -> (loc, bbox_to_anchor or None).
_LEGEND_PLACEMENTS: dict[str, tuple[str, tuple[float, float] | None]] = {
    "best (auto)": ("best", None),
    "outside right": ("center left", (1.02, 0.5)),
    "outside top": ("lower center", (0.5, 1.02)),
    "outside bottom": ("upper center", (0.5, -0.12)),
    "upper left": ("upper left", None),
    "upper right": ("upper right", None),
    "lower left": ("lower left", None),
    "lower right": ("lower right", None),
    "hidden": ("__hidden__", None),
}

_COLORBAR_POSITIONS = ("right", "left", "top-right", "bottom-right")
_BACKGROUNDS = ("white", "transparent")

# In-graph annotation boxes (e.g. the metric boxes on the predicted-vs-actual plot) can be parked in
# any axes corner. Maps the dialog label -> (x, y, ha, va) in axes fraction; boxes anchored to the
# same corner stack away from that corner (the box helper handles the stacking direction).
_LABEL_CORNERS: dict[str, tuple[float, float, str, str]] = {
    "top-left": (0.02, 0.98, "left", "top"),
    "top-right": (0.98, 0.98, "right", "top"),
    "bottom-left": (0.02, 0.02, "left", "bottom"),
    "bottom-right": (0.98, 0.02, "right", "bottom"),
}


@dataclass(frozen=True)
class ExportSpec:
    """The one standardised style every visualisation exports at.

    Defaults target a **single-column thesis figure**: ~8.5 cm wide, 300 DPI (print-quality
    raster; PDF/SVG are vector regardless). The same spec is applied to every graph, so exported
    figures share size and typography no matter their on-screen panel size. Fonts are per-element
    (title / axis / tick / legend) rather than one global size, so a small column figure can keep a
    readable title without an oversized legend swamping the data.
    """

    # Geometry.
    width_cm: float = 8.5
    height_cm: float = 6.0
    dpi: int = 300

    # Per-element font sizes (pt). Tuned for the default 8.5 cm figure: a slightly larger title over
    # compact axis/tick/legend text. Kept small on purpose — at 8.5 cm, 9–10 pt fonts plus a legend
    # overflow the tiny figure (that was the original bug); these sizes leave room for everything.
    title_pt: float = 9.0
    axis_pt: float = 8.0
    tick_pt: float = 7.0
    legend_pt: float = 6.5

    # Legend placement (a key into _LEGEND_PLACEMENTS). Default pushes the legend outside the axes
    # on the right so it can never overlap data in a tight figure.
    legend_loc: str = "outside right"

    # Multiplies scatter-marker area, line widths and bar thickness so elements are not oversized in
    # a small figure; constrained-layout padding (in font-size units) keeps text off the edges. The
    # 0.12 pad is what stops the title / y-label clipping at the figure edge in a dense plot.
    element_scale: float = 1.0
    pad: float = 0.12

    # In-graph annotation boxes (metric boxes etc.). ``label_pt`` is the baseline font for every box;
    # ``label_overrides`` carries per-box tweaks keyed by the provider's stable label id —
    # ``{"visible": bool, "scale": float, "corner": str}`` (any subset). A box absent from the dict
    # uses the defaults (visible, scale 1.0, the provider's default corner). Kept generic so every
    # graph that draws boxes is sized / placed / toggled the same way through the export dialog.
    label_pt: float = 8.0
    label_overrides: dict = field(default_factory=dict)

    # ---- Field-map-only controls (ignored by matplotlib graphs) ----
    map_margin_frac: float = 0.06       # whitespace around the field, as a fraction of its size
    map_offset_x: float = 0.0           # recentre the field horizontally (fraction of width)
    map_offset_y: float = 0.0           # recentre the field vertically (fraction of height)
    map_rotation_deg: float = 0.0       # whole-field rotation in the export
    map_zoom: float = 1.0               # zoom factor (>1 zooms in), like scrolling the live canvas
    show_colorbar: bool = True
    colorbar_pos: str = "right"
    colorbar_width_frac: float = 0.16   # colorbar panel width as a fraction of the image width
    colorbar_tick_pt: float = 9.0
    colorbar_label_pt: float = 10.0
    plot_label_scale: float = 1.0       # scales the value labels drawn inside each plot
    show_title: bool = True
    title_overlay_pt: float = 12.0
    background: str = "white"

    @property
    def size_inches(self) -> tuple[float, float]:
        return (self.width_cm / _CM_PER_INCH, self.height_cm / _CM_PER_INCH)

    @property
    def px_size(self) -> tuple[int, int]:
        """Pixel dimensions at this spec's DPI (for the Qt-scene raster export)."""
        w_in, h_in = self.size_inches
        return (max(1, round(w_in * self.dpi)), max(1, round(h_in * self.dpi)))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "ExportSpec":
        """Build a spec from a (possibly partial / stale) dict, defaulting any missing key.

        Tolerant on purpose: an old ``export_spec.json`` without the new fields, or with an unknown
        key, still loads — known fields are taken, the rest fall back to the dataclass defaults.
        """
        if not isinstance(data, dict):
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


# The session-wide spec. Edited via the export dialog; every menu reads this live, so a change
# applies to all graphs at once. Seeded from config at startup (see seed_spec_from_config).
SPEC = ExportSpec()

# Save-dialog filter -> the extension we write. PDF and SVG are vector; PNG is the raster fallback.
_FILTERS = "PDF (*.pdf);;SVG (*.svg);;PNG (*.png)"


# --------------------------------------------------------------------------- #
# Provider protocol                                                           #
# --------------------------------------------------------------------------- #
class ExportProvider(Protocol):
    """What :func:`attach_export_menu` needs from a visualisation.

    A matplotlib graph supplies ``draw_into(ax)``; the field map supplies ``build_export(spec)``
    instead. A provider implements exactly one of the two paths, plus ``export_title``.

    A provider whose chart maps a category label to a single number (bar charts, point series)
    may *additionally* implement the optional ``export_values()`` -> :class:`ChartValues`. When it
    does, the right-click menu gains a "Copy values" submenu that copies the raw x/y pairs as text.
    Providers where this makes no sense (the field map, a value distribution) simply omit it, so no
    such menu appears for them.
    """

    def export_title(self) -> str:
        """A short default file name (no extension) for the Save dialog, e.g. ``"loss_curve"``."""
        ...


@dataclass(frozen=True)
class ChartValues:
    """The raw x/y data behind a chart, for the "Copy values" menu.

    ``rows`` is ``[(label, value), …]`` in plot order; ``value`` may be ``nan`` for a bar the chart
    drew as "n/a". ``unit`` is the y-axis description copied after every number (e.g. ``"rRMSE (%)"``
    or ``""`` for a unitless metric). The text formatting (one line vs. a pasteable table) lives in
    :func:`_values_to_text`, so a provider only has to hand back the numbers.
    """

    rows: list[tuple[str, float]]
    unit: str = ""


def _mean(values: ChartValues) -> float | None:
    """The mean of the finite values (``None`` if every value is non-finite / n/a)."""
    import math

    nums = [v for _, v in values.rows
            if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(nums) / len(nums) if nums else None


# U+0332 COMBINING LOW LINE: appended after each character it underlines. Plain text stays
# pasteable everywhere, but rich-text targets (Word, etc.) render it as a continuous underline —
# the only way to "underline" inside a clipboard string that has no formatting of its own.
_COMBINING_UNDERLINE = "̲"


def _underline(text: str) -> str:
    """Return ``text`` with a combining underline under every (non-space) character."""
    return "".join(c + _COMBINING_UNDERLINE if not c.isspace() else c for c in text)


def _values_to_text(values: ChartValues, *, as_table: bool, with_average: bool = False,
                    underline_average: bool = False) -> str:
    """Render :class:`ChartValues` as clipboard text.

    ``as_table=False`` -> one comma-separated line ("PLS: 14.4 rRMSE (%), Elastic: 13.1 rRMSE (%)").
    ``as_table=True``  -> tab-separated ``label<TAB>value`` rows (a unit header row first if there is
    a unit), which pastes into a spreadsheet as two columns. A non-finite value renders as "n/a".

    ``with_average=True`` appends the mean of the finite values at the end — a trailing
    "Average: 0.43 R²" cell on the line, or an "Average<TAB>0.43" row at the foot of the table.
    ``underline_average=True`` underlines that appended cell/row via combining low lines (renders
    as an underline in rich-text targets; ignored if ``with_average`` is off).
    """
    import math

    unit = values.unit.strip()
    underline = with_average and underline_average

    def num(v: float) -> str:
        return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.4g}"

    avg = _mean(values) if with_average else None

    if as_table:
        lines = ([f"\t{unit}"] if unit else [])
        lines += [f"{label}\t{num(v)}" for label, v in values.rows]
        if with_average:
            row = f"Average\t{num(avg)}"
            lines.append(_underline(row) if underline else row)
        return "\n".join(lines)

    def cell(label: str, v: float) -> str:
        text = num(v)
        # Drop the unit after an "n/a" so it doesn't read like "n/a rRMSE (%)".
        if unit and text != "n/a":
            text = f"{text} {unit}"
        return f"{label}: {text}"

    cells = [cell(label, v) for label, v in values.rows]
    if with_average:
        avg_cell = cell("Average", avg)
        cells.append(_underline(avg_cell) if underline else avg_cell)
    return ", ".join(cells)


def _copy_values(provider, *, as_table: bool, with_average: bool = False,
                 underline_average: bool = False) -> None:
    """Copy the provider's raw chart values to the clipboard as text (line or table)."""
    values = provider.export_values()
    if values is None or not values.rows:
        return
    QGuiApplication.clipboard().setText(_values_to_text(
        values, as_table=as_table, with_average=with_average, underline_average=underline_average))


# --------------------------------------------------------------------------- #
# Matplotlib path                                                             #
# --------------------------------------------------------------------------- #
def render_matplotlib(draw_into: Callable[[object], None], spec: ExportSpec):
    """Render ``draw_into(ax)`` onto a fresh figure sized by ``spec``; return the ``Figure``.

    The figure is created independently of any on-screen canvas and styled inside an
    ``rc_context``, so the live widget's figure (and global rcParams) are untouched. The caller
    owns the returned figure (save it, embed it in a preview, then drop it).

    The provider's ``draw_into`` is style-agnostic (it just plots data and labels); this function
    owns the *export style*: per-element font sizes via rcParams, element scaling and legend
    placement applied to the axes *after* the draw (so they override whatever ``draw_into`` set),
    and constrained-layout padding so nothing clips. That keeps every provider untouched while the
    one spec controls how the figure looks.
    """
    import matplotlib
    from matplotlib.figure import Figure

    rc = {
        # font.size seeds anything unscoped (label_pt drives the annotation boxes via draw_boxes).
        "font.size": spec.axis_pt,
        "axes.titlesize": spec.title_pt,
        "axes.labelsize": spec.axis_pt,
        "xtick.labelsize": spec.tick_pt,
        "ytick.labelsize": spec.tick_pt,
        "legend.fontsize": spec.legend_pt,
        "savefig.dpi": spec.dpi,
        "figure.dpi": spec.dpi,
    }
    with matplotlib.rc_context(rc):
        figure = Figure(figsize=spec.size_inches, dpi=spec.dpi, layout="constrained")
        try:
            figure.get_layout_engine().set(w_pad=spec.pad, h_pad=spec.pad)
        except Exception:
            pass
        ax = figure.add_subplot(111)
        # Publish the spec so a provider's draw_into can read per-label overrides via draw_boxes
        # without changing the draw_into(ax) contract every provider shares.
        global _ACTIVE_SPEC
        _ACTIVE_SPEC = spec
        try:
            draw_into(ax)
        finally:
            _ACTIVE_SPEC = None
        _scale_elements(ax, spec.element_scale)
        _enforce_tick_size(ax, spec.tick_pt)
        _place_legend(ax, spec.legend_loc)
        _wrap_text(ax)
    return figure


# The spec in effect for the draw_into currently running, so draw_boxes can read its label overrides.
# Set only for the duration of render_matplotlib's draw_into; None on the live canvas.
_ACTIVE_SPEC: "ExportSpec | None" = None


def _enforce_tick_size(ax, tick_pt: float) -> None:
    """Force both axes' tick-label size to ``tick_pt``, overriding whatever the provider set.

    The rc_context already seeds ``xtick/ytick.labelsize``, but some providers call
    ``set_xticklabels(..., fontsize=…)`` / ``set_yticklabels`` themselves, which bakes a fixed size
    into the label artists and ignores the spec. Re-applying via ``tick_params`` after the draw wins
    over those overrides and also sets the axis default so any ticks the locator regenerates at save
    time use the spec size too — so the slider works on every graph, on both axes.
    """
    try:
        ax.tick_params(axis="both", which="both", labelsize=tick_pt)
        for label in (*ax.get_xticklabels(), *ax.get_yticklabels(),
                      *ax.get_xticklabels(minor=True), *ax.get_yticklabels(minor=True)):
            label.set_fontsize(tick_pt)
    except Exception:
        pass


def _wrap_text(ax) -> None:
    """Let the title and axis labels wrap to the figure width instead of clipping at the edge.

    A long title (e.g. "Permutation importance (Held-out plots (leakage-free))") overruns a narrow
    thesis figure and gets cut off — constrained-layout reserves vertical room for it but never
    shrinks or wraps it. Enabling ``wrap`` makes matplotlib break it across lines so it always fits.
    """
    for artist in (ax.title, ax.xaxis.label, ax.yaxis.label):
        try:
            artist.set_wrap(True)
        except Exception:
            pass


def _label_override(spec: "ExportSpec | None", label_id: str) -> dict:
    """The per-label override dict for ``label_id`` (``{}`` if none / no spec)."""
    if spec is None:
        return {}
    ov = spec.label_overrides.get(label_id)
    return ov if isinstance(ov, dict) else {}


def draw_boxes(ax, boxes: list[tuple[str, str]], *, default_corner: str = "top-left") -> None:
    """Draw a provider's in-graph annotation boxes, honouring the live export label overrides.

    ``boxes`` is ``[(label_id, text), …]`` where ``label_id`` is a stable key (so the export dialog
    can offer per-box size / position / on-off controls — see ``export_labels``). For each box the
    active :data:`_ACTIVE_SPEC` (set during an export render) supplies an optional override:
    ``visible`` (drop the box), ``scale`` (multiply ``label_pt``) and ``corner`` (which axes corner
    to anchor in). Boxes sharing a corner stack inward from it. On the live canvas the spec is None,
    so every box shows at the rcParams font and the provider's ``default_corner`` — unchanged
    behaviour. One helper means every graph's boxes size / move / toggle the same way.
    """
    import matplotlib as mpl

    spec = _ACTIVE_SPEC
    base_pt = spec.label_pt if spec is not None else mpl.rcParams.get("font.size", 10.0)
    axes_h_in = ax.figure.get_figheight() * ax.get_position().height

    # Group the (still-visible) boxes by the corner they end up in, so each corner stacks on its own.
    by_corner: dict[str, list[tuple[str, float]]] = {}
    for label_id, text in boxes:
        ov = _label_override(spec, label_id)
        if ov.get("visible") is False:
            continue
        corner = ov.get("corner") or default_corner
        if corner not in _LABEL_CORNERS:
            corner = default_corner
        pt = base_pt * float(ov.get("scale", 1.0))
        by_corner.setdefault(corner, []).append((text, pt))

    for corner, items in by_corner.items():
        x, y0, ha, va = _LABEL_CORNERS[corner]
        downward = va == "top"  # stack down from a top corner, up from a bottom one
        y = y0
        for text, pt in items:
            ax.text(x, y, text, transform=ax.transAxes, va=va, ha=ha, fontsize=pt,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            line_frac = (pt * 1.3) / 72.0 / max(axes_h_in, 1e-6)  # 1.3 ~ line spacing
            block = line_frac * (text.count("\n") + 2) + 0.02     # +1 line for the box padding
            y += -block if downward else block


def _scale_elements(ax, scale: float) -> None:
    """Multiply scatter-marker area, line widths and bar/patch edges on ``ax`` by ``scale``.

    Done post-draw so providers stay style-agnostic: they pick relative sizes, the export scales
    them all together so points/bars/lines are not oversized in a small thesis figure.
    """
    if scale == 1.0:
        return
    for coll in ax.collections:  # scatter PathCollections carry sizes in points-squared
        try:
            sizes = coll.get_sizes()
            if sizes is not None and len(sizes):
                coll.set_sizes([s * scale for s in sizes])
            coll.set_linewidth([lw * scale for lw in _as_list(coll.get_linewidth())])
        except Exception:
            pass
    for line in ax.lines:
        try:
            line.set_linewidth(line.get_linewidth() * scale)
            ms = line.get_markersize()
            if ms:
                line.set_markersize(ms * scale)
        except Exception:
            pass
    for patch in ax.patches:  # bars / boxes
        try:
            patch.set_linewidth(patch.get_linewidth() * scale)
        except Exception:
            pass


def _as_list(value) -> list:
    try:
        return list(value)
    except TypeError:
        return [value]


def _place_legend(ax, legend_loc: str) -> None:
    """Re-place (or hide) the axes legend per the spec, overriding the provider's own ``ax.legend``.

    Providers call ``ax.legend(loc="best")`` themselves; running here afterwards lets the user move
    the legend outside the data (the default) or hide it without touching any provider. "Outside"
    anchors are reserved by constrained-layout, so the legend never overlaps the plot.
    """
    loc, anchor = _LEGEND_PLACEMENTS.get(legend_loc, _LEGEND_PLACEMENTS["best (auto)"])
    existing = ax.get_legend()
    if loc == "__hidden__":
        if existing is not None:
            existing.remove()
        return
    if existing is None and not ax.get_legend_handles_labels()[0]:
        return  # nothing labelled to show
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    if anchor is not None:
        ax.legend(handles, labels, loc=loc, bbox_to_anchor=anchor, borderaxespad=0.0)
    else:
        ax.legend(handles, labels, loc=loc)


def _save_matplotlib(figure, path: Path, spec: ExportSpec) -> None:
    # No bbox_inches="tight": constrained-layout already keeps everything inside the figure, and a
    # tight bbox would crop to a different rect than the preview rasterises — so the saved file
    # would not match the preview. Saving the figure as-is keeps them pixel-identical.
    figure.savefig(path, dpi=spec.dpi)


def _matplotlib_to_image(figure, spec: ExportSpec) -> QImage:
    """Rasterise a matplotlib figure to a :class:`QImage` (for clipboard copy and preview)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    width, height = canvas.get_width_height()
    buf = canvas.buffer_rgba()
    # QImage does not own the buffer, so copy() before the numpy buffer is freed.
    return QImage(bytes(buf), width, height, QImage.Format_RGBA8888).copy()


# --------------------------------------------------------------------------- #
# Qt-scene path (the field map)                                               #
# --------------------------------------------------------------------------- #
def _field_layout(spec: ExportSpec, width: int, height: int) -> tuple[QRectF, QRectF | None, QRectF | None]:
    """Split the ``width × height`` canvas into (field rect, colorbar rect, title rect).

    The colorbar is reserved a column on the chosen edge so it never overlaps the field; the title
    (if shown) takes a strip at the top. The field gets whatever is left. Returns the rects in
    device pixels (or None for an absent piece).
    """
    cbar_w = spec.colorbar_width_frac * width if spec.show_colorbar else 0.0
    title_h = (spec.title_overlay_pt * 2.2 * (spec.dpi / 72.0)) if (spec.show_title and spec.title_overlay_pt) else 0.0
    title_rect = QRectF(0, 0, width, title_h) if title_h else None

    top = title_h
    body_h = height - top
    if not spec.show_colorbar:
        return QRectF(0, top, width, body_h), None, title_rect

    pos = spec.colorbar_pos
    if pos == "left":
        cbar = QRectF(0, top, cbar_w, body_h)
        field = QRectF(cbar_w, top, width - cbar_w, body_h)
    elif pos == "top-right":
        cbar = QRectF(width - cbar_w, top, cbar_w, body_h * 0.5)
        field = QRectF(0, top, width - cbar_w, body_h)
    elif pos == "bottom-right":
        cbar = QRectF(width - cbar_w, top + body_h * 0.5, cbar_w, body_h * 0.5)
        field = QRectF(0, top, width - cbar_w, body_h)
    else:  # "right"
        cbar = QRectF(width - cbar_w, top, cbar_w, body_h)
        field = QRectF(0, top, width - cbar_w, body_h)
    return field, cbar, title_rect


def _paint_field(painter: QPainter, export, spec: ExportSpec, width: int, height: int) -> None:
    """Paint the whole field figure (scene + colorbar + title) onto ``painter`` at ``width×height``.

    Shared by the raster preview/copy and the SVG/PDF vector save, so every output is identical.
    ``export`` is a :class:`gui.results.field_canvas.FieldExport`.
    """
    from .results.field_canvas import draw_colorbar

    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.TextAntialiasing, True)
    field_rect, cbar_rect, title_rect = _field_layout(spec, width, height)

    # Field: render the scene into its rect with the live Y-flip / rotation baked in.
    painter.save()
    if export.y_flip or export.rotation_deg:
        painter.translate(field_rect.center())
        if export.rotation_deg:
            painter.rotate(export.rotation_deg)
        painter.scale(1.0, -1.0 if export.y_flip else 1.0)
        painter.translate(-field_rect.center())
    export.scene.render(painter, field_rect, export.source_rect, Qt.KeepAspectRatio)
    painter.restore()

    if cbar_rect is not None:
        draw_colorbar(painter, cbar_rect, export.stops, export.lo, export.hi, export.colorbar_label,
                      tick_pt=spec.colorbar_tick_pt * (spec.dpi / 72.0),
                      label_pt=spec.colorbar_label_pt * (spec.dpi / 72.0))
    if title_rect is not None and export.title:
        painter.save()
        painter.setPen(Qt.black)
        from PySide6.QtGui import QFont
        f = QFont()
        f.setPointSizeF(spec.title_overlay_pt * (spec.dpi / 72.0))
        f.setBold(True)
        painter.setFont(f)
        painter.drawText(title_rect, Qt.AlignCenter, export.title)
        painter.restore()


def render_field_image(export, spec: ExportSpec) -> QImage:
    """Rasterise the field-map figure (scene + baked-in colorbar + title) to a :class:`QImage`."""
    width, height = spec.px_size
    image = QImage(width, height, QImage.Format_ARGB32)
    image.fill(Qt.transparent if spec.background == "transparent" else Qt.white)
    painter = QPainter(image)
    _paint_field(painter, export, spec, width, height)
    painter.end()
    return image


def _save_field(export, path: Path, spec: ExportSpec) -> None:
    """Save the field figure. PNG saves the raster; SVG/PDF re-render to a vector surface."""
    ext = path.suffix.lower()
    if ext == ".svg":
        from PySide6.QtCore import QSize
        from PySide6.QtSvg import QSvgGenerator

        width, height = spec.px_size
        gen = QSvgGenerator()
        gen.setFileName(str(path))
        gen.setSize(QSize(width, height))
        gen.setViewBox(QRectF(0, 0, width, height))
        painter = QPainter(gen)
        _paint_field(painter, export, spec, width, height)
        painter.end()
    elif ext == ".pdf":
        from PySide6.QtCore import QSizeF
        from PySide6.QtGui import QPageSize
        from PySide6.QtPrintSupport import QPrinter

        printer = QPrinter(QPrinter.HighResolution)
        printer.setOutputFormat(QPrinter.PdfFormat)
        printer.setOutputFileName(str(path))
        # Page size is the physical figure size in points (1 pt = 1/72"), independent of pixels.
        w_in, h_in = spec.size_inches
        printer.setPageSize(QPageSize(QSizeF(w_in * 72.0, h_in * 72.0), QPageSize.Unit.Point))
        painter = QPainter(printer)
        rect = painter.viewport()
        _paint_field(painter, export, spec, rect.width(), rect.height())
        painter.end()
    else:
        render_field_image(export, spec).save(str(path), "PNG")


# --------------------------------------------------------------------------- #
# Rendering a provider (both kinds) + save / copy                             #
# --------------------------------------------------------------------------- #
def _is_field(provider) -> bool:
    """True if ``provider`` is the field map (the Qt-scene path), false for a matplotlib graph."""
    return hasattr(provider, "build_export")


def _provider_image(provider, spec: ExportSpec) -> QImage:
    """Render ``provider`` to a :class:`QImage` at ``spec`` (works for both provider kinds)."""
    if _is_field(provider):
        return render_field_image(provider.build_export(spec), spec)
    figure = render_matplotlib(provider.draw_into, spec)
    return _matplotlib_to_image(figure, spec)


def _copy(provider, spec: ExportSpec | None = None) -> None:
    QGuiApplication.clipboard().setImage(_provider_image(provider, spec or SPEC))


def _save(widget: QWidget, provider, spec: ExportSpec | None = None) -> None:
    spec = spec or SPEC
    # Figure export has no configured default folder (it is not part of a preset); the dialog opens
    # wherever the OS last left it, pre-filled with a sensible default file name.
    default_name = f"{provider.export_title()}.pdf"
    start = default_name
    path_str, selected = QFileDialog.getSaveFileName(widget, "Save figure", start, _FILTERS)
    if not path_str:
        return
    path = Path(path_str)
    if not path.suffix:  # the user typed a bare name; take the extension from the chosen filter
        ext = {"PDF (*.pdf)": ".pdf", "SVG (*.svg)": ".svg", "PNG (*.png)": ".png"}.get(selected, ".pdf")
        path = path.with_suffix(ext)

    if _is_field(provider):
        _save_field(provider.build_export(spec), path, spec)
    else:
        figure = render_matplotlib(provider.draw_into, spec)
        _save_matplotlib(figure, path, spec)


# --------------------------------------------------------------------------- #
# The right-click menu                                                        #
# --------------------------------------------------------------------------- #
# Session-wide Copy-values toggles, shared by every Copy-values menu so the choice sticks across
# right-clicks and graphs. One-element lists so the checkbox lambdas can flip them in place without
# a module-level ``global``. ``_underline_average`` only takes effect while ``_append_average`` is on.
_append_average: list[bool] = [False]
_underline_average: list[bool] = [False]


def attach_export_menu(widget: QWidget, provider) -> None:
    """Install a right-click export menu on ``widget`` driven by ``provider``.

    ``provider`` is either:

    * a matplotlib graph exposing ``draw_into(ax)`` + ``export_title()``, or
    * the field map exposing ``build_export(spec)`` + ``export_title()``.

    The menu opens the merged Export dialog (live preview + all controls), plus quick Copy / Save
    shortcuts that use the current shared :data:`SPEC`.
    """
    widget.setContextMenuPolicy(Qt.CustomContextMenu)

    def show_menu(pos):
        menu = QMenu(widget)
        menu.addAction("Export…", lambda: _preview(widget, provider))
        menu.addAction("Copy image", lambda: _copy(provider))
        # Only charts that expose their raw x/y data (bars, point series) offer "Copy values"; the
        # field map and value distributions don't implement export_values, so they get no submenu.
        if callable(getattr(provider, "export_values", None)):
            values_menu = menu.addMenu("Copy values")
            values_menu.addAction(
                "As line", lambda: _copy_values(
                    provider, as_table=False,
                    with_average=_append_average[0], underline_average=_underline_average[0]))
            values_menu.addAction(
                "As table (TSV)", lambda: _copy_values(
                    provider, as_table=True,
                    with_average=_append_average[0], underline_average=_underline_average[0]))
            # The toggles sit last (below a separator) so the two copy actions aren't easy to
            # mis-click; when on, both append a trailing "Average: …" cell / row (mean of the
            # finite values), optionally underlined. Session-wide so the choice sticks between
            # right-clicks. "Underline" only bites when "Append average" is on.
            values_menu.addSeparator()
            avg = values_menu.addAction("Append average")
            avg.setCheckable(True)
            avg.setChecked(_append_average[0])
            avg.toggled.connect(lambda v: _append_average.__setitem__(0, v))
            ul = values_menu.addAction("Underline average")
            ul.setCheckable(True)
            ul.setChecked(_underline_average[0])
            ul.setEnabled(_append_average[0])
            ul.toggled.connect(lambda v: _underline_average.__setitem__(0, v))
            avg.toggled.connect(ul.setEnabled)
        menu.addAction("Save as…", lambda: _save(widget, provider))
        menu.exec(widget.mapToGlobal(pos))

    widget.customContextMenuRequested.connect(show_menu)


def _preview(widget: QWidget, provider) -> None:
    ExportDialog(widget, provider).show()


def export_preview(widget: QWidget, provider) -> None:
    """Open the export dialog for ``provider`` (the public entry the field-map menu uses)."""
    _preview(widget, provider)


def open_export_settings(parent: QWidget) -> bool:
    """Back-compat shim: the settings now live in the export dialog. Returns False (no standalone)."""
    return False


# --------------------------------------------------------------------------- #
# Presets                                                                      #
# --------------------------------------------------------------------------- #
def _preset(width_cm, height_cm, dpi, base_pt, pad=0.12) -> ExportSpec:
    """A preset spec: a balanced font hierarchy derived from one ``base_pt`` axis size, plus geometry.

    The title sits one point above the axis label; ticks and legend a touch below — the same
    hierarchy as the defaults. ``base_pt`` is deliberately small for the narrow single-column figure
    (more room for the legend) and grows with the figure for the wider presets.
    """
    return ExportSpec(
        width_cm=width_cm, height_cm=height_cm, dpi=dpi, pad=pad,
        title_pt=base_pt + 1.0, axis_pt=base_pt, tick_pt=base_pt - 1.0, legend_pt=base_pt - 1.5,
    )


_PRESETS: dict[str, ExportSpec] = {
    "Thesis single-column": _preset(8.5, 6.0, 300, 8.0),
    "Thesis double-column": _preset(17.0, 9.0, 300, 9.0),
    "Square (map)": _preset(12.0, 12.0, 300, 9.0),
    "Presentation (16:9)": _preset(24.0, 13.5, 200, 13.0),
}


# --------------------------------------------------------------------------- #
# Config persistence                                                          #
# --------------------------------------------------------------------------- #
def seed_spec_from_config() -> None:
    """Seed the session :data:`SPEC` from the export-style sidecar at startup (tolerant of missing keys)."""
    global SPEC
    from common.config import load_export_spec

    SPEC = ExportSpec.from_dict(load_export_spec())


def _persist_spec(spec: ExportSpec) -> None:
    """Write ``spec`` back to ``export_spec.json`` so the user's tuned export survives a restart."""
    global SPEC
    SPEC = spec
    try:
        from common.config import save_export_spec
        save_export_spec(spec.to_dict())
    except Exception:
        pass  # persistence is best-effort; the session SPEC is already updated


# --------------------------------------------------------------------------- #
# A slider bound to a spin box (the unit control)                             #
# --------------------------------------------------------------------------- #
class _Slider(QWidget):
    """A horizontal slider + spin box editing one float value, kept in sync.

    Sliders are quick to drag; the spin box fine-tunes and shows the exact number. Both drive one
    ``on_change(value)`` callback. ``decimals``/``step`` shape the spin box; the slider works on an
    integer scale of ``1/step`` ticks across ``[lo, hi]``.
    """

    def __init__(self, lo: float, hi: float, value: float, *, decimals: int, step: float,
                 suffix: str, on_change) -> None:
        super().__init__()
        self._lo, self._hi, self._step = lo, hi, step
        self._on_change = on_change
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, int(round((hi - lo) / step)))
        self._box = QDoubleSpinBox()
        self._box.setRange(lo, hi)
        self._box.setDecimals(decimals)
        self._box.setSingleStep(step)
        self._box.setSuffix(suffix)
        self._set(value)
        self._slider.valueChanged.connect(self._from_slider)
        self._box.valueChanged.connect(self._from_box)
        row.addWidget(self._slider, 1)
        row.addWidget(self._box)

    def _set(self, value: float) -> None:
        for w in (self._slider, self._box):
            w.blockSignals(True)
        self._box.setValue(value)
        self._slider.setValue(int(round((value - self._lo) / self._step)))
        for w in (self._slider, self._box):
            w.blockSignals(False)

    def _from_slider(self, tick: int) -> None:
        self._set(self._lo + tick * self._step)
        self._on_change(self._box.value())

    def _from_box(self, value: float) -> None:
        self._set(value)
        self._on_change(value)

    def set_value(self, value: float) -> None:
        self._set(value)


# --------------------------------------------------------------------------- #
# The merged preview + settings dialog                                        #
# --------------------------------------------------------------------------- #
class ExportDialog(QDialog):
    """One window: a live preview of the export (left) and all style controls (right).

    Editing any control rebuilds the working :class:`ExportSpec` and re-renders the preview (throttled
    so dragging a slider stays smooth). Save / Copy use the working spec; closing persists it as the
    new session default. The *Map* group only appears for the field map.
    """

    def __init__(self, parent: QWidget, provider) -> None:
        super().__init__(parent)
        self._provider = provider
        self._spec = SPEC
        self._is_field = _is_field(provider)
        self.setWindowTitle(f"Export — {provider.export_title()}")
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.resize(1080, 680)

        outer = QVBoxLayout(self)
        body = QHBoxLayout()
        outer.addLayout(body, 1)

        # --- Left: preview ---
        # The preview always fits the available panel: the full-resolution render is kept in
        # _preview_image and only ever shown down-scaled to the viewport (never up-scaled past 1:1),
        # so a small window still shows the whole figure instead of a clipped corner. Re-scaling on
        # viewport resize is handled by the event filter below.
        left = QVBoxLayout()
        self._info = QLabel()
        self._info.setStyleSheet("color: gray;")
        left.addWidget(self._info)
        self._preview_scroll = QScrollArea()
        self._preview_scroll.setWidgetResizable(True)
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignCenter)
        self._preview_scroll.setWidget(self._image_label)
        self._preview_scroll.viewport().installEventFilter(self)
        self._preview_image: QImage | None = None
        left.addWidget(self._preview_scroll, 1)
        body.addLayout(left, 1)

        # --- Right: controls (scrollable so it never outgrows the window) ---
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFixedWidth(330)
        controls = QWidget()
        self._controls_layout = QVBoxLayout(controls)
        controls_scroll.setWidget(controls)
        body.addWidget(controls_scroll)

        # Probe what the graph actually contains (title? legend? axis labels?) so the panel only
        # shows controls for elements that exist — no Legend group on a legend-less graph, etc.
        self._features = self._detect_features()
        self._build_controls()

        # --- Buttons ---
        buttons = QDialogButtonBox()
        buttons.addButton("Save as…", QDialogButtonBox.AcceptRole).clicked.connect(
            lambda: _save(self, provider, self._spec))
        buttons.addButton("Copy", QDialogButtonBox.ActionRole).clicked.connect(
            lambda: _copy(provider, self._spec))
        buttons.addButton("Reset to default", QDialogButtonBox.ResetRole).clicked.connect(
            self._reset)
        buttons.addButton(QDialogButtonBox.Close).clicked.connect(self.close)
        outer.addWidget(buttons)

        # Throttle re-renders: coalesce a burst of slider ticks into one render.
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self._render)
        self._render()

    def export_title(self) -> str:  # so the nested _save can use this dialog as the parent
        return self._provider.export_title()

    # ---- spec editing ----
    def _edit(self, *, rebuild: bool = False, **changes) -> None:
        self._spec = replace(self._spec, **changes)
        if self._preset_box.currentText() != "Custom":
            self._preset_box.blockSignals(True)
            self._preset_box.setCurrentText("Custom")
            self._preset_box.blockSignals(False)
        if rebuild:
            # A toggle that shows/hides dependent rows (colorbar, title) rebuilds the panel, then
            # renders immediately so the preview and the new control set stay in lockstep.
            self._rebuild_controls()
            self._render()
        else:
            self._timer.start()

    def _reset(self) -> None:
        self._spec = ExportSpec()
        self._rebuild_controls()
        self._render()

    def _render(self) -> None:
        image = _provider_image(self._provider, self._spec)
        self._preview_image = image
        self._show_scaled()
        s = self._spec
        self._info.setText(
            f"{s.width_cm:g}×{s.height_cm:g} cm @ {s.dpi} DPI "
            f"({image.width()}×{image.height()} px) — preview matches the saved file."
        )
        _persist_spec(self._spec)

    def _show_scaled(self) -> None:
        """Show the current render scaled to fit the preview viewport (down only, aspect kept)."""
        from PySide6.QtGui import QPixmap

        image = self._preview_image
        if image is None or image.isNull():
            return
        # Fit inside the viewport, leaving a small margin; never enlarge past the true pixel size, so
        # a big figure in a small window shrinks to fit while a small figure stays crisp at 1:1.
        avail = self._preview_scroll.viewport().size()
        max_w = max(1, avail.width() - 4)
        max_h = max(1, avail.height() - 4)
        target_w = min(image.width(), max_w)
        target_h = min(image.height(), max_h)
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._image_label.setPixmap(scaled)
        self._image_label.adjustSize()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Resize and obj is self._preview_scroll.viewport():
            self._show_scaled()
        return super().eventFilter(obj, event)

    # ---- feature detection ----
    def _detect_features(self) -> set[str]:
        """Which adjustable elements this graph actually has, so absent ones get no controls.

        Renders the provider once at the working spec and inspects the axes: a graph with no title /
        no legend / no axis labels shouldn't show those controls. The field map is a fixed feature
        set (it has its own chrome). Best-effort — on any probe failure we show everything rather than
        hide a real control.
        """
        if self._is_field:
            return {"map"}
        feats = {"axes", "elements"}  # ticks/axis-scale always apply to a matplotlib graph
        try:
            figure = render_matplotlib(self._provider.draw_into, self._spec)
            ax = figure.axes[0] if figure.axes else None
            if ax is not None:
                if (ax.get_title() or "").strip():
                    feats.add("title")
                if ax.xaxis.get_label().get_text().strip() or ax.yaxis.get_label().get_text().strip():
                    feats.add("axis_labels")
                if ax.get_legend() is not None or ax.get_legend_handles_labels()[0]:
                    feats.add("legend")
        except Exception:
            feats |= {"title", "axis_labels", "legend"}
        return feats

    # ---- control building ----
    def _build_controls(self) -> None:
        s = self._spec
        self._sliders: dict[str, _Slider] = {}

        # Preset picker.
        self._preset_box = QComboBox()
        self._preset_box.addItem("Custom")
        self._preset_box.addItems(list(_PRESETS))
        self._preset_box.currentTextChanged.connect(self._apply_preset)
        preset_box = QGroupBox("Preset")
        pl = QVBoxLayout(preset_box)
        pl.addWidget(self._preset_box)
        self._controls_layout.addWidget(preset_box)

        # Output size — one compact group, label+slider per row.
        g = self._form("Output")
        self._row(g, "width_cm", "Width", 2.0, 40.0, s.width_cm, decimals=1, step=0.5, suffix=" cm")
        self._row(g, "height_cm", "Height", 2.0, 40.0, s.height_cm, decimals=1, step=0.5, suffix=" cm")
        self._row(g, "dpi", "Resolution", 72, 1200, s.dpi, decimals=0, step=1, suffix=" DPI")

        if self._is_field:
            self._build_map_controls(s)
            return

        f = self._features
        # Title — font lives with the (only) title, not a generic "Fonts" bucket.
        if "title" in f:
            g = self._form("Title")
            self._row(g, "title_pt", "Font", 4.0, 32.0, s.title_pt, decimals=1, step=0.5, suffix=" pt")

        # Axes — axis-label font (only if labelled), tick font, and the edge padding all belong here.
        g = self._form("Axes")
        if "axis_labels" in f:
            self._row(g, "axis_pt", "Label font", 4.0, 32.0, s.axis_pt, decimals=1, step=0.5, suffix=" pt")
        self._row(g, "tick_pt", "Tick font", 4.0, 32.0, s.tick_pt, decimals=1, step=0.5, suffix=" pt")
        self._row(g, "pad", "Edge padding", 0.0, 0.5, s.pad, decimals=2, step=0.01, suffix="")

        # Legend — position and its font clustered together (the user's example), only if there is one.
        if "legend" in f:
            g = self._form("Legend")
            self._legend_box = QComboBox()
            self._legend_box.addItems(list(_LEGEND_PLACEMENTS))
            self._legend_box.setCurrentText(s.legend_loc)
            self._legend_box.currentTextChanged.connect(lambda t: self._edit(legend_loc=t))
            g.addRow("Position", self._legend_box)
            self._row(g, "legend_pt", "Font", 4.0, 32.0, s.legend_pt, decimals=1, step=0.5, suffix=" pt")

        # Data elements.
        g = self._form("Data")
        self._row(g, "element_scale", "Marker / line size", 0.3, 3.0, s.element_scale,
                  decimals=2, step=0.05, suffix="×")

        self._build_label_controls(s)

    def _build_label_controls(self, s: ExportSpec) -> None:
        """Per in-graph-box controls (show / size / corner), if the provider declares any boxes.

        Driven by the provider's optional ``export_labels()`` -> [(id, name), …]; each box gets its
        own toggle, scale slider and corner picker, all writing into ``spec.label_overrides[id]``.
        Graphs without boxes (no ``export_labels``) get no group, so the panel stays clean.
        """
        labels = []
        getter = getattr(self._provider, "export_labels", None)
        if callable(getter):
            try:
                labels = list(getter())
            except Exception:
                labels = []
        if not labels:
            return

        # One "Boxes" group: a shared baseline font, then one compact row-set per box (show / size /
        # position together) so every box is tuned in place without a separate group each.
        g = self._form("Boxes")
        self._row(g, "label_pt", "Font", 4.0, 32.0, s.label_pt, decimals=1, step=0.5, suffix=" pt")
        for label_id, name in labels:
            ov = s.label_overrides.get(label_id) if isinstance(s.label_overrides.get(label_id), dict) else {}
            chk = QCheckBox(name)
            chk.setChecked(ov.get("visible", True))
            chk.toggled.connect(lambda v, i=label_id: self._edit_label(i, visible=v))
            g.addRow(chk)
            slider = _Slider(0.3, 3.0, float(ov.get("scale", 1.0)), decimals=2, step=0.05, suffix="×",
                             on_change=lambda v, i=label_id: self._edit_label(i, scale=v))
            self._sliders[f"label::{label_id}::scale"] = slider
            corner = QComboBox()
            corner.addItems(list(_LABEL_CORNERS))
            corner.setCurrentText(ov.get("corner", "top-left"))
            corner.currentTextChanged.connect(lambda t, i=label_id: self._edit_label(i, corner=t))
            g.addRow("    size", slider)
            g.addRow("    position", corner)

    def _edit_label(self, label_id: str, **changes) -> None:
        """Merge ``changes`` into ``spec.label_overrides[label_id]`` and re-render (copy, not mutate).

        The spec is frozen, so we build a fresh overrides dict each time rather than editing in place —
        keeps every working spec immutable and lets Reset/preset swaps replace it cleanly.
        """
        overrides = {k: dict(v) for k, v in self._spec.label_overrides.items()}
        overrides.setdefault(label_id, {}).update(changes)
        self._edit(label_overrides=overrides)

    def _build_map_controls(self, s: ExportSpec) -> None:
        # Framing — zoom (like scrolling the live canvas), rotation about the centre, and the breathing
        # room around the field. Offsets are kept for fine recentring but folded in here.
        g = self._form("Framing")
        self._row(g, "map_zoom", "Zoom", 0.3, 4.0, s.map_zoom, decimals=2, step=0.05, suffix="×")
        self._row(g, "map_rotation_deg", "Rotation", 0.0, 360.0, s.map_rotation_deg, decimals=0, step=1, suffix="°")
        self._row(g, "map_margin_frac", "Margin", 0.0, 0.5, s.map_margin_frac, decimals=2, step=0.01, suffix="")
        self._row(g, "map_offset_x", "X offset", -0.5, 0.5, s.map_offset_x, decimals=2, step=0.01, suffix="")
        self._row(g, "map_offset_y", "Y offset", -0.5, 0.5, s.map_offset_y, decimals=2, step=0.01, suffix="")
        self._row(g, "plot_label_scale", "Plot labels", 0.3, 3.0, s.plot_label_scale, decimals=2, step=0.05, suffix="×")

        # Colorbar — toggle, and (only when shown) its position and fonts, clustered together.
        g = self._form("Colorbar")
        self._cbar_chk = QCheckBox("Show")
        self._cbar_chk.setChecked(s.show_colorbar)
        # Rebuild so the dependent position/font rows appear or vanish with the toggle.
        self._cbar_chk.toggled.connect(lambda v: self._edit(show_colorbar=v, rebuild=True))
        g.addRow(self._cbar_chk)
        if s.show_colorbar:
            self._cbar_pos = QComboBox()
            self._cbar_pos.addItems(_COLORBAR_POSITIONS)
            self._cbar_pos.setCurrentText(s.colorbar_pos)
            self._cbar_pos.currentTextChanged.connect(lambda t: self._edit(colorbar_pos=t))
            g.addRow("Position", self._cbar_pos)
            self._row(g, "colorbar_width_frac", "Width", 0.08, 0.4, s.colorbar_width_frac, decimals=2, step=0.01, suffix="")
            self._row(g, "colorbar_tick_pt", "Tick font", 4.0, 24.0, s.colorbar_tick_pt, decimals=1, step=0.5, suffix=" pt")
            self._row(g, "colorbar_label_pt", "Label font", 4.0, 24.0, s.colorbar_label_pt, decimals=1, step=0.5, suffix=" pt")

        # Title — toggle plus (only when shown) its font.
        g = self._form("Title")
        self._title_chk = QCheckBox("Show")
        self._title_chk.setChecked(s.show_title)
        self._title_chk.toggled.connect(lambda v: self._edit(show_title=v, rebuild=True))
        g.addRow(self._title_chk)
        if s.show_title:
            self._row(g, "title_overlay_pt", "Font", 6.0, 32.0, s.title_overlay_pt, decimals=1, step=0.5, suffix=" pt")

        # Background.
        g = self._form("Background")
        self._bg_box = QComboBox()
        self._bg_box.addItems(_BACKGROUNDS)
        self._bg_box.setCurrentText(s.background)
        self._bg_box.currentTextChanged.connect(lambda t: self._edit(background=t))
        g.addRow("Fill", self._bg_box)

    def _form(self, title: str) -> QFormLayout:
        """A compact group whose rows are ``label : control`` on one line (saves vertical space)."""
        box = QGroupBox(title)
        layout = QFormLayout(box)
        layout.setLabelAlignment(Qt.AlignRight)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setVerticalSpacing(4)
        self._controls_layout.addWidget(box)
        return layout

    def _row(self, form: QFormLayout, field_name: str, label: str, lo: float, hi: float,
             value: float, *, decimals: int, step: float, suffix: str) -> None:
        """One ``label : slider`` row in a :meth:`_form` group, bound to spec field ``field_name``."""
        as_int = decimals == 0

        def on_change(v):
            self._edit(**{field_name: (int(round(v)) if as_int else v)})

        slider = _Slider(lo, hi, value, decimals=decimals, step=step, suffix=suffix, on_change=on_change)
        self._sliders[field_name] = slider
        form.addRow(label, slider)

    def _apply_preset(self, name: str) -> None:
        if name not in _PRESETS:
            return
        preset = _PRESETS[name]
        # Keep the user's map-only tweaks; presets only carry geometry + font hierarchy.
        self._spec = replace(
            self._spec, width_cm=preset.width_cm, height_cm=preset.height_cm, dpi=preset.dpi,
            title_pt=preset.title_pt, axis_pt=preset.axis_pt, tick_pt=preset.tick_pt,
            legend_pt=preset.legend_pt,
        )
        self._sync_sliders()
        self._render()

    def _sync_sliders(self) -> None:
        # Only the sliders bound 1:1 to a spec field re-sync here (presets carry no label tweaks);
        # the per-label sliders use composite keys (no matching attribute) and are skipped.
        s = self._spec
        for name, slider in self._sliders.items():
            if hasattr(s, name):
                slider.set_value(float(getattr(s, name)))

    def _rebuild_controls(self) -> None:
        """Tear down and rebuild the controls panel (Reset, or a toggle that changes which rows show)."""
        while self._controls_layout.count():
            item = self._controls_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                # Detach now (not just deleteLater) so the rebuilt groups don't briefly stack on top
                # of the outgoing ones before the event loop processes the deletion.
                w.setParent(None)
                w.deleteLater()
        self._build_controls()

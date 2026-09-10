"""A GPU-friendly QGraphicsView map canvas for the results field map.

This replaces the matplotlib canvas the map view used to draw on. The field map is fundamentally
a 2-D scene of ~80 coloured plot footprints plus per-plot labels that the user pans, zooms and
rotates — exactly what a :class:`QGraphicsScene` / :class:`QGraphicsView` does in hardware: pan,
zoom and rotate are *view transforms*, so the scene items are never re-rasterised and the view
stays at 60 fps+ no matter how many plots there are (matplotlib re-rendered the whole figure on
every drag, which capped it at ~10 fps).

Two layouts share one scene:

* **Field** — each plot as its real oriented-bounding-box footprint at its true world position,
  with a single value drawn *inside* it, sized to fill the box just short of the borders and
  rotated to match the plot's own orientation. The labels are vector text scaled by the view, so
  they stay crisp and readable at every zoom instead of clumping at full-field zoom.
* **Grid** — uniform unit squares laid out from the chosen layout spec, each with up to three
  stacked value lines (the tidy, schematic view).

The host (:class:`gui.results.map_view.MapView`) owns the controls and the data; this widget only
draws what it is handed via :meth:`set_field` / :meth:`set_grid` and reports clicks back through a
signal. It keeps no model state beyond what it needs to hit-test and re-colour.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPen,
    QPolygonF,
    QTransform,
)
from PySide6.QtCore import QSize
from PySide6.QtGui import QLinearGradient
from PySide6.QtWidgets import (
    QGraphicsPolygonItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QLabel,
    QPushButton,
    QWidget,
)


# Plot items carry their plot number on this Qt item-data key, so a click can read it back.
_PLOT_KEY = 0
# Scene Y grows downward in Qt; world/data Y grows upward. We flip Y when placing items so the
# field reads the same way it did under matplotlib (north up) and a positive view rotation turns
# the field anticlockwise on screen, which is the intuitive direction.
_BG = "#ffffff"


@dataclass
class PlotPatch:
    """One plot to draw: its footprint polygon (world XY) and the colour to fill it with.

    ``polygon`` is an ``(n, 2)`` array of world coordinates (4 corners for a field footprint, the
    unit square's 4 corners for the grid). ``angle_deg`` is the orientation the field label should
    take (the footprint's long-axis angle); the grid passes 0. ``fill`` is any Qt-acceptable colour.
    """

    plot: int
    polygon: np.ndarray
    fill: QColor
    angle_deg: float = 0.0
    held_out: bool = False
    selected: bool = False
    border_width: float = 2.4  # held-out border pen width (px); only used when ``held_out``


@dataclass
class FieldExport:
    """Everything :mod:`gui.export` needs to render the field map as a thesis figure.

    The colorbar (``stops``/``lo``/``hi``/``colorbar_label``) and ``title`` are passed alongside the
    scene because they live as viewport overlays on screen — the scene render cannot see them — so
    the export bakes them back onto the image itself.
    """

    scene: QGraphicsScene
    source_rect: QRectF
    y_flip: bool
    rotation_deg: float
    stops: list
    lo: float
    hi: float
    colorbar_label: str
    title: str


class FieldCanvas(QGraphicsView):
    """Pan/zoom/rotate scene-graph canvas for the results map. Emits clicks as plot numbers."""

    # plot number, Qt mouse button (Qt.LeftButton / Qt.RightButton), was-double-click
    plot_clicked = Signal(int, object, bool)

    def __init__(self) -> None:
        self._scene = QGraphicsScene()
        super().__init__(self._scene)
        self.setBackgroundBrush(QBrush(QColor(_BG)))
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.TextAntialiasing, True)
        # Zoom anchored under the cursor; drag pans. Selecting is toggled by the host via a flag.
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        # Panning is implemented as scrolling, so the scroll bars must be *available* (the drag has
        # nothing to move otherwise — that was why zooming dumped the field into a corner with no way
        # to pan back). We just hide them so the canvas stays clean.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # The scene uses a Y-down coordinate system internally but we feed it Y-up world data and
        # flip once here, so north stays up and rotation reads naturally.
        self.scale(1.0, -1.0)
        self.setMouseTracking(True)
        # The viewport deliberately stays on Qt's raster engine. An OpenGL viewport gave the ~80
        # polygons no measurable headroom but distorted them on zoom (cosmetic-pen footprints
        # wobbled / changed shape as the view scaled), so software rasterising is the correct
        # trade-off here: it is already comfortably 60 fps+ and renders the footprints exactly.

        self._rotation_deg = 0.0          # extra user rotation of the whole view
        self._title = ""
        self._content_rect: QRectF | None = None  # plot-geometry bounds, for pen-independent framing
        # Last set_scene inputs, kept so the export can rebuild a throwaway scene (scaled labels).
        self._patches: list[PlotPatch] = []
        self._field_mode = True
        self._label_for = lambda _plot: []
        self._build_overlays()

    # ------------------------------------------------------------------ #
    # Overlay chrome (title / colour legend / home), layered over the view #
    # ------------------------------------------------------------------ #
    def _build_overlays(self) -> None:
        """Title, colour legend and a Home button, parented to the viewport so they float *over* the
        rendered scene (fixed screen-space UI) instead of scrolling/rotating with it."""
        vp = self.viewport()

        self._title_label = QLabel(vp)
        self._title_label.setStyleSheet(
            "color: #222; background: transparent;"
        )
        f = QFont()
        f.setPointSizeF(15)
        f.setBold(True)
        self._title_label.setFont(f)
        self._title_label.setAlignment(Qt.AlignCenter)
        self._title_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        # The colour legend now floats over the canvas (top-right), so its gradient and percentage
        # ticks sit on the white field rather than off to the side in the window chrome.
        self.colorbar = ColorBar(vp)

        self._home_button = QPushButton("Home view", vp)
        self._home_button.setToolTip("Reframe the whole field (use if the view gets lost).")
        self._home_button.setCursor(Qt.PointingHandCursor)
        self._home_button.setStyleSheet(
            "QPushButton { background: rgba(255,255,255,0.85); border: 1px solid #999;"
            " border-radius: 4px; padding: 3px 10px; color: #222; }"
            "QPushButton:hover { background: rgba(235,235,235,0.95); }"
        )
        self._home_button.clicked.connect(self.reset_view)
        self._home_button.adjustSize()

        self._layout_overlays()

    def _layout_overlays(self) -> None:
        """Position the floating chrome within the viewport (called on every resize)."""
        vp = self.viewport()
        w, h = vp.width(), vp.height()
        self._title_label.setGeometry(0, 6, w, 30)
        bar_w = self.colorbar.width()
        self.colorbar.setGeometry(w - bar_w - 6, 40, bar_w, max(120, h - 70))
        self._home_button.move(8, h - self._home_button.height() - 8)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._layout_overlays()

    def scrollContentsBy(self, dx: int, dy: int) -> None:  # noqa: N802
        """Keep the floating chrome put when the user pans.

        Panning scrolls the viewport, which drags the viewport's child widgets (our overlays) along
        with it — so the title and legend slid with the field and only snapped back on the next full
        repaint. Repositioning them after every scroll pins them to fixed screen space.
        """
        super().scrollContentsBy(dx, dy)
        self._layout_overlays()

    # ------------------------------------------------------------------ #
    # Scene building                                                     #
    # ------------------------------------------------------------------ #
    def set_scene(self, patches: list[PlotPatch], *, field_mode: bool, title: str,
                  label_for, label_font_pt: float | None = None) -> None:
        """Rebuild the whole scene from ``patches``.

        ``label_for(plot)`` returns the lines to draw inside a plot, each a ``(text, font_pt, bold)``
        triple where ``font_pt == 0`` means "auto-size" (the view picks one shared size for every
        such line). Both views stack up to three lines; ``field_mode`` only changes whether the
        block is rotated to the plot's own orientation (field) or kept axis-aligned (grid), and what
        the world cell size is (a real footprint vs. the unit square).
        """
        self._scene.clear()
        self._title = title
        self._title_label.setText(title)
        # Frame the view from the plot *geometry* only (computed here), never from the scene's item
        # bounding rect — a held-out plot's thick border pen would otherwise inflate that rect and
        # make the whole field zoom out a little whenever a border was shown.
        self._content_rect = _polygons_bounds([p.polygon for p in patches])
        # Remember the inputs so the export can rebuild a throwaway scene (e.g. with scaled labels)
        # without disturbing this live one.
        self._patches = patches
        self._field_mode = field_mode
        self._label_for = label_for
        self._populate(self._scene, patches, field_mode, label_for, label_scale=1.0)
        self._fit()

    def _populate(self, scene: QGraphicsScene, patches: list[PlotPatch], field_mode: bool,
                  label_for, *, label_scale: float) -> None:
        """Fill ``scene`` with the patches and their stacked labels (shared by live + export)."""
        # Auto-sized lines share ONE world-scale across every plot, so the numbers read at a
        # consistent size regardless of how big each plot's footprint is. Sized to the narrowest
        # plot so even there nothing overflows. (Manual per-line point sizes bypass this.)
        auto_scale = self._auto_label_scale(patches, field_mode) * label_scale
        for p in patches:
            self._add_patch(scene, p)
            self._add_lines(scene, p, label_for(p.plot), auto_scale, rotate=field_mode,
                            label_scale=label_scale)

    def _auto_label_scale(self, patches: list[PlotPatch], field_mode: bool) -> float:
        """One shared world-scale for auto-sized lines so they all render at the same font size.

        A ~13 px-tall authored glyph block scaled by this factor sets its world height. Based on the
        *smallest* plot's short side (grid cells are all 1 unit) so the shared size fits everywhere.
        """
        shorts = []
        for p in patches:
            short, long_ = _box_sides(p.polygon)
            if short > 0 and long_ > 0:
                shorts.append(short)
        if not shorts:
            return 1.0 / 13.0
        ref_h = 13.0
        # Auto lines fill ~26% of the narrowest plot's short side each, leaving room to stack three.
        return (min(shorts) * 0.26) / ref_h

    def _add_patch(self, scene: QGraphicsScene, p: PlotPatch) -> None:
        poly = QPolygonF([QPointF(float(x), float(y)) for x, y in p.polygon])
        item = QGraphicsPolygonItem(poly)
        item.setBrush(QBrush(p.fill))
        # A hairline white seam between plots; cosmetic pen keeps it 1px at every zoom.
        pen = QPen(QColor("white"))
        pen.setCosmetic(True)
        pen.setWidthF(1.0)
        item.setPen(pen)
        item.setData(_PLOT_KEY, int(p.plot))
        scene.addItem(item)

        if p.held_out:  # thick black border on held-out plots, drawn on top
            border = QGraphicsPolygonItem(poly)
            border.setBrush(QBrush(Qt.NoBrush))
            bpen = QPen(QColor("black"))
            bpen.setCosmetic(True)
            bpen.setWidthF(float(p.border_width))
            border.setPen(bpen)
            border.setZValue(5)
            border.setData(_PLOT_KEY, int(p.plot))
            scene.addItem(border)
        if p.selected:  # a black corner triangle marking a plot picked for the 3-D view
            self._add_selection_marker(scene, p)

    def _add_selection_marker(self, scene: QGraphicsScene, p: PlotPatch) -> None:
        xy = p.polygon
        # Top-left corner of the footprint's world bbox, with a small triangle inside it.
        hx0, hy1 = float(xy[:, 0].min()), float(xy[:, 1].max())
        w = (float(xy[:, 0].max()) - hx0) * 0.3
        tri = QGraphicsPolygonItem(QPolygonF([
            QPointF(hx0, hy1), QPointF(hx0 + w, hy1), QPointF(hx0, hy1 - w),
        ]))
        tri.setBrush(QBrush(QColor("black")))
        tri.setPen(QPen(Qt.NoPen))
        tri.setZValue(6)
        scene.addItem(tri)

    def _add_lines(self, scene: QGraphicsScene, p: PlotPatch, lines, auto_scale: float, *,
                   rotate: bool, label_scale: float = 1.0) -> None:
        """Stack up to three ``(text, font_pt, bold)`` lines centred in the plot.

        ``font_pt == 0`` auto-sizes the line to the scene-wide ``auto_scale`` (uniform size across
        plots); any other value is a manual point size mapped to a cell-relative target height. Each
        line is clamped down if it would overrun *this* plot's footprint, so nothing crosses a
        border. When ``rotate`` the whole block is turned to the plot's long-axis angle (field map);
        otherwise it stays axis-aligned (grid). The vertical offsets shrink with the short side so a
        narrow footprint still stacks its lines inside itself.
        """
        lines = [ln for ln in (lines or []) if ln and ln[0]]
        if not lines:
            return
        short, long_ = _box_sides(p.polygon)
        if short <= 0 or long_ <= 0:
            return
        cx, cy = float(p.polygon[:, 0].mean()), float(p.polygon[:, 1].mean())
        # Stack along the LONG axis (footprints are long thin strips); spacing is a fraction of it.
        step = long_ * 0.30
        offsets = {1: [0.0], 2: [0.5 * step, -0.5 * step],
                   3: [step, 0.0, -step]}.get(len(lines), [0.0])
        for (text, pt, bold), dly in zip(lines, offsets):
            item = _make_text(text, QColor(_text_colour(p.fill)), bold=bold)
            br = item.boundingRect()
            if br.width() <= 0 or br.height() <= 0:
                continue
            # Auto (pt==0) uses the shared scale (already includes label_scale); a manual pt maps to
            # a fraction of the short side, scaled by the same factor for the export.
            if pt <= 0:
                s = auto_scale
            else:
                target_h = max(0.06, min(0.5, pt / 45.0)) * short * label_scale
                s = target_h / br.height()
            # Clamp so this line never spills past the footprint's borders.
            if br.width() * s > long_ * 0.92:
                s = (long_ * 0.92) / br.width()
            if br.height() * s > short * 0.92:
                s = min(s, (short * 0.92) / br.height())
            item.setAcceptedMouseButtons(Qt.NoButton)  # click-through to the fill below
            # Place the line offset along the (possibly rotated) long axis of the plot.
            ang = math.radians(p.angle_deg) if rotate else 0.0
            ox, oy = (-math.sin(ang) * dly, math.cos(ang) * dly)
            self._place_rotated(scene, item, cx + ox, cy + oy, s, p.angle_deg if rotate else 0.0)

    def _place_rotated(self, scene: QGraphicsScene, item: QGraphicsSimpleTextItem, cx: float,
                       cy: float, scale: float, angle_deg: float) -> None:
        """Centre ``item`` on world ``(cx, cy)`` at ``scale``, rotated ``angle_deg``, Y-flipped.

        Text items are authored in a Y-down frame; the view's overall Y-flip would mirror them, so
        we flip the item's own Y back (scale -y) and rotate it into the plot's orientation.
        """
        br = item.boundingRect()
        t = QTransform()
        t.translate(cx, cy)
        t.rotate(angle_deg)
        t.scale(scale, -scale)            # undo the view's Y-flip so glyphs read upright
        t.translate(-br.width() / 2.0, -br.height() / 2.0)
        item.setTransform(t)
        item.setZValue(3)
        scene.addItem(item)

    # ------------------------------------------------------------------ #
    # View transform: fit / zoom / pan / rotate                          #
    # ------------------------------------------------------------------ #
    def _fit(self) -> None:
        """Frame the whole scene with a small margin, preserving the current user rotation."""
        content = getattr(self, "_content_rect", None) or self._scene.itemsBoundingRect()
        if content.isEmpty():
            return
        margin = 0.06 * max(content.width(), content.height())
        framed = content.adjusted(-margin, -margin, margin, margin)
        # Give the scene rect a generous border of empty space around the content so that, once the
        # user zooms in, there is somewhere to pan to. Without this the scrollable area collapses to
        # the content and ScrollHandDrag has nothing to move, so a zoom strands the field in a corner.
        pad = 1.5 * max(framed.width(), framed.height())
        self._scene.setSceneRect(framed.adjusted(-pad, -pad, pad, pad))
        self.resetTransform()
        self.scale(1.0, -1.0)
        if self._rotation_deg:
            self.rotate(self._rotation_deg)
        self.fitInView(framed, Qt.KeepAspectRatio)
        self.viewport().update()

    def reset_view(self) -> None:
        self._fit()

    # ------------------------------------------------------------------ #
    # Export (thesis figures)                                            #
    # ------------------------------------------------------------------ #
    def build_export(self, spec=None):
        """A :class:`FieldExport` describing how to render this map at an :class:`gui.export.ExportSpec`.

        Carries everything the export layer needs to composite a faithful, print-resolution figure:
        the scene to render (the live one, or a throwaway with scaled in-plot labels), the framed
        source rect (margin + recentre offsets from the spec), the Y-flip + whole-field rotation,
        and the colour-legend payload + title so the export can bake the colorbar back in (the live
        colorbar is a viewport overlay the scene render cannot see). ``spec`` may be None for the
        plain defaults (used by the legacy 6% framing).
        """
        from ..export import SPEC as _DEFAULT_SPEC

        spec = spec or _DEFAULT_SPEC
        # Use the live scene when labels are unscaled; otherwise rebuild a throwaway scaled scene so
        # the live view is never disturbed.
        if abs(spec.plot_label_scale - 1.0) < 1e-6 or not self._patches:
            scene = self._scene
        else:
            scene = QGraphicsScene()
            self._populate(scene, self._patches, self._field_mode, self._label_for,
                           label_scale=spec.plot_label_scale)

        content = self._content_rect or self._scene.itemsBoundingRect()
        # A SQUARE source rect centred on the content's centre. Squaring the window is what makes
        # rotation behave: the export rotates the painter about the rendered field's centre, so the
        # field's own centre must sit at the centre of the source window — otherwise a wide field
        # swings off-frame as it turns. The half-extent covers the larger content dimension (so the
        # whole field fits at any rotation), grown by the margin and shrunk by the zoom.
        cx = content.center().x() + spec.map_offset_x * content.width()
        cy = content.center().y() + spec.map_offset_y * content.height()
        half = 0.5 * max(content.width(), content.height(), 1.0)
        half *= (1.0 + 2.0 * spec.map_margin_frac)
        half /= max(spec.map_zoom, 1e-3)
        source = QRectF(cx - half, cy - half, 2 * half, 2 * half)

        rotation = spec.map_rotation_deg if spec.map_rotation_deg else self._rotation_deg
        stops, lo, hi, label = self.colorbar.scale_data()
        return FieldExport(
            scene=scene, source_rect=source, y_flip=True, rotation_deg=rotation,
            stops=stops, lo=lo, hi=hi, colorbar_label=label, title=self._title,
        )

    def export_title(self) -> str:
        return "field_map"

    def set_rotation(self, degrees: float) -> None:
        """Rotate the entire field view to ``degrees`` (absolute), keeping it framed."""
        self._rotation_deg = float(degrees) % 360.0
        self._fit()

    def wheelEvent(self, event) -> None:
        """Mouse-wheel zooms about the cursor (a pure view transform — no re-render of items)."""
        factor = 1.0015 ** event.angleDelta().y()
        self.scale(factor, factor)

    # ------------------------------------------------------------------ #
    # Clicks                                                             #
    # ------------------------------------------------------------------ #
    def mouseDoubleClickEvent(self, event) -> None:
        plot = self._plot_at(event.position().toPoint())
        if plot is not None:
            self.plot_clicked.emit(plot, event.button(), True)
            return
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event) -> None:
        # Right-click never pans (it opens the context menu); report it and swallow the drag.
        if event.button() == Qt.RightButton:
            plot = self._plot_at(event.position().toPoint())
            if plot is not None:
                self.plot_clicked.emit(plot, event.button(), False)
            return
        super().mousePressEvent(event)
        self._press_pos = event.position().toPoint()

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        if event.button() != Qt.LeftButton:
            return
        # A click (not a drag) selects/inspects the plot; a drag is a pan and is ignored here.
        start = getattr(self, "_press_pos", None)
        if start is not None and (event.position().toPoint() - start).manhattanLength() <= 3:
            plot = self._plot_at(event.position().toPoint())
            if plot is not None:
                self.plot_clicked.emit(plot, Qt.LeftButton, False)

    def _plot_at(self, view_pos) -> int | None:
        """Plot number under a viewport point, or ``None``.

        Adjacent plots' oriented footprints can overlap slightly, so several fill items may contain
        the same point. We pick the plot whose centre is nearest the click — the one the user
        plainly meant — rather than whichever happens to sit on top of the z-order.
        """
        scene_pos = self.mapToScene(view_pos)
        best_plot, best_d2 = None, None
        for item in self._scene.items(scene_pos):
            data = item.data(_PLOT_KEY)
            if data is None:
                continue
            c = item.boundingRect().center()
            d2 = (c.x() - scene_pos.x()) ** 2 + (c.y() - scene_pos.y()) ** 2
            if best_d2 is None or d2 < best_d2:
                best_plot, best_d2 = int(data), d2
        return best_plot


# --------------------------------------------------------------------------- #
# Small helpers                                                               #
# --------------------------------------------------------------------------- #
def _make_text(text: str, colour: QColor, bold: bool = False) -> QGraphicsSimpleTextItem:
    item = QGraphicsSimpleTextItem(text)
    f = QFont()
    f.setPointSizeF(10.0)  # authored size; the item transform scales it to the footprint
    f.setBold(bold)
    item.setFont(f)
    item.setBrush(QBrush(colour))
    return item


def _polygons_bounds(polygons: list[np.ndarray]) -> QRectF | None:
    """The world bounding rect spanning every polygon's corners (pen-independent), or None."""
    pts = [p for p in polygons if p is not None and len(p)]
    if not pts:
        return None
    allxy = np.vstack(pts)
    x0, y0 = float(allxy[:, 0].min()), float(allxy[:, 1].min())
    x1, y1 = float(allxy[:, 0].max()), float(allxy[:, 1].max())
    return QRectF(x0, y0, x1 - x0, y1 - y0)


def _box_sides(polygon: np.ndarray) -> tuple[float, float]:
    """The (short, long) edge lengths of a 4-corner box polygon, in world units."""
    if len(polygon) < 4:
        w = float(polygon[:, 0].max() - polygon[:, 0].min())
        h = float(polygon[:, 1].max() - polygon[:, 1].min())
        return (min(w, h), max(w, h))
    e1 = float(np.hypot(*(polygon[1] - polygon[0])))
    e2 = float(np.hypot(*(polygon[2] - polygon[1])))
    return (min(e1, e2), max(e1, e2))


def box_angle_deg(polygon: np.ndarray) -> float:
    """Orientation of a 4-corner box's long axis, in degrees (for rotating the field label).

    Returns an angle in (-90, 90]; text is always upright-ish (we never flip it past vertical).
    """
    if len(polygon) < 4:
        return 0.0
    e1 = polygon[1] - polygon[0]
    e2 = polygon[2] - polygon[1]
    long_edge = e1 if np.hypot(*e1) >= np.hypot(*e2) else e2
    angle = math.degrees(math.atan2(float(long_edge[1]), float(long_edge[0])))
    # Keep the text from reading upside-down: fold the angle into (-90, 90].
    while angle > 90:
        angle -= 180
    while angle <= -90:
        angle += 180
    return angle


def _text_colour(fill: QColor) -> str:
    """Black or white label, by the WCAG luminance of the fill (mirrors the old map heuristic)."""
    r, g, b = fill.redF(), fill.greenF(), fill.blueF()
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luminance > 0.5 else "white"


class ColorBar(QWidget):
    """A vertical colour-scale legend that floats *over* the map (replaces matplotlib's colorbar).

    Painted directly, so it costs nothing and never participates in the laggy redraw path. It is
    parented to the canvas viewport and positioned over the rendered field (top-right), with a
    translucent backing panel so the gradient and its numeric ticks read clearly on the white field.
    The host calls :meth:`set_scale` with the colour stops (low->high) and the data range + label.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Wide enough for the colour bar, the numeric tick labels beside it, and the rotated axis
        # label on the right without any of them clipping or overlapping.
        self.setFixedWidth(132)
        # Floats over the scene, so its own background must not block the field behind the margins.
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._stops: list[tuple[float, QColor]] = [(0.0, QColor("#1a9850")), (1.0, QColor("#d73027"))]
        self._lo, self._hi = 0.0, 1.0
        self._label = ""

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(132, 200)

    def set_scale(self, stops: list[tuple[float, QColor]], lo: float, hi: float, label: str) -> None:
        self._stops = stops or self._stops
        self._lo, self._hi, self._label = float(lo), float(hi), label
        self.update()

    def scale_data(self) -> tuple[list[tuple[float, QColor]], float, float, str]:
        """The current colour stops + data range + axis label, for the export to redraw the bar."""
        return list(self._stops), self._lo, self._hi, self._label

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        draw_colorbar(painter, QRectF(0, 0, self.width(), self.height()),
                      self._stops, self._lo, self._hi, self._label)


def draw_colorbar(painter: QPainter, rect: QRectF, stops: list[tuple[float, QColor]],
                  lo: float, hi: float, label: str, *, tick_pt: float = 10.0,
                  label_pt: float = 11.0) -> None:
    """Paint the vertical colour-scale legend (panel, gradient, ticks, rotated axis label) in ``rect``.

    Shared by the on-screen :class:`ColorBar` overlay and the figure export, so both render the bar
    identically — the export just supplies a print-resolution ``rect`` and larger fonts. All
    geometry scales with ``rect`` so the bar fills whatever panel it is given.
    """
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.translate(rect.topLeft())
    w, h = rect.width(), rect.height()
    # Opaque backing panel so the legend stays crisp and fully saturated over the field.
    painter.setPen(QColor("#bbb"))
    painter.setBrush(QBrush(QColor(255, 255, 255)))
    painter.drawRoundedRect(QRectF(0.5, 0.5, w - 1, h - 1), 6, 6)

    margin = max(10.0, h * 0.05)
    bar_x, bar_w, top, bot = 10.0, max(14.0, w * 0.18), margin, h - margin
    grad = QLinearGradient(0, bot, 0, top)  # low at the bottom, high at the top
    for pos, col in stops:
        grad.setColorAt(max(0.0, min(1.0, pos)), col)
    painter.fillRect(QRectF(bar_x, top, bar_w, bot - top), QBrush(grad))
    # Outline only — the brush is still the white panel brush, so without NoBrush this would fill
    # the bar solid white and erase the gradient.
    painter.setPen(QColor("#888"))
    painter.setBrush(Qt.NoBrush)
    painter.drawRect(QRectF(bar_x, top, bar_w, bot - top))
    # Tick labels along the range, in the gutter between the bar and the rotated axis label.
    painter.setPen(QColor("#222"))
    f = QFont()
    f.setPointSizeF(tick_pt)
    painter.setFont(f)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = bot - frac * (bot - top)
        val = lo + frac * (hi - lo)
        painter.drawText(QPointF(bar_x + bar_w + 6, y + tick_pt * 0.5), f"{val:.3g}")
    if label:
        painter.save()
        painter.translate(w - 6, h / 2)
        painter.rotate(-90)
        lf = QFont()
        lf.setPointSizeF(label_pt)
        lf.setBold(True)
        painter.setFont(lf)
        painter.drawText(QRectF(-h / 2, -label_pt - 6, h, label_pt + 6), Qt.AlignCenter, label)
        painter.restore()
    painter.restore()

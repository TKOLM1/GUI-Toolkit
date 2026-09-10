"""Shared field-grid plumbing for the results map view.

The map view draws a field layout (:mod:`ml.plot_layout`) as a grid of squares; this module owns
the small shared pieces it still uses — a button factory and the cell-origin math for the grid
squares. (Hit-testing now lives in :class:`gui.results.field_canvas.FieldCanvas`, which does its own
nearest-centre lookup, so the old matplotlib click-coordinate helpers were removed.)

The map view can swap in a user-chosen layout (different dimensions / fill order), so ``cell_origin``
takes an optional ``layout`` mapping; when omitted it falls back to the historical grid size.
"""

from __future__ import annotations

from PySide6.QtWidgets import QPushButton

from ml.plot_layout import N_COLS, N_ROWS


def make_button(text: str, slot) -> QPushButton:
    """A push button wired to ``slot`` (the sub-tabs build many of these)."""
    button = QPushButton(text)
    button.clicked.connect(slot)
    return button


def _dims(layout: dict[int, tuple[int, int]]) -> tuple[int, int]:
    """``(n_rows, n_cols)`` spanned by a layout (its max row/col + 1)."""
    if not layout:
        return N_ROWS, N_COLS
    rows = max(r for r, _ in layout.values()) + 1
    cols = max(c for _, c in layout.values()) + 1
    return rows, cols


def cell_origin(row: int, col: int, layout: dict[int, tuple[int, int]] | None = None) -> tuple[int, int]:
    """Bottom-left ``(x, y)`` of a plot's square (row 0 is drawn at the top)."""
    rows, _ = _dims(layout or {})
    return col, rows - 1 - row

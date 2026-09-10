"""The field layout for the results map: a small generator plus the historical default.

The physical field is a grid of plots numbered in a regular sweep. Historically this was an
8x10 boustrophedon ("snake"): the top row runs 438..429 right-to-left, the next 439..448
left-to-right, and so on, ending 499..508 on the bottom row.

That snake is no longer a hand-written table - it is :func:`build_layout` called with the
``SNAKE_*`` arguments. The generator covers every regular fill order from three independent
choices (start corner, row/column-major, snake on/off), so the results map's grid view lets the
user pick the field's dimensions and fill order (saved per project) without any special-casing.

The map's *default* layout is ``DEFAULT_*`` (start top-left, run right, step down).
:data:`PLOT_TO_CELL` and friends are pinned to the historical snake so the other results
consumers (e.g. the model-performance tab) keep importing the original ready-made mapping.
"""

from __future__ import annotations

# -- the map's default grid: 8x10, start top-left, run right then step down ------------------ #
DEFAULT_BASE = 429       # lowest plot number; sits at the start corner
DEFAULT_ROWS = 8
DEFAULT_COLS = 10
# The default fill order: start top-left, run right along each row, then step down, ending
# bottom-right. (The historical 8x10 *snake* is a different order, kept as the SNAKE_* values
# below and used for PLOT_TO_CELL.)
DEFAULT_START = "top-left"
DEFAULT_MAJOR = "row"        # numbers run along a row before stepping to the next
DEFAULT_SNAKE = False        # each line restarts from the same side (no boustrophedon)

# The original results-map numbering, kept available for callers that want the historical look.
SNAKE_START = "top-right"
SNAKE_MAJOR = "row"
SNAKE_SNAKE = True

# Back-compat aliases (older modules import these names).
BASE_PLOT = DEFAULT_BASE
N_ROWS = DEFAULT_ROWS
N_COLS = DEFAULT_COLS

START_CORNERS = ["bottom-left", "bottom-right", "top-left", "top-right"]
MAJOR_AXES = ["row", "column"]


def _line_order(count: int, ascending: bool) -> list[int]:
    return list(range(count)) if ascending else list(range(count - 1, -1, -1))


def build_layout(
    *,
    rows: int = DEFAULT_ROWS,
    cols: int = DEFAULT_COLS,
    base: int = DEFAULT_BASE,
    start: str = DEFAULT_START,
    major: str = DEFAULT_MAJOR,
    snake: bool = DEFAULT_SNAKE,
) -> dict[int, tuple[int, int]]:
    """Map consecutive plot numbers (from ``base``) onto ``(row, col)`` cells in a fill order.

    The order is fully determined by three independent choices, which together cover every
    regular sweep a field uses:

    * ``start`` - which corner the numbering starts in (one of :data:`START_CORNERS`). ``row``
      indices count from the top (row 0 = top), matching how the map is drawn.
    * ``major`` - ``"row"`` walks along a whole row before stepping to the next row;
      ``"column"`` walks down a whole column first.
    * ``snake`` - when True, every other line reverses (boustrophedon); when False, each line
      restarts from the same side.

    The default arguments give the map's default grid; the historical snake is
    ``build_layout(start=SNAKE_START, major=SNAKE_MAJOR, snake=SNAKE_SNAKE)``.
    """
    if rows <= 0 or cols <= 0:
        return {}
    top = "top" in start
    left = "left" in start

    # The two axes' base traversal directions, before snaking is applied.
    rows_order = _line_order(rows, ascending=top)        # top-start -> rows ascend (0,1,2,..)
    cols_order = _line_order(cols, ascending=left)       # left-start -> cols ascend

    mapping: dict[int, tuple[int, int]] = {}
    plot = base
    if major == "column":
        # Walk a whole column (down/up) before stepping across to the next column.
        for i, col in enumerate(cols_order):
            line = rows_order if not (snake and i % 2) else rows_order[::-1]
            for row in line:
                mapping[plot] = (row, col)
                plot += 1
    else:
        # Default: walk a whole row before stepping to the next row.
        for i, row in enumerate(rows_order):
            line = cols_order if not (snake and i % 2) else cols_order[::-1]
            for col in line:
                mapping[plot] = (row, col)
                plot += 1
    return mapping


# The ready-made historical *snake* mapping other modules (e.g. the model-performance tab)
# import directly - kept as the original look, independent of the new default fill order.
PLOT_TO_CELL: dict[int, tuple[int, int]] = build_layout(
    start=SNAKE_START, major=SNAKE_MAJOR, snake=SNAKE_SNAKE
)
CELL_TO_PLOT: dict[tuple[int, int], int] = {cell: plot for plot, cell in PLOT_TO_CELL.items()}
PLOTS: list[int] = sorted(PLOT_TO_CELL)


def cells():
    """Yield ``(plot, row, col)`` for every plot in the snake layout, in plot-number order."""
    for plot in PLOTS:
        row, col = PLOT_TO_CELL[plot]
        yield plot, row, col

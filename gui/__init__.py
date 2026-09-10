"""PySide6 GUI: the combined three-module shell."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .shell import MainShell
from .widgets import NoScrollFilter

__all__ = ["MainShell", "run"]


def run() -> int:
    """Launch the combined app; returns the Qt exit code."""
    app = QApplication.instance() or QApplication(sys.argv)
    # Apply the active preset's plot-label format process-wide before any page reads it.
    from common.config import CONFIG
    from common.naming import set_label_format
    set_label_format(CONFIG.label_start, CONFIG.label_end)
    # Seed the export style from its sidecar so the user's tuned thesis-figure settings persist.
    from .export import seed_spec_from_config
    seed_spec_from_config()
    # Stop the mouse wheel from changing unfocused spinbox/combo values while scrolling the panels.
    # Parented to the app so the filter object outlives this function (else it'd be garbage-collected).
    no_scroll = NoScrollFilter(app)
    app.installEventFilter(no_scroll)
    window = MainShell()
    window.show()
    return app.exec()

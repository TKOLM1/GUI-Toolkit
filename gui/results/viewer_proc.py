"""A GUI-side handle to the single persistent polyscope viewer subprocess.

The GUI process must never import polyscope (one ``ps.init`` per process; Qt + GL together is
unstable), so all 3-D rendering happens in :mod:`tools.view_laz`, launched once as a child
process. :class:`ViewerProcess` owns that child and talks to it through a small JSON control
file written atomically: each :meth:`send` bumps a monotonic ``seq`` the viewer watches, so the
displayed cloud(s) reload **in place** rather than spawning a second window.

A single instance is owned by the results page and shared by every sub-tab, which is what keeps
the "never two windows" guarantee.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_VIEWER = _PROJECT_ROOT / "tools" / "view_laz.py"
# Polyscope's built-in screenshot button auto-names files relative to the process cwd, with no
# API to set an output directory. Running the viewer from here keeps those dumps out of the
# project root and in one tidy folder instead.
_SCREENSHOT_DIR = _PROJECT_ROOT / "Polyscope screenshots"


class ViewerProcess:
    """Owns the one polyscope child process and its JSON control file."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._control_path = Path(tempfile.gettempdir()) / f"gp_viewer_{os.getpid()}.json"
        self._seq = 0

    # ------------------------------------------------------------------ #
    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def ensure_started(self) -> None:
        """Launch the viewer if it isn't already running."""
        if self.is_alive():
            return
        # Seed the control file so a freshly started viewer has nothing stale to apply.
        self._write({"seq": self._seq, "clouds": []})
        # Run the viewer from the screenshots folder so Polyscope's built-in screenshot button
        # drops its auto-named PNGs there instead of the project root. The viewer resolves its own
        # imports and cloud paths absolutely, so its cwd is otherwise unused.
        _SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        self._proc = subprocess.Popen(
            [sys.executable, str(_VIEWER), str(self._control_path)], cwd=str(_SCREENSHOT_DIR)
        )

    def send(
        self,
        clouds: list[dict],
        origin: list[float] | None = None,
        features: dict | None = None,
        slope_strata_pct: float | None = None,
        height_channel: str | None = None,
    ) -> None:
        """Show ``clouds`` (re)starting the viewer if needed. See module docstring for the schema.

        ``features`` (optional) asks the viewer to also draw the feature-geometry structures for
        one plot, e.g. ``{"path": "plot(429).laz"}``. ``slope_strata_pct`` (optional) is the
        top/bottom height % the project's features were generated with, so the drawn slope planes
        match the numbers; the viewer falls back to its default when it is ``None``.
        ``height_channel`` (optional) is the point dimension the project's features used for
        height-above-ground (e.g. ``RelativeHeight`` or ``Z``), so the drawn cloud and feature
        geometry sit at the same heights; the viewer falls back to ``RelativeHeight`` when ``None``.
        """
        self.ensure_started()
        self._seq += 1
        self._write(
            {"seq": self._seq, "origin": origin, "clouds": clouds, "features": features,
             "slope_strata_pct": slope_strata_pct, "height_channel": height_channel}
        )

    def close(self) -> None:
        """Terminate the viewer and remove its control file (best effort)."""
        if self.is_alive():
            self._proc.terminate()
        self._proc = None
        try:
            self._control_path.unlink()
        except OSError:
            pass

    # ------------------------------------------------------------------ #
    def _write(self, payload: dict) -> None:
        """Atomically write the control file (temp file + os.replace).

        On Windows ``os.replace`` fails with ``PermissionError`` (WinError 5/32) if the viewer
        happens to have the target open for reading at that instant. The viewer re-reads only on
        an mtime change and holds the file only briefly, so a short retry clears the collision.
        """
        tmp = self._control_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        for attempt in range(20):  # ~0.2 s worst case; collisions clear in a frame or two
            try:
                os.replace(tmp, self._control_path)
                return
            except PermissionError:
                time.sleep(0.01)
        os.replace(tmp, self._control_path)  # final try: surface the error if it truly persists

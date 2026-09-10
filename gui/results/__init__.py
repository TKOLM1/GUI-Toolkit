"""The results module's sub-tabs and the shared polyscope viewer handle.

The results page is split into sub-tabs (split consistency, the merged map + 3-D view, model
performance), each its own module here, hosted by :class:`gui.results_page.ResultsPage`. They share one
:class:`~gui.results.viewer_proc.ViewerProcess` so there is never more than one polyscope
window open.
"""

from .viewer_proc import ViewerProcess

__all__ = ["ViewerProcess"]

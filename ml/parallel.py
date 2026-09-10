"""Thread-budget arbiter shared by the parallel stages (optimizer, feature importance).

The problem this solves: a few stages fan out *independent fits* across threads (joblib's threading
backend — scikit-learn fits release the GIL, so this is a real multi-core win). But some estimators
*also* parallelise internally — notably :class:`RandomForestRegressor` is built with ``n_jobs=-1``
(all cores) when a single model is trained. If a stage spreads N fold/repeat fits across N threads
**and** each forest grabs all cores, you get N×cores threads fighting for the CPU — slower than serial.

So whenever a stage runs its own fit-level threading, it wraps that section in :func:`inner_fit_limit`,
and every self-parallelising factory reads :func:`inner_fit_n_jobs` to decide its own ``n_jobs``.
Inside the context the factory builds single-threaded (the outer threading owns the cores); outside it
(a single model train) the factory keeps using all cores as before. This is the *arbiter*: exactly one
level of parallelism is active at a time, never both.

Mechanics — deliberately minimal and safe:

* A single module-level integer guarded by a lock. The stage sets it on its *own* thread immediately
  before the ``joblib.Parallel(...)`` fan-out and restores it immediately after (``with`` block). The
  joblib worker threads only ever *read* it, and only while running strictly inside that ``with``
  block, so a plain shared integer is correct without per-thread state.
* Re-entrant by save/restore: nesting (e.g. a future caller that wraps two limited sections) restores
  the previous value, not an unconditional reset, so an outer limit survives an inner one.
* ``None`` (the default, unlimited) means "no stage is fanning out", so factories use their normal
  ``n_jobs`` (``-1`` for the forest). A positive value caps inner fits to that many threads — stages
  pass ``1`` so each fit is single-threaded while the fold/repeat loop owns the cores.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager

_lock = threading.Lock()
_inner_fit_n_jobs: int | None = None  # None = unlimited (no outer fan-out active)


def inner_fit_n_jobs(default: int = -1) -> int:
    """The ``n_jobs`` a self-parallelising estimator should use right now.

    Returns ``default`` (``-1`` = all cores) when no stage is fanning out fits across threads, or the
    active cap (the stages pass ``1``) when one is — so the forest builds single-threaded inside a
    threaded fold/repeat loop and oversubscription can't happen. Read by the model factories that set
    their own ``n_jobs`` (currently only Random Forest).
    """
    with _lock:
        return default if _inner_fit_n_jobs is None else _inner_fit_n_jobs


@contextmanager
def inner_fit_limit(n_jobs: int | None):
    """Temporarily cap self-parallelising estimators to ``n_jobs`` inner threads within this block.

    A stage that runs its *own* fit-level threading wraps the parallel section in this context with
    ``n_jobs=1`` so each fit stays single-threaded (the outer loop owns the cores). ``n_jobs=1`` is a
    no-op for the caps' purpose when the stage is serial — passing ``1`` is harmless either way, but
    callers only enter the context when they actually fan out, to keep single-model behaviour (``-1``)
    untouched. Restores the previous value on exit (nesting-safe), even on exception.
    """
    global _inner_fit_n_jobs
    with _lock:
        previous = _inner_fit_n_jobs
        _inner_fit_n_jobs = n_jobs
    try:
        yield
    finally:
        with _lock:
            _inner_fit_n_jobs = previous

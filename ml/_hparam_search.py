"""Shared hyperparameter-search helpers for both the classical and deep optimizers.

The search space for a model is *derived* from its :class:`HParam` list (the same specs that drive
the GUI form), so the classical and deep Optuna studies suggest parameters identically. This module
holds the two pieces both need: :func:`suggest` (draw one trial's parameters) and
:func:`describe_search_space` (a read-only, human-readable summary the GUI shows so nothing about
the space is hidden from the user).
"""

from __future__ import annotations

# A float range searched on a log scale once max/min spans this many orders of magnitude.
_LOG_SPAN = 1000.0


def _float_bounds(h, bounds: dict) -> tuple[float, float, bool]:
    """The effective (lo, hi, use_log) for a float HParam, honouring any runtime ``bounds``."""
    bmin, bmax = bounds.get(h.name, (None, None))
    lo = float(bmin if bmin is not None else (h.min if h.min is not None else 0.0))
    hi = float(bmax if bmax is not None else (h.max if h.max is not None else 1e9))
    # Tolerate floating-point error at the boundary (e.g. 1e-2/1e-5 evaluates to 999.999…) so a
    # range spanning exactly _LOG_SPAN orders of magnitude is still searched on a log scale.
    use_log = lo > 0 and hi / lo >= _LOG_SPAN * (1 - 1e-9)
    return lo, hi, use_log


def _int_bounds(h, bounds: dict) -> tuple[int, int, int]:
    """The effective (lo, hi, step) for an int HParam, honouring any runtime ``bounds``."""
    bmin, bmax = bounds.get(h.name, (None, None))
    lo = int(bmin if bmin is not None else (h.min if h.min is not None else 0))
    hi = int(bmax if bmax is not None else (h.max if h.max is not None else 1_000_000))
    step = int(h.step) if h.step else 1
    return lo, hi, step


def suggest(trial, hparams, *, bounds: dict | None = None) -> dict:
    """Draw one hyperparameter dict from ``trial`` using each :class:`HParam`'s metadata.

    ``bounds`` optionally overrides a parameter's ``(min, max)`` at runtime - used to clamp
    PLS ``n_components`` to the selected feature count, which the static HParam can't know.
    """
    bounds = bounds or {}
    params: dict = {}
    for h in hparams:
        if h.kind == "int":
            lo, hi, step = _int_bounds(h, bounds)
            params[h.name] = trial.suggest_int(h.name, lo, hi, step=step)
        elif h.kind == "float":
            lo, hi, use_log = _float_bounds(h, bounds)
            params[h.name] = trial.suggest_float(h.name, lo, hi, log=use_log)
        elif h.kind == "choice":
            params[h.name] = trial.suggest_categorical(h.name, list(h.choices or ()))
        else:  # bool
            params[h.name] = trial.suggest_categorical(h.name, [True, False])
    return params


def describe_search_space(hparams, *, bounds: dict | None = None) -> str:
    """A read-only, multi-line summary of the search space the optimizer will explore.

    Mirrors exactly what :func:`suggest` does (including the log-scale rule and any runtime
    ``bounds`` clamp), so the GUI can show the user the true space instead of hiding it. Returns a
    line per hyperparameter; an empty string when the model has no tunable hyperparameters.
    """
    bounds = bounds or {}
    lines: list[str] = []
    for h in hparams:
        if h.kind == "int":
            lo, hi, step = _int_bounds(h, bounds)
            step_txt = f", step {step}" if step != 1 else ""
            lines.append(f"{h.label}: integer [{lo} … {hi}]{step_txt}")
        elif h.kind == "float":
            lo, hi, use_log = _float_bounds(h, bounds)
            scale = " (log scale)" if use_log else ""
            lines.append(f"{h.label}: float [{lo:g} … {hi:g}]{scale}")
        elif h.kind == "choice":
            choices = ", ".join(str(c) for c in (h.choices or ()))
            lines.append(f"{h.label}: one of {{{choices}}}")
        else:  # bool
            lines.append(f"{h.label}: True or False")
    return "\n".join(lines)

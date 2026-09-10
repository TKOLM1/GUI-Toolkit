"""The shared shape of a training run: its config, its per-split results, and the metric cube.

This module owns the *data structures* every run produces, not a run loop of its own. A run is
performed by :func:`ml.validate.validate_procedure`, which partitions the plots into outer folds,
builds a model per fold and returns its folds as this module's :class:`SplitModel` objects inside a
:class:`TrainHistory`. Keeping those types here (rather than in the engine that fills them) is what
lets the bundle, the Results tab and the explainability code all read a run without depending on how
it was produced.

Each :class:`SplitModel` is one fold's full, selectable outcome: its fitted pipeline, its held-out
predictions, its **train and held-out relative RMSE (rRMSE %)**, a precomputed *metric cube* (every
scope × augmented-data combination, so the Results toggles never refit) and a per-plot prediction
table for every row (so Results switches splits by lookup). Every fold is kept, and any of them can
be made the active model; the lowest-held-out-rRMSE one is the default.

These are two-way grouped **train / held-out** splits, with no separate validation fold:
hyperparameters are selected by the optimiser's own internal cross-validation (see
:mod:`ml.optimize`) on the *inner* folds, so the held-out set here is a genuine out-of-sample test
set, not a tuning set. "Held-out" and "test" therefore name the same thing throughout this module;
the word "validation" is reserved for the optimiser's CV folds, where it is the textbook meaning.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import standard_metrics


class TrainControl:
    """Thread-safe pause/stop flag a worker sets and the training loop polls.

    The run calls :meth:`wait_if_paused` at each fold so a Pause blocks the worker thread there until
    Resume, and checks :attr:`stopped` to end early while keeping every fold finished so far. Default
    state is *running*.
    """

    def __init__(self) -> None:
        self._running = threading.Event()
        self._running.set()  # set == running; cleared == paused
        self._stop = False

    def pause(self) -> None:
        self._running.clear()

    def resume(self) -> None:
        self._running.set()

    def stop(self) -> None:
        self._stop = True
        self._running.set()  # unblock a paused loop so it can see the stop and exit

    def wait_if_paused(self) -> None:
        """Block while paused (returns immediately when running or stopped)."""
        self._running.wait()

    @property
    def stopped(self) -> bool:
        return self._stop


class TrainingStopped(Exception):
    """Raised when a run is stopped before any fold finished (nothing to keep)."""


@dataclass
class TrainConfig:
    """Everything needed to build one model: what to fit, on which columns, and how to split.

    The fold *counts* and the split *modes* are not here: they are arguments to
    :func:`ml.validate.validate_procedure`, because the outer and inner loops need one each and a
    single field could only describe one of them. ``test_size`` is the outer hold-out ratio.
    """

    model_key: str
    params: dict = field(default_factory=dict)
    feature_columns: list[str] | None = None  # None = all feature columns
    normalize_columns: list[str] | None = None  # which features to standardise; None = all
    test_size: float = 0.25       # fraction of plots held out per outer fold (the block size)
    seed: int = 0
    train_on_augmented: bool = True      # fit on augmented rows too? (off -> originals only)
    validate_on_augmented: bool = False  # score the held-out set on augmented rows too?
    # Target handling: "none" fits the raw target, "log" fits log1p(y) and back-transforms every
    # prediction into the target's own units (see :mod:`ml.target_transform`). ``target_bias_correction``
    # picks how the back-transform undoes the exponential's median bias: "smearing" (Duan, the default)
    # or "naive" (plain exponential, knowingly low). Because the inversion happens inside the fitted
    # estimator's ``predict``, every metric in this package is already in original units.
    target_transform: str = "none"
    target_bias_correction: str = "smearing"


@dataclass
class CycleResult:
    cycle: int                    # 1-based split number
    train_loss: float
    val_loss: float               # held-out test rRMSE (%) for this split
    val_r2: float = float("nan")  # held-out test R^2 for this split


@dataclass
class SplitModel:
    """One split's full, selectable outcome (every split is kept, not just the best).

    Any split can be made the *active* model: the results map and the model-performance view both
    read whichever :class:`SplitModel` ``TrainHistory.active_split`` points at, so the user can
    inspect any split, not only the one that scored best.
    """

    split: int                    # 1-based split number
    model: object                 # the split's fitted pipeline
    metrics: dict                 # standard metrics of this split's held-out test set
    predictions: pd.DataFrame     # this split's held-out test rows: plot, actual, predicted
    train_plots: set              # plots fitted in this split
    test_plots: set               # plots held out in this split
    n_train: int
    n_test: int
    train_metrics: dict = field(default_factory=dict)  # standard metrics on this split's fit rows
    # Precomputed metric "cube" so the Results tab can show metrics under either augmented-data
    # choice without recomputing at view time. Shape:
    #   metric_variants["held_out"][v] / ["train"][v]   with v in {"orig_only", "with_aug"}
    #   metric_variants["overall"][k]                    with k in
    #       {"orig_orig", "aug_orig", "orig_aug", "aug_aug"}  ("<train>_<val>" aug choices)
    # Each cell is a standard_metrics() dict. Empty for bundles saved before this field existed
    # (their consumers fall back via variant_metrics()).
    metric_variants: dict = field(default_factory=dict)
    # Predictions for EVERY plot+aug under this split's model (actual / predicted / plot / aug,
    # indexed by filename) — not just the held-out rows. Lets the Results tab build its per-plot
    # table by lookup, with no data source or model re-prediction (instant model/split switching).
    # error_pct and role are derived at view time (role from train_plots). Empty for bundles saved
    # before this field existed (their consumers fall back to predicting from the source).
    plot_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class TrainHistory:
    """The full outcome of a training run."""

    cycles: list[CycleResult]
    best_split: int               # 1-based split with the lowest test rRMSE
    final_metrics: dict           # rrmse / mae / r2 of the best split's held-out test set
    predictions: pd.DataFrame     # best split's test rows: actual, predicted, plot
    model: object                 # the best split's fitted pipeline
    target_column: str
    n_train: int
    n_test: int
    feature_columns: list[str] = field(default_factory=list)  # X columns the model was fit on
    best_train_plots: set = field(default_factory=set)        # plots fitted in the best split
    best_test_plots: set = field(default_factory=set)         # plots held out in the best split
    splits: list[SplitModel] = field(default_factory=list)    # every split, selectable
    active_split: int = 0         # 1-based split currently selected as the active model
    train_on_augmented: bool = True  # were augmented copies fitted in the training set?
    validate_on_augmented: bool = False  # were augmented copies scored in the held-out set?
    had_augmented: bool | None = None  # did the source have augmented rows? None = unknown (legacy)
    # Target handling this run used, carried so a saved bundle and the Results tab can state it.
    # "none" for every bundle saved before the transform existed (the field defaults to it).
    target_transform: str = "none"
    target_bias_correction: str = "smearing"


_STANDARD_KEYS = ("rrmse", "r2", "r", "mape")


def _mean_metric(values) -> dict:
    """Mean of each standard metric over an iterable of per-split metric dicts (ignoring NaNs)."""
    dicts = list(values)
    out: dict[str, float] = {}
    for key in _STANDARD_KEYS:
        arr = np.asarray([d.get(key, float("nan")) for d in dicts], dtype=float)
        out[key] = float(np.nanmean(arr)) if arr.size and np.isfinite(arr).any() else float("nan")
    return out


def average_split_metrics(history: TrainHistory) -> dict:
    """Mean of each standard held-out metric across every split in ``history``.

    Returns ``{'rrmse','r2','r','mape'}`` averaged over ``history.splits`` (ignoring NaNs), so the
    console/report can show a stable, all-split estimate rather than only the best split's numbers.
    """
    return _mean_metric(sm.metrics for sm in history.splits)


def average_split_train_metrics(history: TrainHistory) -> dict:
    """Mean of each standard *train* metric across every split — the train-side companion of
    :func:`average_split_metrics`, so the console can report averaged train AND test numbers."""
    return _mean_metric(sm.train_metrics for sm in history.splits)


def active_split_model(history: TrainHistory) -> SplitModel | None:
    """The :class:`SplitModel` for ``history.active_split`` (falls back to the best split)."""
    if not history.splits:
        return None
    want = history.active_split or history.best_split
    for sm in history.splits:
        if sm.split == want:
            return sm
    return history.splits[0]


def _compute_variants(predict, y, tr_full, te_full, is_aug) -> dict:
    """All (scope × augmented-data choice) standard metrics for one fitted split.

    ``predict(idx)`` returns predictions for the rows at positional indices ``idx``; ``y`` is the
    full target array/Series aligned to the same index space as ``tr_full``/``te_full``; ``is_aug``
    is a boolean mask over that space (True = an augmented copy). Held-out and training each get an
    ``orig_only`` and a ``with_aug`` cell; "overall" (train ∪ held-out) gets all four combinations of
    the two toggles, because metrics aren't linearly combinable — the mixed case (e.g. augmented in
    training but not in validation, the default) can't be derived from the pure ones. Cheap: a
    handful of predictions over already-available rows.
    """
    yv = np.asarray(y, dtype=float)

    def m(idx) -> dict:
        idx = np.asarray(idx, dtype=int)
        return standard_metrics(yv[idx], predict(idx)) if idx.size else {}

    tr_orig = tr_full[~is_aug[tr_full]]
    te_orig = te_full[~is_aug[te_full]]
    out: dict = {
        "held_out": {"orig_only": m(te_orig), "with_aug": m(te_full)},
        "train": {"orig_only": m(tr_orig), "with_aug": m(tr_full)},
        "overall": {},
    }
    for ta in (False, True):
        for va in (False, True):
            tr_sel = tr_full if ta else tr_orig
            te_sel = te_full if va else te_orig
            key = f"{'aug' if ta else 'orig'}_{'aug' if va else 'orig'}"
            out["overall"][key] = m(np.concatenate([tr_sel, te_sel]))
    return out


def variant_metrics(sm: SplitModel, scope: str, variant: str) -> dict:
    """One cell of a split's metric cube, with graceful fallback for older bundles.

    ``scope`` is ``"held_out"``/``"train"``/``"overall"``; ``variant`` is the matching cube key
    (``"orig_only"``/``"with_aug"`` for the first two, ``"<train>_<val>"`` for overall). Returns a
    standard_metrics dict, or ``{}`` when genuinely unavailable. Bundles saved before the cube
    existed have no ``metric_variants``; for those, held-out/training fall back to the single stored
    series (``sm.metrics``/``sm.train_metrics``, scored under that model's own train-time choice) and
    overall returns ``{}`` (it can't be reconstructed), which callers render as n/a.
    """
    cell = (getattr(sm, "metric_variants", None) or {}).get(scope, {}).get(variant)
    if cell:
        return cell
    if scope == "held_out":
        return getattr(sm, "metrics", {}) or {}
    if scope == "train":
        return getattr(sm, "train_metrics", {}) or {}
    return {}


def has_variant_cube(history: TrainHistory | None) -> bool:
    """True when every split in ``history`` carries a precomputed metric cube (a format-3 bundle)."""
    splits = getattr(history, "splits", None) or []
    return bool(splits) and all(getattr(sm, "metric_variants", None) for sm in splits)


def aug_toggles_meaningful(history: TrainHistory | None) -> bool:
    """True when the augmented-data toggles can actually change a metric for this run.

    The metric cube is built for *every* run, but when the source data had no augmented rows its
    ``with_aug`` cells equal its ``orig_only`` cells — so flipping a toggle does nothing. The GUI uses
    this to grey the toggles out for an originals-only run. Requires both a cube *and* the run having
    had augmented rows (``had_augmented``). Older bundles predate the flag, so an unset/missing
    ``had_augmented`` falls back to the cube check alone (today's behaviour), never regressing them.
    """
    if not has_variant_cube(history):
        return False
    had = getattr(history, "had_augmented", None)
    return True if had is None else bool(had)


def has_stored_predictions(history: TrainHistory | None) -> bool:
    """True when every split carries a full stored per-plot prediction table (a format-4 bundle).

    When True the Results tab can build its per-plot table by lookup, with no data source and no
    model re-prediction; when False it must predict from the source (the legacy path).
    """
    splits = getattr(history, "splits", None) or []
    if not splits:
        return False
    for sm in splits:
        frame = getattr(sm, "plot_predictions", None)
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return False
    return True

"""Model-agnostic feature-importance explanations for a saved classical model.

The Results tab can rank a model's features by **permutation importance**: shuffle one feature's
column and measure how much the model's score gets worse. The bigger the score drop, the more the
model relied on that feature. It is *model-agnostic* — it only ever feeds the fitted pipeline inputs
and reads its score, so it works identically for every classical model (Random Forest, SVR, k-NN,
GPR, the linear family, …), unlike native importances that exist only for trees and linear models.

Unlike the metric cube (which is precomputed in the bundle), this is a *new* analysis run live from
the feature workbook and the active split's fitted model, so it lives here as a pure, testable core
(mirroring :mod:`ml.results`) that the GUI sub-tab and the tests both call.

Two deliberate defaults guard against the method's one weakness — the randomness of a single shuffle:

* importance is scored on the split's **held-out plots** (leakage-free, and it measures importance
  for *generalisation*, not in-sample fit — what you actually want to report), and
* every feature is shuffled **many times** (``n_repeats``) and the score drops averaged, with the
  spread across repeats kept as a std so the caller can show how stable each ranking is.

Importance is reported as a **drop in rRMSE** (the app's standard loss): a value of ``2.0`` means
shuffling that feature worsens the held-out rRMSE by 2 percentage points on average. Higher = more
important; values near zero (or slightly negative, from shuffle noise) mean the model barely used it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from common.naming import aug_number_from_name

from .dataset import Dataset, make_xy
from .metrics import rrmse
from .trainer import TrainHistory, active_split_model

# Strong-against-randomness defaults: many repeats, scored on held-out plots, seeded.
DEFAULT_N_REPEATS = 30
DEFAULT_SCORE_ON = "held_out"  # "held_out" | "train" | "all"
SCORE_CHOICES = ("held_out", "train", "all")


@dataclass
class ImportanceResult:
    """A feature-importance ranking for one model/split, sorted most-important first.

    ``features`` / ``mean`` / ``std`` are aligned arrays: ``mean[i]`` is feature ``features[i]``'s
    average rRMSE increase when shuffled, ``std[i]`` the spread across the ``n_repeats`` shuffles
    (the error bar). ``baseline_rrmse`` is the model's rRMSE on the scored rows *before* any shuffle,
    so the importances can be read as "how much worse than this baseline". ``n_scored`` is how many
    rows were scored and ``score_on`` which subset (held-out / train / all).
    """

    features: list[str]
    mean: np.ndarray
    std: np.ndarray
    baseline_rrmse: float
    n_scored: int
    score_on: str
    n_repeats: int
    include_aug: bool = True
    notes: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.features) == 0


def _scored_rows(X: pd.DataFrame, plots: np.ndarray, active, score_on: str) -> np.ndarray:
    """Boolean mask over ``X``'s rows selecting the *plot* subset to score importance on.

    ``held_out`` keeps rows whose plot was held out of the active split (leakage-free, the default);
    ``train`` keeps the fitted plots; ``all`` keeps everything. Falls back to all rows when the split
    membership is unavailable (e.g. a legacy history with empty plot sets). Aug-copy filtering is
    applied separately by the caller — this mask is purely the plot membership.
    """
    if score_on == "all" or active is None:
        return np.ones(len(X), dtype=bool)
    test_plots = getattr(active, "test_plots", set()) or set()
    train_plots = getattr(active, "train_plots", set()) or set()
    want = test_plots if score_on == "held_out" else train_plots
    if not want:
        return np.ones(len(X), dtype=bool)
    return np.array([p in want for p in plots], dtype=bool)


def _aug_keep_rows(names: np.ndarray, score_on: str, include_aug: bool) -> np.ndarray:
    """Boolean mask dropping augmented copies unless ``include_aug`` is set for this subset.

    Mirrors the rest of the Results tab: an original plot (no ``aug(...)`` in its file name) always
    counts; an augmented copy counts only when the aug toggle for the side it sits on is on. For
    ``score_on == "all"`` (a mix of training and held-out rows) the single ``include_aug`` flag the
    host passes is the OR of the two toggles, so copies survive when *either* side wants them — the
    finer per-side split isn't knowable here without the plot membership, and the host only exposes a
    pooled choice for the "all" subset.
    """
    if include_aug:
        return np.ones(len(names), dtype=bool)
    return np.array([aug_number_from_name(str(n)) is None for n in names], dtype=bool)


def permutation_importance(
    dataset: Dataset,
    history: TrainHistory,
    *,
    n_repeats: int = DEFAULT_N_REPEATS,
    score_on: str = DEFAULT_SCORE_ON,
    include_aug: bool = True,
    seed: int = 0,
    n_jobs: int = 1,
) -> ImportanceResult:
    """Rank the model's features by permutation importance on the active split.

    Re-implements the permutation idea directly (rather than ``sklearn.inspection``) so the score is
    the app's own **rRMSE** — the same loss every other tab reports — and so the leakage-free
    held-out subset is selected by *plot group*, matching how the model was split. For each feature:
    shuffle its column ``n_repeats`` times (a fresh permutation each time, seeded for reproducibility)
    and record how much the held-out rRMSE rises versus the unshuffled baseline; the mean rise is the
    importance, the std the error bar.

    Runs on the **active split's** fitted pipeline, so the whole imputer→scaler→estimator is permuted
    end-to-end in raw-feature space (no need to know which columns the model scaled). Classical models
    only — a deep history carries no tabular ``feature_columns`` and yields an empty result.

    ``include_aug`` mirrors the Results tab's "Aug in … metrics" toggles: originals always score, but
    augmented copies of the scored plots are dropped unless ``include_aug`` is True, so the ranking is
    measured on the same row population the other sub-tabs report under the current choice.

    ``n_jobs`` (> 1) scores the shuffles in parallel across threads (joblib's threading backend —
    sklearn ``predict`` releases the GIL). Reproducibility is preserved: **all** permutation indices
    are drawn serially from the seeded ``rng`` first (same order as the serial path), and each task
    scores against its own private column copy (no shared-state race), so the threaded result matches
    the serial one to floating-point rounding — the same run-to-run agreement the serial path already
    has (some estimators' ``predict`` jitters by ~1e-15 regardless of threads). While the shuffles fan
    out, self-parallelising estimators are pinned single-threaded
    (:func:`ml.parallel.inner_fit_limit`) to avoid oversubscription. ``n_jobs == 1`` is the plain
    serial loop.
    """
    notes: list[str] = []
    if score_on not in SCORE_CHOICES:
        score_on = DEFAULT_SCORE_ON

    feature_columns = list(history.feature_columns)
    if not feature_columns:
        return ImportanceResult([], np.empty(0), np.empty(0), float("nan"), 0, score_on, n_repeats,
                                include_aug=include_aug,
                                notes=["This model has no tabular features to explain."])

    active = active_split_model(history)
    model = active.model if active is not None else history.model
    if model is None:
        return ImportanceResult([], np.empty(0), np.empty(0), float("nan"), 0, score_on, n_repeats,
                                include_aug=include_aug,
                                notes=["The selected split has no fitted model."])

    X, y, groups = make_xy(dataset, feature_columns)
    if len(X) == 0:
        return ImportanceResult([], np.empty(0), np.empty(0), float("nan"), 0, score_on, n_repeats,
                                include_aug=include_aug,
                                notes=["No rows with a target to score."])

    plots = np.array([int(str(g).split(":")[-1]) if ":" in str(g) else -1 for g in groups])
    mask = _scored_rows(X, plots, active, score_on)
    if not mask.any():
        mask = np.ones(len(X), dtype=bool)
        notes.append("No rows matched the chosen subset; scored on all rows instead.")

    # Drop augmented copies unless the toggle for this subset wants them (originals always stay).
    aug_mask = _aug_keep_rows(np.asarray(X.index), score_on, include_aug)
    if (mask & aug_mask).any():
        mask = mask & aug_mask
    elif not include_aug:
        notes.append("No original (non-augmented) rows in the chosen subset; scored on all copies.")

    Xs = X.loc[mask].reset_index(drop=True)
    ys = y.loc[mask].to_numpy(dtype=float)
    n_scored = len(Xs)
    if n_scored < 2:
        return ImportanceResult([], np.empty(0), np.empty(0), float("nan"), n_scored, score_on,
                                n_repeats, include_aug=include_aug,
                                notes=["Too few rows to score importance."])

    baseline = rrmse(ys, np.ravel(model.predict(Xs)))

    # Draw EVERY permutation up front, serially, in the same order the serial loop would
    # (feature-by-feature, repeat-by-repeat). The scoring below depends only on these fixed indices,
    # so the result is identical whether the shuffles run serial or threaded — reproducibility (same
    # seed -> same ranking) is guaranteed regardless of n_jobs.
    rng = np.random.default_rng(seed)
    perms = [[rng.permutation(n_scored) for _ in range(n_repeats)] for _ in feature_columns]
    base_values = {col: Xs[col].to_numpy(copy=True) for col in feature_columns}

    def _shuffled_drop(col: str, perm: np.ndarray) -> float:
        """Score one shuffle of ``col`` against its own private copy of the feature frame.

        A fresh copy per call means threads never touch shared mutable state — only ``col`` is replaced
        by its permuted values, every other column untouched — so the parallel path is race-free."""
        Xp = Xs.copy()
        Xp[col] = base_values[col][perm]
        return rrmse(ys, np.ravel(model.predict(Xp))) - baseline

    # Flatten (feature, repeat) into one task list so the threading is a single clean fan-out. Folding
    # the results back per feature recovers the per-feature mean/std exactly as the nested loop did.
    tasks = [(j, col, perms[j][r]) for j, col in enumerate(feature_columns) for r in range(n_repeats)]
    if n_jobs > 1 and len(tasks) > 1:
        from joblib import Parallel, delayed

        from .parallel import inner_fit_limit

        # Pin self-parallelising estimators single-threaded while the shuffles own the cores.
        with inner_fit_limit(1):
            flat = Parallel(n_jobs=n_jobs, backend="threading")(
                delayed(_shuffled_drop)(col, perm) for _, col, perm in tasks
            )
    else:
        flat = [_shuffled_drop(col, perm) for _, col, perm in tasks]

    means = np.empty(len(feature_columns), dtype=float)
    stds = np.empty(len(feature_columns), dtype=float)
    for j in range(len(feature_columns)):
        drops = np.asarray(flat[j * n_repeats:(j + 1) * n_repeats], dtype=float)
        means[j] = float(np.mean(drops))
        stds[j] = float(np.std(drops))

    order = np.argsort(means)[::-1]  # most important first
    return ImportanceResult(
        features=[feature_columns[i] for i in order],
        mean=means[order],
        std=stds[order],
        baseline_rrmse=float(baseline),
        n_scored=n_scored,
        score_on=score_on,
        n_repeats=n_repeats,
        include_aug=include_aug,
        notes=notes,
    )

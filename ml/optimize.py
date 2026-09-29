"""Smart hyperparameter optimisation with Optuna - leakage-safe, plot-grouped CV.

The GUI's "Optimize hyperparameters" button runs an Optuna (TPE / Bayesian) study over the
selected model's hyperparameter space. Each trial is scored by **grouped K-fold cross-validation**
on the loaded data: folds are split by plot number (``groups`` from :func:`make_xy`), so a plot's
originals and augmented copies never span a fold, and the model pipeline (imputer/scaler) is rebuilt
and fit *inside* every fold - the same leakage guards the trainer uses. The validation rows of each
fold honour the same "validate on augmented?" rule as training, so tuning optimises the metric the
user will actually report.

The study returns the best hyperparameter dict, which the GUI writes back into the form. For a quick
inspection it also logs (via ``note``) the best trial's mean rRMSE and mean R² across its CV folds —
read straight back from that trial, not recomputed — so the user can see how the tuned config
generalised over the folds before training the final model. Optuna is imported lazily so the rest of
the app runs without it installed.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from common.naming import aug_number_from_name
from ._hparam_search import describe_search_space, suggest as _suggest
from .dataset import Dataset, make_xy
from .metrics import mape, r, r2, rrmse
from .models import MODELS_BY_KEY
from .parallel import inner_fit_limit
from .splitting import (
    grouped_split,
    max_splits_for_test_size,
    random_systematic_group_split_by_count,
    sequential_group_split_by_count,
    systematic_group_split_by_count,
)
from .target_transform import build_estimator
from .trainer import TrainConfig

# Called once per finished trial with (trial_number, total_trials, best_value_so_far).
ProgressCallback = Callable[[int, int, float], None]


def _safe_nanmean(values) -> float:
    """``np.nanmean`` that returns NaN (no warning) when every value is NaN or the list is empty."""
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if arr.size and np.isfinite(arr).any() else float("nan")


def pls_bounds(model_key: str, n_features: int) -> dict:
    """Runtime search-space clamps for ``model_key`` given the selected feature count.

    PLS requires ``n_components <= n_features``; the static HParam max (20) can exceed the number of
    selected columns, so the search space is clamped here. Other models have no runtime clamp.
    Shared by the optimizer and the GUI's search-space readout so they agree.
    """
    if model_key == "pls":
        return {"n_components": (1, min(20, max(1, n_features)))}
    return {}


def classical_search_space_text(model_key: str, n_features: int | None = None) -> str:
    """Read-only summary of a classical model's Optuna search space (for the GUI)."""
    bounds = pls_bounds(model_key, n_features) if n_features is not None else {}
    # Only the *core* hyperparameters are tuned; iteration limits (max_iter/tol) are solver controls,
    # not quality knobs, so they are fixed at the form value and excluded from the search readout.
    return describe_search_space(MODELS_BY_KEY[model_key].core_hparams(), bounds=bounds)


def sampler_seed_for_folds(seed: int | None, config: TrainConfig) -> int:
    """The effective seed used both for the TPE sampler and the random CV-fold shuffle."""
    return config.seed if seed is None else seed


def grouped_cv_folds(groups, n_splits: int, split_mode: str, test_size: float, seed: int):
    """Grouped CV folds (plots never span a fold), built from the *same* split machinery the
    trainer uses so the optimizer and the final run agree exactly.

    The three without-replacement partition modes all deal disjoint hold-out blocks of ``round(test_size
    · n_groups)`` plots — the block shrunk to ``⌊n_groups / n_splits⌋`` when the rounded ratio would
    overrun, so all ``n_splits`` blocks tile the field (honouring both ratio and count) — and return
    ``min(n_splits, ⌊1 / test_size⌋)`` folds (the splits·ratio ≤ 1 cap, so the blocks never
    overlap): ``"sequential"`` = contiguous blocks
    (:func:`sequential_group_split_by_count`), ``"systematic"`` = strided blocks every ``n_splits``-th
    plot (:func:`systematic_group_split_by_count`), ``"random_systematic"`` = one-shuffle-then-contiguous
    blocks (:func:`random_systematic_group_split_by_count`, seeded by ``seed``). All three are
    leakage-free — the held-out blocks don't overlap, every plot held out at most once — differing only
    in layout. For ``"random"`` exactly ``n_splits`` **independent** random grouped folds are drawn at
    the ``test_size`` ratio, seeded by ``seed + i`` so the study reproduces (the one mode whose folds may
    overlap, and whose count is unbounded by the ratio).
    """
    groups = np.asarray(groups)
    # The three without-replacement partition modes all deal disjoint hold-out blocks; the block is
    # shrunk to fit all ``n_splits`` of them (``_block_size``), so the only remaining cap is the
    # continuous splits·ratio ≤ 1 (``max_splits_for_test_size``) — clamp the fold count to it so a
    # request past that cap can't drive blocks below a sensible size. The GUI's cap normally makes this
    # a no-op.
    if split_mode in ("sequential", "systematic", "random_systematic"):
        n_blocks = min(n_splits, max_splits_for_test_size(len(np.unique(groups)), test_size))
        if split_mode == "sequential":
            return [
                sequential_group_split_by_count(groups, n_blocks, test_size, cycle=i)
                for i in range(n_blocks)
            ]
        if split_mode == "systematic":
            return [
                systematic_group_split_by_count(groups, n_blocks, test_size, cycle=i)
                for i in range(n_blocks)
            ]
        # random_systematic: one shared shuffle seeded by ``seed`` (NOT ``seed + i``) so the per-fold
        # blocks dovetail into a single without-replacement partition; the fold index picks the block.
        return [
            random_systematic_group_split_by_count(groups, n_blocks, test_size, seed, cycle=i)
            for i in range(n_blocks)
        ]
    return [
        grouped_split(groups, test_size, seed + i, mode="random", cycle=i)
        for i in range(n_splits)
    ]


# Internal alias kept terse at the call site.
_grouped_cv_folds = grouped_cv_folds


def optimize_hyperparameters(
    dataset: Dataset,
    config: TrainConfig,
    n_trials: int,
    n_cv_splits: int = 5,
    *,
    seed: int | None = None,
    split_mode: str = "random",
    fit_on_augmented: bool = True,
    validate_on_augmented: bool = False,
    n_jobs: int = 1,
    note: Callable[[str], None] | None = None,
    progress: ProgressCallback | None = None,
    control=None,
) -> dict:
    """Search the model's hyperparameters with Optuna; return the best parameter dict.

    Scoring is grouped K-fold CV (by plot), with the pipeline rebuilt per fold so scaling is fit on
    training rows only. ``config`` supplies the model, the selected feature columns and the
    per-feature ``normalize_columns``. The two augmented-data rules are passed **separately** from
    the main-model config: ``fit_on_augmented`` drops augmented rows from each fold's training set
    when off, and ``validate_on_augmented`` drops them from each fold's validation set when off
    (both off => augmented data is excluded from the search entirely).

    ``n_cv_splits`` sets the number of grouped CV folds. ``seed`` seeds the TPE sampler; when
    ``None`` it falls back to ``config.seed``. ``note`` receives one-off informational lines
    (e.g. the PLS ``n_components`` clamp) for the GUI log. ``control`` (a :class:`ml.TrainControl`)
    lets the caller Pause (blocks at the next trial boundary) or Stop (ends the search after the
    current trial, keeping the trials finished so far).

    ``n_jobs`` (> 1) scores each trial's CV folds in parallel across threads (joblib's threading
    backend — sklearn fits release the GIL). The folds are independent and each rebuilds its own
    pipeline on its own training rows, so this is **leakage-neutral** and the per-trial mean is
    order-independent — the threaded search returns the same best params as the serial one (to
    floating-point rounding). While the folds fan out, self-parallelising estimators are pinned
    single-threaded (see :func:`ml.parallel.inner_fit_limit`) so the two levels of threading never
    oversubscribe the CPU. ``n_jobs == 1`` is the plain serial loop.
    """
    import optuna  # lazy: keeps optuna optional for the rest of the app

    X, y, groups = make_xy(dataset, config.feature_columns)
    if len(X) < 2:
        raise ValueError("Not enough labelled rows to optimise (need at least 2).")
    cols = list(X.columns)
    model_def = MODELS_BY_KEY[config.model_key]

    # Iteration limits (max_iter/tol) are not tuned — but they ARE honoured. Carry the user's chosen
    # values from config.params into every trial's fit so the form's "Iteration limits" controls (a
    # higher cap, a looser tolerance) take effect during tuning too. Without this, the inner CV fits
    # would silently fall back to the model-def defaults and ignore the user's setting. Falls back to
    # the model default for any limit the caller didn't supply.
    iter_limits = {
        h.name: config.params.get(h.name, h.default)
        for h in model_def.iteration_limit_hparams()
    }

    is_aug = np.array([aug_number_from_name(str(i)) is not None for i in X.index])
    if (not fit_on_augmented or not validate_on_augmented) and not (~is_aug).any():
        raise ValueError(
            "No original (non-augmented) rows available. Turn on the hyperopt augmented-data "
            "toggles or load some original plots."
        )

    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise ValueError(
            f"Need at least 2 distinct plot groups for cross-validation; found {n_groups}."
        )
    # The user's requested fold count, clamped only to what the field can supply. A single fold is a
    # degenerate but valid single-holdout score (no averaging); the only hard floor is ≥2 distinct
    # groups so a grouped split exists at all.
    n_splits = max(1, min(n_cv_splits, n_groups))
    folds = _grouped_cv_folds(
        groups, n_splits, split_mode, config.test_size, sampler_seed_for_folds(seed, config)
    )

    # PLS requires n_components <= n_features; the static HParam max (20) can exceed the
    # number of selected columns, so clamp the search space to the feature count here.
    bounds = pls_bounds(config.model_key, len(cols))
    if config.model_key == "pls" and note is not None and len(cols) < 20:
        note(f"PLS n_components capped at {bounds['n_components'][1]} (selected feature count).")

    def _score_fold(tr, va, params):
        """Score one CV fold for ``params``; return ``(rrmse, r2, r, mape)`` or ``None`` if skipped.

        Self-contained (reads only this fold's row indices + the shared, read-only X/y/is_aug and the
        trial's params), so it is safe to call from several threads at once: each call rebuilds its own
        pipeline and fits on its own training rows — no shared mutable state, no cross-fold leakage.
        """
        tr_fit = tr if fit_on_augmented else tr[~is_aug[tr]]
        va_eval = va if validate_on_augmented else va[~is_aug[va]]
        if len(tr_fit) == 0 or len(va_eval) == 0:
            return None  # this fold lacks original rows for fitting/scoring; skip it
        # Built through build_estimator so the inner search honours the run's target transform: the
        # trial is scored on back-transformed predictions, i.e. in the same units — and against the same
        # rRMSE — the final run reports. Tuning in log space and reporting in raw space would optimise
        # the wrong objective.
        pipe = build_estimator(
            model_def, params, normalize_columns=config.normalize_columns, feature_columns=cols,
            target_transform=config.target_transform,
            bias_correction=config.target_bias_correction, seed=config.seed,
        )
        pipe.fit(X.iloc[tr_fit], y.iloc[tr_fit])
        y_va = y.iloc[va_eval]
        pred = np.ravel(pipe.predict(X.iloc[va_eval]))
        return rrmse(y_va, pred), r2(y_va, pred), r(y_va, pred), mape(y_va, pred)

    def objective(trial) -> float:
        """Mean held-out rRMSE across the CV folds for this trial's suggested hyperparameters.

        Also records the mean R² and the fold count on the trial, so the post-study summary can read
        the best trial's metrics back by name without re-fitting anything.
        """
        # Tune only the core hyperparameters; iteration limits (max_iter/tol) are solver controls,
        # not quality knobs, so Optuna never sweeps them. They are instead carried in from the form
        # (iter_limits) so the user's chosen cap/tolerance applies to these inner fits as well.
        params = {**_suggest(trial, model_def.core_hparams(), bounds=bounds), **iter_limits}
        # Score the folds, in parallel across threads when n_jobs > 1. Folds are independent and each
        # rebuilds its own pipeline, so the result is identical to the serial loop — only faster. While
        # they fan out, pin self-parallelising estimators (the forest) to one thread so threads × folds
        # don't oversubscribe the CPU. The per-trial mean below is order-independent, so determinism
        # (seeded fits) holds regardless of n_jobs.
        if n_jobs > 1 and len(folds) > 1:
            from joblib import Parallel, delayed

            with inner_fit_limit(1):
                fold_results = Parallel(n_jobs=n_jobs, backend="threading")(
                    delayed(_score_fold)(tr, va, params) for tr, va in folds
                )
        else:
            fold_results = [_score_fold(tr, va, params) for tr, va in folds]

        scored = [fr for fr in fold_results if fr is not None]
        if not scored:
            raise ValueError("No fold had original rows for both fitting and validation.")
        fold_rrmse = [s[0] for s in scored]
        fold_r2 = [s[1] for s in scored]
        fold_r = [s[2] for s in scored]
        fold_mape = [s[3] for s in scored]
        mean_rrmse = float(np.mean(fold_rrmse))
        trial.set_user_attr("mean_rrmse", mean_rrmse)
        trial.set_user_attr("mean_r2", float(np.mean(fold_r2)))
        trial.set_user_attr("mean_r", _safe_nanmean(fold_r))
        trial.set_user_attr("mean_mape", float(np.mean(fold_mape)))
        trial.set_user_attr("n_folds", len(fold_rrmse))
        return mean_rrmse

    sampler_seed = sampler_seed_for_folds(seed, config)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=sampler_seed)
    )

    def _callback(study_, trial_) -> None:
        if progress is not None:
            progress(trial_.number + 1, n_trials, float(study_.best_value))
        # Pause/Stop are honoured at trial boundaries (each trial is short). Pause blocks the worker
        # thread here; Stop asks Optuna to finish after this trial, keeping completed trials.
        if control is not None:
            control.wait_if_paused()
            if control.stopped:
                study_.stop()

    study.optimize(objective, n_trials=n_trials, callbacks=[_callback])

    # Quick-inspection summary: the best trial's mean rRMSE and mean R² across its CV folds, read
    # back from the metrics it recorded during the study (no re-fitting). Reported through ``note``
    # so it lands in the optimization log without changing the return contract (callers still
    # receive the params dict).
    if note is not None:
        best = study.best_trial.user_attrs
        note(
            f"Best trial across {best.get('n_folds', n_splits)} CV fold(s): "
            f"mean rRMSE={best.get('mean_rrmse', float('nan')):.4g}%, "
            f"R²={best.get('mean_r2', float('nan')):.4g}, "
            f"R={best.get('mean_r', float('nan')):.4g}, "
            f"MAPE={best.get('mean_mape', float('nan')):.4g}%."
        )
    # Return the tuned core params, plus the iteration limits actually used (the user's form values,
    # not just the model defaults), so a downstream fit (e.g. validate's per-fold final fit) honours
    # them. defaults() underneath fills any param neither tuned nor an iteration limit.
    return {**model_def.defaults(), **study.best_params, **iter_limits}

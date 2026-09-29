"""Rotating-vault nested cross-validation — an honest estimate of the whole
optimize→train procedure, not of a single fitted model.

The optimizer tunes hyperparameters using *all* the labelled plots. If the final model's CV error is
then reported on those same plots, the test plots have already influenced *which hyperparameters were
chosen* — **selection bias**. Per-fold scaling is leakage-safe, but the tuning step is not, and on ~80
plots that bias can be several points of rRMSE.

The fix is **nested cross-validation**, framed here as a "rotating vault":

* **Outer loop — a leakage-free without-replacement partition.** The plots are dealt into a user-set
  number of disjoint hold-out folds of ``round(test_size · n_groups)`` plots each (no plot held out
  twice; ``n_outer · block ≤ n_groups``, the GUI's splits·ratio ≤ 1 cap — at the product 1 every plot is
  held out exactly once, below it the surplus plots stay untested): contiguous blocks (``"sequential"``,
  :func:`ml.splitting.sequential_group_split_by_count`), interleaved strides (``"systematic"``,
  :func:`ml.splitting.systematic_group_split_by_count`, every n-th plot), or shuffled hold-out blocks
  (``"random_systematic"``, :func:`ml.splitting.random_systematic_group_split_by_count`, one shared
  shuffle so the blocks tile). Plain ``"random"`` is not a partition (its draws overlap) and is not
  offered for the outer loop. The held-out fold is the "vault": invisible to everything that builds the
  model scored on it.
* **Inside each outer fold** the *entire* chosen procedure re-runs on the outer-train plots only —
  optionally the optimizer (re-tuning hyperparameters), then a fit — and is scored on the untouched
  outer-test block. Because each block is reserved by a procedure blind to it, every plot gets one
  honest test.
* **The inner CV is the user's choice.** The optimizer step *inside* each outer fold splits the
  outer-train plots by ``inner_split_mode`` at ``inner_n_splits``. Random folds keep the inner
  noise-averaging defence at full strength; the sequential or systematic partitions make the whole
  procedure deterministic end to end.

With ``do_optimize=False`` the run is simply an honest cross-validation of the fixed pipeline the user
configured — the "just train it" path — so this one function covers both the plain and the tuned case.

:func:`sweep_models` runs that whole procedure once per model in the registry and reports them ranked,
so "try everything and keep the best" is one call.

The result is a **measurement and a model**: :class:`VaultResult` carries a :class:`TrainHistory`
whose ``splits`` are the outer folds (so the existing Results "split consistency" view renders the
per-fold spread directly), plus the per-fold tuned params (a stability signal the bundle-centric views
can't show). The mean held-out rRMSE is the honest label to ship it under.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import pandas as pd

from common.naming import aug_number_from_name, plot_number_from_name
from .dataset import Dataset, make_xy
from .metrics import standard_metrics
from .models import MODELS, MODELS_BY_KEY
from .optimize import optimize_hyperparameters
from .splitting import (
    max_splits_for_test_size,
    random_systematic_group_split_by_count,
    sequential_group_split_by_count,
    systematic_group_split_by_count,
)
from .target_transform import LOG, build_estimator, describe, forward
from .trainer import (
    CycleResult,
    SplitModel,
    TrainConfig,
    TrainHistory,
    TrainingStopped,
    _compute_variants,
)

# Called once per finished outer fold with (fold_number, total_folds, held_out_rrmse, train_rrmse,
# train_metrics, test_metrics). The two trailing dicts are the fold's full standard_metrics() for the
# outer-train fit and the outer-test (held-out) rows, so the GUI can plot every metric per outer fold.
ProgressCallback = Callable[[int, int, float, float, dict, dict], None]

# Called during an outer fold's INNER optimizer search with
# (fold_number, total_folds, stage, trial, total_trials) — ``stage`` is "optimize". Lets the GUI keep
# its live status line ticking through the long inner searches (which are where a nested-CV run spends
# most of its time), not just at outer-fold boundaries.
InnerProgressCallback = Callable[[int, int, str, int, int], None]


@dataclass
class VaultResult:
    """The outcome of one rotating-vault run: an honest per-fold measurement, no saved model.

    ``history`` is a :class:`TrainHistory` whose ``splits`` are the outer folds (each fold's fitted
    model, held-out predictions and full metric cube), so the Results "split consistency" view renders
    the per-fold spread unchanged. ``fold_features``/``fold_params`` record the features used and the
    hyperparameters tuned in each fold — the cross-fold stability the per-split view can't show (stable
    ⇒ the tuning is robust; jumpy ⇒ the headline number leaned on luck). ``did_optimize`` notes whether
    the vault wrapped the optimizer or scored fixed hyperparameters. ``model_key`` records which model
    produced it, so a sweep's results can be ranked and the winner named.
    """

    history: TrainHistory
    fold_features: list[list[str]]
    fold_params: list[dict]
    did_optimize: bool
    target_column: str
    n_outer: int
    model_key: str = ""

    @property
    def mean_held_out_rrmse(self) -> float:
        arr = np.asarray([sm.metrics.get("rrmse", float("nan")) for sm in self.history.splits],
                         dtype=float)
        return float(np.nanmean(arr)) if arr.size and np.isfinite(arr).any() else float("nan")

    @property
    def std_held_out_rrmse(self) -> float:
        arr = np.asarray([sm.metrics.get("rrmse", float("nan")) for sm in self.history.splits],
                         dtype=float)
        return float(np.nanstd(arr)) if arr.size and np.isfinite(arr).any() else float("nan")


def _sub_dataset(dataset: Dataset, train_plots: set) -> Dataset:
    """A copy of ``dataset`` whose frame keeps only rows belonging to ``train_plots``.

    Filtering is by plot number (``plot(N)``), so a plot's augmented copies travel with it — required
    for the inner feature-selection/optimizer augmented-data toggles to behave exactly as on the full
    dataset.
    The feature/target metadata is carried over unchanged; only the rows shrink.
    """
    keep = [plot_number_from_name(str(idx)) in train_plots for idx in dataset.frame.index]
    return Dataset(
        frame=dataset.frame.loc[keep],
        feature_columns=list(dataset.feature_columns),
        target_column=dataset.target_column,
    )


def _outer_cycle_seeds(base_seed: int, n_outer: int) -> list[int]:
    """One independent seed per outer cycle for its inner optimizer search.

    The inner search derives its random-mode fold seeds as ``seed + fold``, so plain ``base_seed + i``
    per cycle made cycle ``i`` fold ``j`` collide with cycle ``i+1`` fold ``j-1``. Spawning children of
    one :class:`numpy.random.SeedSequence` keeps them reproducible but statistically independent.
    Kept to 31 bits so the inner ``+ fold`` offset stays inside numpy's legacy 32-bit seed range.
    """
    children = np.random.SeedSequence(base_seed).spawn(n_outer)
    return [int(c.generate_state(1)[0]) >> 1 for c in children]


def _fit_score_fold(dataset: Dataset, config: TrainConfig, fold_features: list[str],
                    tr_full, te_full, split_index: int) -> SplitModel:
    """Fit one model on the outer-train rows and score the outer-test rows — a single fixed split.

    Returns the fold's :class:`SplitModel` carrying everything a saved bundle needs: its fitted
    pipeline, its held-out predictions, the full metric cube (so the Results augmented-data toggles
    work) and a per-plot prediction table for every row (so Results switches splits by lookup rather
    than re-predicting). It is built over the *whole* dataset's index space, so ``tr_full`` /
    ``te_full`` / ``is_aug`` all stay consistent.
    """
    X, y, _ = make_xy(dataset, fold_features)
    cols = list(X.columns)
    is_aug = np.array([aug_number_from_name(str(i)) is not None for i in X.index])

    # Honour the two augmented-data toggles exactly as the trainer does.
    tr_fit = tr_full if config.train_on_augmented else tr_full[~is_aug[tr_full]]
    te_eval = te_full if config.validate_on_augmented else te_full[~is_aug[te_full]]
    if len(tr_fit) == 0 or len(te_eval) == 0:
        raise ValueError(
            "An outer fold had no original rows to fit on or score — enable the augmented-data "
            "toggles or adjust the test split."
        )

    model_def = MODELS_BY_KEY[config.model_key]
    # When the target transform is on, this is a TransformedTargetPipeline: it fits on log1p(y) and
    # its ``predict`` returns the target's own units (bias-corrected), so every metric, cube cell and
    # stored prediction below is in original units without any further handling here.
    pipe = build_estimator(
        model_def, config.params, normalize_columns=config.normalize_columns,
        feature_columns=cols, target_transform=config.target_transform,
        bias_correction=config.target_bias_correction, seed=config.seed,
    )
    pipe.fit(X.iloc[tr_fit], y.iloc[tr_fit])

    y_fit = y.iloc[tr_fit]
    train_pred = np.ravel(pipe.predict(X.iloc[tr_fit]))
    train_metrics = standard_metrics(y_fit, train_pred)
    y_test = y.iloc[te_eval]
    test_pred = np.ravel(pipe.predict(X.iloc[te_eval]))
    test_metrics = standard_metrics(y_test, test_pred)

    preds = pd.DataFrame(
        {"actual": y_test.to_numpy(), "predicted": test_pred}, index=X.index[te_eval]
    )
    preds.insert(0, "plot", [plot_number_from_name(str(i)) for i in preds.index])
    preds.index.name = "filename"

    # The predict closure, y, tr_full/te_full and is_aug all index the SAME (whole-dataset) space, so
    # the metric cube slices are consistent — this consistency is the most likely bug site, so the
    # closure deliberately uses the same X the masks were built from.
    assert len(is_aug) == len(X), "is_aug must align with the dataset's row index"
    variants = _compute_variants(
        lambda idx: np.ravel(pipe.predict(X.iloc[idx])), y, tr_full, te_full, is_aug
    )

    # Predictions for EVERY plot+aug under this fold's model, so a saved bundle carries its own
    # per-plot table (format 4) and the Results tab can build it by lookup — no data source, no
    # re-prediction, instant model/split switching. One extra full predict per fold.
    all_pred = np.ravel(pipe.predict(X))
    plot_preds = pd.DataFrame(
        {
            "actual": y.to_numpy(dtype=float),
            "predicted": all_pred,
            "plot": [plot_number_from_name(str(i)) for i in X.index],
            "aug": [aug_number_from_name(str(i)) or 0 for i in X.index],
        },
        index=X.index,
    )
    plot_preds.index.name = "filename"

    index = X.index.to_numpy()
    return SplitModel(
        split=split_index,
        model=pipe,
        metrics=test_metrics,
        predictions=preds,
        train_plots={plot_number_from_name(str(i)) for i in index[tr_full]},
        test_plots={plot_number_from_name(str(i)) for i in index[te_full]},
        n_train=len(tr_fit),
        n_test=len(te_eval),
        train_metrics=train_metrics,
        metric_variants=variants,
        plot_predictions=plot_preds,
    )


def validate_procedure(
    dataset: Dataset,
    config: TrainConfig,
    *,
    do_optimize: bool = True,
    n_outer_splits: int = 4,
    outer_split_mode: str = "sequential",
    inner_test_size: float | None = None,
    inner_split_mode: str = "sequential",
    opt_n_cv_splits: int = 5,
    opt_trials: int = 50,
    opt_fit_on_augmented: bool = True,
    opt_validate_on_augmented: bool = False,
    opt_n_jobs: int = 1,
    seed: int | None = None,
    note: Callable[[str], None] | None = None,
    progress: ProgressCallback | None = None,
    inner_progress: InnerProgressCallback | None = None,
    control=None,
) -> VaultResult:
    """Run nested CV over the optimize→train procedure; return a :class:`VaultResult`.

    The OUTER loop is a leakage-free partition into exactly ``n_outer_splits`` folds (the user-set
    count; every plot held out exactly once). ``outer_split_mode`` picks the layout: ``"sequential"``
    (default) tiles the field into contiguous blocks (the last absorbing the remainder); ``"systematic"``
    strides it into interleaved folds (every ``n_outer_splits``-th plot), so each fold samples the whole
    field — the right choice under a spatial gradient along the plot order; ``"random_systematic"`` deals
    random hold-out blocks of ``config.test_size`` plots drawn *without replacement* (a shuffled
    partition, one shared shuffle across the folds). Plain ``"random"`` is **not** accepted for the outer
    loop — its overlapping draws aren't a clean partition. Inside each outer fold the procedure re-runs
    on the outer-train plots only: the optimizer (if ``do_optimize``) re-tunes hyperparameters, then a
    model is fit and scored on the untouched outer-test plots. The INNER optimizer step splits the
    outer-train plots by ``inner_split_mode`` (``"sequential"`` — the default — or ``"systematic"``, so
    the whole procedure is deterministic end to end; ``"random_systematic"`` for a shuffled
    without-replacement partition; or ``"random"`` for independent noise-averaging draws) at
    ``inner_test_size`` (falling back to the outer ``config.test_size`` when ``None``), over
    ``opt_n_cv_splits`` folds. ``opt_n_jobs`` (> 1) scores each trial's inner folds in parallel threads
    (see :func:`ml.optimize.optimize_hyperparameters`); it changes speed only, never the result.

    With ``do_optimize=False`` the current ``config.params`` are used as-is in every fold ⇒ an honest CV
    of the fixed pipeline. The features are always the ones in ``config.feature_columns``.

    ``control`` (a :class:`ml.TrainControl`) Pauses/Stops: the same control is threaded into every
    inner optimizer call and checked at each outer-fold boundary. A Stop keeps only the fully-completed
    outer folds (raising :class:`TrainingStopped` if none completed), so a partial measurement is never
    biased by a half-finished fold.

    ``progress`` fires once per finished outer fold; ``inner_progress`` fires per inner trial during a
    fold's optimizer search (``(fold, n_outer, stage, trial, total)``), so a caller's live status can
    tick through the long inner work, not just at fold boundaries.
    """
    X, y, groups = make_xy(dataset, config.feature_columns)
    if len(X) < 2:
        raise ValueError("Not enough labelled rows to validate (need at least 2).")
    # ``feature_columns=None`` means "every feature column"; resolve it once here (make_xy has already
    # applied the same rule) so the fold loop always has a concrete list to record and to fit on.
    feature_columns = list(X.columns)
    had_aug = bool(np.any([aug_number_from_name(str(i)) is not None for i in X.index]))

    # Fail fast on an impossible target transform: a mid-run failure inside fold 3 would waste the
    # whole search, and the message here can name the offending rows.
    if config.target_transform == LOG:
        forward(y)  # raises with a row count if any target is <= -1
        if note is not None:
            note(describe(config.target_transform, config.target_bias_correction))

    base_seed = config.seed if seed is None else seed
    # The run's seed is also every model's random_state, so an explicit ``seed`` override must win there too.
    config = replace(config, seed=base_seed)
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise ValueError(f"Need at least 2 distinct plot groups to validate; found {n_groups}.")

    n_outer = max(1, min(n_outer_splits, n_groups))
    # All three outer modes are ratio-driven without-replacement partitions: a fold holds out
    # ``round(test_size · n_groups)`` plots, with the per-fold block shrunk to ``⌊n_groups / n_outer⌋``
    # when the rounded ratio would overrun, so the requested fold count is honoured (both ratio AND count
    # respected) and no plot is held out twice. Clamp the fold count to the continuous splits·ratio ≤ 1
    # cap (the GUI normally already guarantees it; this keeps the engine correct if called directly). A
    # single outer fold is allowed: it's one honest held-out block (the procedure never sees it), just
    # not a multi-fold rotation — the only hard constraint is splits·ratio ≤ 1.
    n_outer = max(1, min(n_outer, max_splits_for_test_size(n_groups, config.test_size)))
    inner_seeds = _outer_cycle_seeds(base_seed, n_outer)

    index = X.index.to_numpy()
    splits: list[SplitModel] = []
    fold_features: list[list[str]] = []
    fold_params: list[dict] = []
    best_split = 1
    best_loss = float("inf")

    for i in range(n_outer):
        if control is not None:
            control.wait_if_paused()
            if control.stopped:
                break

        # All three outer modes are leakage-free partitions (every plot held out exactly once).
        # Sequential/Systematic are driven by the count alone; Random Systematic additionally needs the
        # block-size ratio (``config.test_size``) and one shared shuffle seed (``base_seed``, NOT
        # ``base_seed + i``) so the per-cycle random blocks dovetail into a single partition.
        if outer_split_mode == "random_systematic":
            tr_full, te_full = random_systematic_group_split_by_count(
                groups, n_outer, config.test_size, base_seed, cycle=i
            )
        elif outer_split_mode == "systematic":
            tr_full, te_full = systematic_group_split_by_count(
                groups, n_outer, config.test_size, cycle=i
            )
        else:
            tr_full, te_full = sequential_group_split_by_count(
                groups, n_outer, config.test_size, cycle=i
            )
        train_plots = {plot_number_from_name(str(idx)) for idx in index[tr_full]}
        sub = _sub_dataset(dataset, train_plots)
        if len(np.unique(make_xy(sub, feature_columns)[2])) < 2:
            raise ValueError(
                f"Outer fold {i + 1} left fewer than 2 plot groups to train on — the test-split "
                "ratio is too large for nested validation on this dataset."
            )

        if note is not None:
            note(f"Outer fold {i + 1}/{n_outer}: {len(train_plots)} train plot(s), "
                 f"{n_groups - len(train_plots)} held out.")

        fold_feats = list(feature_columns)

        # ---- hyperparameters (optimize on outer-train only) ----
        fold_config = replace(config, feature_columns=fold_feats,
                              normalize_columns=[c for c in (config.normalize_columns or [])
                                                 if c in fold_feats])
        if do_optimize:
            # The inner CV gets its own ratio (falling back to the outer one) and mode.
            opt_config = replace(
                fold_config,
                test_size=(config.test_size if inner_test_size is None else inner_test_size),
            )
            opt_progress = (
                (lambda t, tot, _v, _i=i: inner_progress(_i + 1, n_outer, "optimize", t, tot))
                if inner_progress is not None else None
            )
            best_params = optimize_hyperparameters(
                sub, opt_config, opt_trials, n_cv_splits=opt_n_cv_splits,
                seed=inner_seeds[i], split_mode=inner_split_mode,
                fit_on_augmented=opt_fit_on_augmented,
                validate_on_augmented=opt_validate_on_augmented,
                n_jobs=opt_n_jobs, progress=opt_progress, control=control,
            )
            fold_config = replace(fold_config, params=best_params)
        else:
            fold_config = replace(fold_config, params=dict(config.params))

        # ---- fixed fit + score on this outer fold ----
        sm = _fit_score_fold(dataset, fold_config, fold_feats, tr_full, te_full, split_index=i + 1)
        splits.append(sm)
        fold_features.append(fold_feats)
        fold_params.append(dict(fold_config.params))
        loss = sm.metrics.get("rrmse", float("inf"))
        if loss < best_loss:
            best_loss, best_split = loss, i + 1

        if progress is not None:
            progress(i + 1, n_outer, loss, sm.train_metrics.get("rrmse", float("nan")),
                     sm.train_metrics, sm.metrics)

    if not splits:
        raise TrainingStopped("Validation was stopped before any outer fold finished.")

    best = splits[best_split - 1]
    history = TrainHistory(
        cycles=[CycleResult(sm.split, sm.train_metrics.get("rrmse", float("nan")),
                            sm.metrics.get("rrmse", float("nan")),
                            sm.metrics.get("r2", float("nan"))) for sm in splits],
        best_split=best_split,
        final_metrics=best.metrics,
        predictions=best.predictions,
        model=best.model,
        target_column=dataset.target_column,
        n_train=best.n_train,
        n_test=best.n_test,
        feature_columns=fold_features[best_split - 1],
        best_train_plots=best.train_plots,
        best_test_plots=best.test_plots,
        splits=splits,
        active_split=best_split,
        train_on_augmented=config.train_on_augmented,
        validate_on_augmented=config.validate_on_augmented,
        had_augmented=had_aug,
        target_transform=config.target_transform,
        target_bias_correction=config.target_bias_correction,
    )

    if note is not None:
        arr = np.asarray([sm.metrics.get("rrmse", float("nan")) for sm in splits], dtype=float)
        mean = float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")
        std = float(np.nanstd(arr)) if np.isfinite(arr).any() else float("nan")
        note(f"Validated {len(splits)} of {n_outer} outer fold(s): "
             f"held-out rRMSE = {mean:.4g}% ± {std:.3g}% (honest nested-CV estimate). "
             "Use Save to ship it (this number is its honest label).")

    return VaultResult(
        history=history,
        fold_features=fold_features,
        fold_params=fold_params,
        did_optimize=do_optimize,
        target_column=dataset.target_column,
        n_outer=n_outer,
        model_key=config.model_key,
    )


# Called once per finished model of a sweep with (model_index, n_models, model_key, result_or_None).
# ``None`` marks a model that failed (its error is reported through ``note``), so a caller can show
# the sweep advancing without treating a single bad model as fatal.
SweepCallback = Callable[[int, int, str, "VaultResult | None"], None]


def sweep_models(
    dataset: Dataset,
    config: TrainConfig,
    *,
    model_keys: list[str] | None = None,
    note: Callable[[str], None] | None = None,
    on_model: SweepCallback | None = None,
    control=None,
    **kwargs,
) -> list[VaultResult]:
    """Run :func:`validate_procedure` once per model and return the results, best first.

    This is the "try everything and keep the best" path: every model in ``model_keys`` (the whole
    registry by default, in display order) is validated under the *same* outer/inner splits, the same
    features and the same augmented-data toggles, so the numbers are directly comparable. Each model
    starts from its own registry defaults (``config.params`` describes only the model the user had
    selected, so carrying it across models would be meaningless); with ``do_optimize=True`` those
    defaults are then re-tuned inside every fold, which is what makes the comparison fair — each model
    is judged at its own best rather than at whatever happened to be typed into the form.

    Results are sorted by mean held-out rRMSE ascending, so ``result[0]`` is the winner; models whose
    run raised are reported through ``note`` and left out. A Stop (via ``control``) ends the sweep at
    the current model, keeping whatever models already finished — so a long sweep is always
    interruptible with a usable answer. Raises :class:`TrainingStopped` only if no model finished.
    """
    keys = list(model_keys) if model_keys is not None else [m.key for m in MODELS]
    results: list[VaultResult] = []
    for i, key in enumerate(keys):
        if control is not None:
            control.wait_if_paused()
            if control.stopped:
                break
        model_def = MODELS_BY_KEY.get(key)
        if model_def is None:
            continue
        if note is not None:
            note(f"Sweep {i + 1}/{len(keys)}: {model_def.label}…")
        # Each model is scored from its own defaults — the form's params belong to one model only.
        model_config = replace(config, model_key=key, params=model_def.defaults())
        try:
            result = validate_procedure(
                dataset, model_config, note=note, control=control, **kwargs
            )
        except TrainingStopped:
            break  # a Stop landed inside this model's folds; keep the models already finished
        except Exception as exc:  # noqa: BLE001 - one bad model must not sink the sweep
            if note is not None:
                note(f"  {model_def.label} failed: {exc}")
            if on_model is not None:
                on_model(i + 1, len(keys), key, None)
            continue
        results.append(result)
        if on_model is not None:
            on_model(i + 1, len(keys), key, result)

    if not results:
        raise TrainingStopped("The model sweep finished no model — nothing to report.")
    # NaN means the model produced no usable held-out score; sort those last rather than first.
    results.sort(key=lambda r: (np.isnan(r.mean_held_out_rrmse), r.mean_held_out_rrmse))
    return results

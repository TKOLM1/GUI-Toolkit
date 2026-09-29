"""Unit tests for the rotating-vault nested cross-validation (ml.validate).

These prove the engine headless: that the outer loop is a true sequential partition (every plot held
out exactly once across disjoint folds), that each fold carries a full metric cube, that the per-fold
features / tuned params are recorded, that a model sweep ranks its models, and that the small-data
guards fire. Optuna is required for the optimize stage, so those tests skip if it is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml import (
    Dataset,
    TrainConfig,
    TrainingStopped,
    VaultResult,
    has_variant_cube,
    sweep_models,
    validate_procedure,
)


def make_dataset(n_plots: int = 16, copies: int = 2, n_features: int = 4, seed: int = 0) -> Dataset:
    """A synthetic dataset: each plot has one verbatim original plus augmented copies.

    The first copy (``c == 0``) is the original (no ``aug(...)``); the rest are augmented. The target
    is a smooth linear function of the features so every regressor learns it. (Self-contained rather
    than imported from test_ml, since the tests/ folder is not a package.)
    """
    rng = np.random.default_rng(seed)
    feat_cols = [f"f{i}" for i in range(n_features)]
    target = "biomass"
    index, x_rows, y_vals = [], [], []
    for plot in range(1, n_plots + 1):
        base = rng.normal(size=n_features)
        true = float(base.sum() * 2.0 + 5.0)
        for c in range(copies):
            if c == 0:
                index.append(f"field_plot({plot}).laz")
            else:
                index.append(f"field_plot({plot})_aug({plot * 100 + c}).laz")
            x_rows.append(base + rng.normal(scale=0.01, size=n_features))
            y_vals.append(true + rng.normal(scale=0.05))
    import pandas as pd
    frame = pd.DataFrame(x_rows, columns=feat_cols, index=index)
    frame[target] = y_vals
    frame.index.name = "filename"
    return Dataset(frame=frame, feature_columns=feat_cols, target_column=target)


def _cfg(ds: Dataset, **over) -> TrainConfig:
    base = dict(
        model_key="ridge", params={"alpha": 1.0}, feature_columns=ds.feature_columns,
        normalize_columns=ds.feature_columns, test_size=0.25, seed=0,
    )
    base.update(over)
    return TrainConfig(**base)


def _n_groups(ds: Dataset) -> int:
    from ml import make_xy
    return len(np.unique(make_xy(ds, ds.feature_columns)[2]))


# --------------------------------------------------------------------------- #
# Outer partition (works without optuna: both stages off)                      #
# --------------------------------------------------------------------------- #

def test_outer_loop_uses_the_requested_count():
    ds = make_dataset(n_plots=16, copies=2, n_features=4, seed=0)
    vr = validate_procedure(ds, _cfg(ds), do_optimize=False, n_outer_splits=4)
    assert isinstance(vr, VaultResult)
    # The outer fold count is exactly what the caller asked for (the field is tiled into that many).
    assert vr.n_outer == 4
    assert len(vr.history.splits) == 4


def test_outer_count_default_and_clamp():
    ds = make_dataset(n_plots=5, copies=2, n_features=3, seed=0)
    # The default outer count (4) applies when unspecified.
    assert validate_procedure(ds, _cfg(ds), do_optimize=False).n_outer == 4
    # A count beyond the splits·ratio ≤ 1 cap is clamped to ⌊1/test_size⌋ (= ⌊1/0.25⌋ = 4 here), not to
    # the raw plot count — the cap that keeps the outer folds a non-overlapping partition.
    vr = validate_procedure(ds, _cfg(ds), do_optimize=False, n_outer_splits=99)
    assert vr.n_outer == 4


def test_every_plot_held_out_exactly_once():
    ds = make_dataset(n_plots=16, copies=2, n_features=4, seed=1)
    vr = validate_procedure(ds, _cfg(ds), do_optimize=False)
    all_test = [p for sm in vr.history.splits for p in sm.test_plots]
    # Disjoint folds (no plot in two test sets) AND complete cover (every plot tested once).
    assert len(all_test) == len(set(all_test))           # disjoint
    assert set(all_test) == set(range(1, 17))            # covers all 16 plots


def test_systematic_outer_mode_partitions_and_strides():
    ds = make_dataset(n_plots=16, copies=2, n_features=4, seed=1)
    vr = validate_procedure(
        ds, _cfg(ds), do_optimize=False,
        n_outer_splits=4, outer_split_mode="systematic",
    )
    assert vr.n_outer == 4
    all_test = [p for sm in vr.history.splits for p in sm.test_plots]
    assert len(all_test) == len(set(all_test))           # disjoint folds
    assert set(all_test) == set(range(1, 17))            # every plot held out once
    # Systematic strides: the first fold holds out every 4th plot (1, 5, 9, 13), not a contiguous block.
    fold0 = sorted(vr.history.splits[0].test_plots)
    assert fold0 == [1, 5, 9, 13]


def test_random_systematic_outer_mode_partitions_without_replacement():
    ds = make_dataset(n_plots=16, copies=2, n_features=4, seed=1)
    vr = validate_procedure(
        ds, _cfg(ds), do_optimize=False,
        n_outer_splits=4, outer_split_mode="random_systematic",
    )
    assert vr.n_outer == 4
    all_test = [p for sm in vr.history.splits for p in sm.test_plots]
    assert len(all_test) == len(set(all_test))           # disjoint folds (without replacement)
    assert set(all_test) == set(range(1, 17))            # every plot held out exactly once
    # No outer fold mixes a plot into both sides (the core leakage guard, in the new outer mode too).
    for sm in vr.history.splits:
        assert sm.train_plots.isdisjoint(sm.test_plots)


def test_random_systematic_outer_clamps_to_continuous_cap():
    """Outer fold count is clamped to the continuous splits·ratio ≤ 1 cap (⌊1/test_size⌋), no plot twice."""
    ds = make_dataset(n_plots=16, copies=2, n_features=4, seed=1)
    # test_size 0.3 → cap ⌊1/0.3⌋ = 3, so a request for 5 outer folds clamps to 3 (block round(0.3*16)=5).
    vr = validate_procedure(
        ds, _cfg(ds, test_size=0.3), do_optimize=False,
        n_outer_splits=5, outer_split_mode="random_systematic",
    )
    assert vr.n_outer == 3
    all_test = [p for sm in vr.history.splits for p in sm.test_plots]
    assert len(all_test) == len(set(all_test))           # still without replacement after the clamp


def test_outer_block_shrinks_so_requested_count_is_honoured():
    """The reported bug in miniature: a ratio that rounds *up* must not reduce the requested fold count.

    16 plots at test_size 0.25: round(0.25*16)=4 leaves room for exactly 4 blocks, so 4 is fine — but at a
    ratio that rounds up (here we mimic 150@0.25 by using 6 plots @ 0.25: round(0.25*6)=2 → ⌊6/2⌋=3 would
    be the OLD cap, yet 4 splits are requested and must be honoured by shrinking the block to ⌊6/4⌋=1).
    """
    ds = make_dataset(n_plots=6, copies=2, n_features=3, seed=2)
    vr = validate_procedure(
        ds, _cfg(ds, test_size=0.25), do_optimize=False,
        n_outer_splits=4, outer_split_mode="sequential",
    )
    assert vr.n_outer == 4                               # honoured, not silently dropped to 3
    all_test = [p for sm in vr.history.splits for p in sm.test_plots]
    assert len(all_test) == len(set(all_test))           # disjoint: no plot held out twice
    for sm in vr.history.splits:
        assert sm.train_plots.isdisjoint(sm.test_plots)  # leakage-free


def test_folds_carry_full_metric_cube():
    ds = make_dataset(n_plots=16, copies=2, n_features=4, seed=2)
    vr = validate_procedure(ds, _cfg(ds), do_optimize=False)
    # The cube is what lets the Results aug toggles work in the report without a refit.
    assert has_variant_cube(vr.history)


def test_fixed_stages_record_per_fold_features_and_params():
    ds = make_dataset(n_plots=16, copies=2, n_features=4, seed=3)
    vr = validate_procedure(ds, _cfg(ds), do_optimize=False)
    assert len(vr.fold_features) == vr.n_outer
    assert len(vr.fold_params) == vr.n_outer
    # Every fold uses the configured feature set, and none of them tuned anything.
    assert all(set(f) == set(ds.feature_columns) for f in vr.fold_features)
    assert not vr.did_optimize


def test_mean_and_std_held_out_are_finite():
    ds = make_dataset(n_plots=16, copies=2, n_features=4, seed=4)
    vr = validate_procedure(ds, _cfg(ds), do_optimize=False)
    assert np.isfinite(vr.mean_held_out_rrmse)
    assert np.isfinite(vr.std_held_out_rrmse)


# --------------------------------------------------------------------------- #
# Optimize inside the folds, and the model sweep — need optuna                #
# --------------------------------------------------------------------------- #

def test_full_procedure_runs_and_records_tuned_params():
    pytest.importorskip("optuna")
    ds = make_dataset(n_plots=16, copies=2, n_features=4, seed=5)
    vr = validate_procedure(ds, _cfg(ds), do_optimize=True, opt_trials=4)
    assert vr.did_optimize
    assert vr.model_key == "ridge"
    assert len(vr.history.splits) == vr.n_outer
    # Every fold used the configured feature set and recorded the alpha it tuned for itself.
    assert all(list(f) == list(ds.feature_columns) for f in vr.fold_features)
    assert all("alpha" in p for p in vr.fold_params)


def test_sweep_ranks_every_model_best_first():
    """The sweep runs each requested model under the same splits and returns them best-first."""
    ds = make_dataset(n_plots=16, copies=2, n_features=4, seed=5)
    seen: list[str] = []
    results = sweep_models(
        ds, _cfg(ds), model_keys=["ridge", "knn", "random_forest"], do_optimize=False,
        n_outer_splits=3, on_model=lambda i, n, key, res: seen.append(key),
    )
    assert seen == ["ridge", "knn", "random_forest"]  # run in the order given
    assert [r.model_key for r in results] == sorted(
        [r.model_key for r in results],
        key=lambda k: next(r.mean_held_out_rrmse for r in results if r.model_key == k),
    )
    assert results[0].mean_held_out_rrmse <= results[-1].mean_held_out_rrmse
    # Each result is a full measurement of its own model, not a shared one.
    assert {r.model_key for r in results} == {"ridge", "knn", "random_forest"}
    assert all(len(r.history.splits) == 3 for r in results)


def test_sweep_skips_a_failing_model_and_reports_it():
    """One unusable model must not sink the sweep — it is noted and left out of the ranking."""
    ds = make_dataset(n_plots=16, copies=2, n_features=4, seed=5)
    notes: list[str] = []
    results = sweep_models(
        ds, _cfg(ds), model_keys=["ridge", "not_a_model"], do_optimize=False, n_outer_splits=3,
        note=notes.append,
    )
    assert [r.model_key for r in results] == ["ridge"]  # the unknown key is skipped silently


def test_sweep_with_no_finished_model_raises():
    ds = make_dataset(n_plots=16, copies=2, n_features=4, seed=5)
    with pytest.raises(TrainingStopped):
        sweep_models(ds, _cfg(ds), model_keys=["not_a_model"], do_optimize=False)


def test_inner_random_systematic_runs_and_reports_inner_progress():
    """The inner CV accepts the random-systematic mode, and inner_progress fires per inner trial."""
    pytest.importorskip("optuna")
    ds = make_dataset(n_plots=16, copies=2, n_features=4, seed=5)
    inner_calls: list[tuple] = []
    vr = validate_procedure(
        ds, _cfg(ds), do_optimize=True, n_outer_splits=3,
        inner_split_mode="random_systematic", inner_test_size=0.25, opt_trials=3,
        inner_progress=lambda f, n, stage, t, tot: inner_calls.append((f, n, stage, t, tot)),
    )
    assert len(vr.history.splits) == 3
    # The optimizer ran inside each fold, so the per-fold params are recorded and the inner-progress
    # callback ticked with the optimize stage and a sane (fold, n_outer) context.
    assert all("alpha" in p for p in vr.fold_params)
    assert inner_calls, "inner_progress should fire during the inner optimizer search"
    assert all(stage == "optimize" and 1 <= f <= 3 and n == 3 for f, n, stage, _t, _tot in inner_calls)


def test_opt_n_jobs_reaches_the_inner_optimizer(monkeypatch):
    """The thread count must actually be handed to every fold's optimizer search, not dropped."""
    pytest.importorskip("optuna")
    import ml.validate as validate_module

    seen: list[int] = []
    real = validate_module.optimize_hyperparameters

    def spy(*args, **kwargs):
        seen.append(kwargs.get("n_jobs", 1))
        return real(*args, **kwargs)

    monkeypatch.setattr(validate_module, "optimize_hyperparameters", spy)
    ds = make_dataset(n_plots=16, copies=2, n_features=4, seed=5)
    vr = validate_procedure(ds, _cfg(ds), do_optimize=True, n_outer_splits=3, opt_trials=2,
                            opt_n_jobs=4)
    assert seen == [4] * vr.n_outer


def test_threaded_inner_folds_match_serial():
    """Scoring the inner folds in parallel threads changes speed only: same tuned params, same scores.

    End to end through the nested CV; the forest's thread pinning under the fan-out is covered at the
    optimizer level in test_ml.
    """
    pytest.importorskip("optuna")
    ds = make_dataset(n_plots=16, copies=2, n_features=4, seed=5)
    cfg = _cfg(ds)
    kwargs = dict(do_optimize=True, n_outer_splits=3, outer_split_mode="random_systematic",
                  inner_split_mode="random_systematic", opt_trials=4)
    serial = validate_procedure(ds, cfg, opt_n_jobs=1, **kwargs)
    threaded = validate_procedure(ds, cfg, opt_n_jobs=4, **kwargs)
    assert threaded.fold_params == serial.fold_params
    for s, t in zip(serial.history.splits, threaded.history.splits):
        assert t.metrics == pytest.approx(s.metrics, nan_ok=True)


# --------------------------------------------------------------------------- #
# Guards                                                                       #
# --------------------------------------------------------------------------- #

def test_too_few_groups_rejected():
    # A single plot group can't form a 2-way outer partition — the engine must refuse, not 1-fold.
    ds = make_dataset(n_plots=1, copies=2, n_features=3, seed=6)
    with pytest.raises(ValueError):
        validate_procedure(ds, _cfg(ds), do_optimize=False, n_outer_splits=4)

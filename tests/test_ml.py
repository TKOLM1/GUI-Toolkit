"""Unit tests for the classical-ML core (dataset, splitting, models, the run and the optimizer)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from ml import (
    MODELS,
    MODELS_BY_KEY,
    Dataset,
    TrainConfig,
    grouped_split,
    has_variant_cube,
    load_dataset,
    make_xy,
    optimize_hyperparameters,
    training_split_stats,
    validate_procedure,
    variant_metrics,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def make_dataset(n_plots: int = 8, copies: int = 3, n_features: int = 4, seed: int = 0) -> Dataset:
    """A synthetic dataset: each plot has one verbatim original plus augmented copies.

    The first copy (``c == 0``) is the original (no ``aug(...)``); the rest are augmented,
    mirroring how augmentation writes a verbatim copy alongside its synthetic copies. The
    target is a smooth function of the features so every regressor can learn it.
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
                index.append(f"field_plot({plot}).laz")                       # verbatim original
            else:
                index.append(f"field_plot({plot})_aug({plot * 100 + c}).laz")  # augmented copy
            x_rows.append(base + rng.normal(scale=0.01, size=n_features))
            y_vals.append(true + rng.normal(scale=0.05))
    frame = pd.DataFrame(x_rows, columns=feat_cols, index=index)
    frame[target] = y_vals
    frame.index.name = "filename"
    return Dataset(frame=frame, feature_columns=feat_cols, target_column=target)


# --------------------------------------------------------------------------- #
# Splitting / leakage                                                         #
# --------------------------------------------------------------------------- #

def test_grouped_split_keeps_plots_on_one_side():
    groups = np.array([f"plot:{p}" for p in [1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6]])
    train_idx, test_idx = grouped_split(groups, test_size=0.34, seed=0)
    train_plots = set(groups[train_idx])
    test_plots = set(groups[test_idx])
    assert train_plots.isdisjoint(test_plots)          # no plot leaks across the split
    assert train_plots and test_plots                  # both sides non-empty


def test_grouped_split_needs_two_groups():
    groups = np.array(["plot:1", "plot:1", "plot:1"])
    with pytest.raises(ValueError):
        grouped_split(groups, test_size=0.25, seed=0)


def test_sequential_split_is_contiguous_and_leakage_free():
    groups = np.array([f"plot:{p}" for p in [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]])
    seen_blocks = []
    for cycle in range(3):
        tr, te = grouped_split(groups, test_size=0.34, seed=0, mode="sequential", cycle=cycle)
        train_plots, test_plots = set(groups[tr]), set(groups[te])
        assert train_plots.isdisjoint(test_plots)            # leakage-free
        assert train_plots and test_plots
        # The held-out groups are a contiguous block in sorted plot-number order.
        nums = sorted(int(g.split(":")[1]) for g in test_plots)
        assert nums == list(range(nums[0], nums[0] + len(nums)))
        seen_blocks.append(tuple(nums))
    # Successive cycles slide to different blocks, sweeping the field.
    assert len(set(seen_blocks)) > 1


def test_n_sequential_splits_tiles_field_from_ratio():
    from ml.splitting import n_sequential_splits

    # ceil(1/ratio): the contiguous blocks tile the whole field, every plot held out once.
    assert n_sequential_splits(80, 0.25) == 4
    assert n_sequential_splits(80, 0.20) == 5
    assert n_sequential_splits(80, 0.34) == 3
    assert n_sequential_splits(1, 0.25) == 1  # degenerate: too few groups


def test_grouped_cv_folds_consistency_across_modes():
    from ml.optimize import grouped_cv_folds

    groups = np.array([f"plot:{p}" for p in range(1, 21) for _ in range(2)])
    # Random: exactly the requested number of leakage-free folds.
    rand = grouped_cv_folds(groups, n_splits=5, split_mode="random", test_size=0.25, seed=0)
    assert len(rand) == 5
    for tr, te in rand:
        assert set(groups[tr]).isdisjoint(set(groups[te]))
    # Sequential: ratio-driven disjoint blocks of round(0.2*20)=4 plots; splits·ratio = 5*0.2 = 1, so
    # the 5 blocks tile the field exactly (every plot held out once).
    seq = grouped_cv_folds(groups, n_splits=5, split_mode="sequential", test_size=0.2, seed=0)
    assert len(seq) == 5
    held = set()
    for tr, te in seq:
        tp = set(groups[te])
        assert len(tp) == 4                         # 4 plots per block (the ratio-sized block)
        assert tp.isdisjoint(held)                 # non-overlapping partition
        assert set(groups[tr]).isdisjoint(tp)      # leakage-free
        held |= tp
    assert held == set(groups)                     # every plot tested exactly once (product == 1)
    # Below product 1 the blocks no longer cover the field: 3 blocks of 4 plots leave 8 plots untested.
    partial = grouped_cv_folds(groups, n_splits=3, split_mode="sequential", test_size=0.2, seed=0)
    assert len(partial) == 3
    held_partial = set()
    for tr, te in partial:
        tp = set(groups[te])
        assert tp.isdisjoint(held_partial)         # still without replacement
        assert set(groups[tr]).isdisjoint(tp)      # leakage-free
        held_partial |= tp
    assert len(held_partial) == 3 * 4              # only 12 of 20 plots held out


def test_sequential_group_split_by_count_blocks_by_ratio():
    from ml.splitting import sequential_group_split_by_count

    # 20 plots, ratio 0.25 -> blocks of round(0.25*20) = 5 contiguous plots; 4 blocks tile the field
    # (4*5 = 20), every plot held out once.
    groups = np.array([f"plot:{p}" for p in range(1, 21)])
    held = set()
    sizes = []
    for cycle in range(4):
        tr, te = sequential_group_split_by_count(groups, n_splits=4, test_size=0.25, cycle=cycle)
        tp = set(groups[te])
        assert tp.isdisjoint(held)                       # every plot held out at most once
        assert set(groups[tr]).isdisjoint(tp)            # leakage-free
        nums = sorted(int(g.split(":")[1]) for g in tp)
        assert nums == list(range(nums[0], nums[0] + len(nums)))  # contiguous block
        sizes.append(len(tp))
        held |= tp
    assert sizes == [5, 5, 5, 5]                          # equal ratio-sized blocks
    assert held == set(groups)                            # product == 1 → every plot held out once

    # Asking for 6 blocks at ratio 0.25 (splits·ratio = 1.5 > 1): the block shrinks to ⌊20/6⌋ = 3 plots
    # so all 6 disjoint blocks tile the field — both ratio and count honoured, no plot mixed into both
    # sides of one split.
    held6 = set()
    for cycle in range(6):
        tr6, te6 = sequential_group_split_by_count(groups, n_splits=6, test_size=0.25, cycle=cycle)
        tp6 = set(groups[te6])
        assert tp6.isdisjoint(held6)                     # without replacement across all 6 blocks
        assert set(groups[tr6]).isdisjoint(tp6)          # leakage-free
        assert len(tp6) == 3                             # block shrunk round(5) → ⌊20/6⌋ = 3
        held6 |= tp6


def test_systematic_group_split_by_count_strides_and_partitions():
    from ml.splitting import systematic_group_split_by_count

    # 12 plots, ratio 1/3 -> block of round(12/3) = 4 strided plots per fold; 3 folds, stride 3, so
    # fold c takes the first 4 of positions c, c+3, c+6, c+9 -> exactly the every-3rd-plot partition.
    groups = np.array([f"plot:{p}" for p in range(1, 13)])
    held = set()
    expected = {0: {1, 4, 7, 10}, 1: {2, 5, 8, 11}, 2: {3, 6, 9, 12}}
    for cycle in range(3):
        tr, te = systematic_group_split_by_count(groups, n_splits=3, test_size=1 / 3, cycle=cycle)
        tp = set(groups[te])
        nums = {int(g.split(":")[1]) for g in tp}
        assert nums == expected[cycle]                   # every n-th plot, offset by the fold
        assert tp.isdisjoint(held)                       # held out at most once
        assert set(groups[tr]).isdisjoint(tp)            # leakage-free
        held |= tp
    assert held == set(groups)                           # product == 1 → every plot held out once


def test_systematic_group_split_by_count_partial_when_product_below_one():
    from ml.splitting import systematic_group_split_by_count

    # 20 plots, ratio 0.2 -> block of 4 strided plots; with only 3 folds (3*0.2 = 0.6 < 1) the strides
    # stay disjoint but leave plots untested.
    groups = np.array([f"plot:{p}" for p in range(1, 21)])
    held = set()
    sizes = []
    for cycle in range(3):
        tr, te = systematic_group_split_by_count(groups, n_splits=3, test_size=0.2, cycle=cycle)
        tp = set(groups[te])
        assert tp.isdisjoint(held)                       # without replacement
        assert set(groups[tr]).isdisjoint(tp)            # leakage-free
        sizes.append(len(tp))
        held |= tp
    assert sizes == [4, 4, 4]                             # equal ratio-sized strided blocks
    assert len(held) == 12                                # 3*4 plots held out, 8 never tested


def test_systematic_group_split_unaffected_by_block_order():
    """Systematic folds depend on numeric plot order, not on the row order in the array."""
    from ml.splitting import systematic_group_split_by_count

    a = np.array([f"plot:{p}" for p in [1, 2, 3, 4, 5, 6]])
    shuffled = np.array([f"plot:{p}" for p in [6, 1, 4, 2, 5, 3]])
    # ratio 0.5 -> block of 3 strided plots; 2 folds, stride 2 -> fold 0 = positions 0,2,4 = plots 1,3,5.
    tr_a, te_a = systematic_group_split_by_count(a, n_splits=2, test_size=0.5, cycle=0)
    tr_b, te_b = systematic_group_split_by_count(shuffled, n_splits=2, test_size=0.5, cycle=0)
    # Same held-out *plots* (odd positions in numeric order: plots 1, 3, 5) regardless of array order.
    assert set(a[te_a]) == {"plot:1", "plot:3", "plot:5"}
    assert set(shuffled[te_b]) == {"plot:1", "plot:3", "plot:5"}


def test_grouped_cv_folds_systematic_is_a_partition():
    from ml.optimize import grouped_cv_folds

    groups = np.array([f"plot:{p}" for p in range(1, 21) for _ in range(2)])
    # ratio 0.2 -> block of 4 strided plots; 5*0.2 = 1, so the 5 strided folds tile the field.
    folds = grouped_cv_folds(groups, n_splits=5, split_mode="systematic", test_size=0.2, seed=0)
    assert len(folds) == 5                                # 5 folds fit (product == 1)
    held = set()
    for tr, te in folds:
        tp = set(groups[te])
        assert tp.isdisjoint(held)                        # non-overlapping
        assert set(groups[tr]).isdisjoint(tp)             # leakage-free
        held |= tp
    assert held == set(groups)                            # every plot tested exactly once
    # Within the cap (5*0.2 = 1) but with a ratio that rounds up: round(0.25*20)=5 would only let 4 folds
    # fit, yet 5 are requested — the block shrinks to ⌊20/5⌋=4 so all 5 strided folds still tile, honouring
    # both the ratio and the count (5*0.25 = 1.25 is over the cap; the GUI keeps the product ≤ 1, but the
    # engine clamps the count to ⌊1/0.25⌋ = 4 if called past it).
    capped = grouped_cv_folds(groups, n_splits=5, split_mode="systematic", test_size=0.25, seed=0)
    assert len(capped) == 4                               # engine clamp: ⌊1/0.25⌋ = 4 folds


def test_random_systematic_is_a_without_replacement_partition():
    """Random-systematic shuffles once and deals disjoint random blocks across the splits."""
    from ml.splitting import random_systematic_group_split_by_count as rsg

    groups = np.array([f"plot:{p}" for p in range(1, 21) for _ in range(3)])
    n_splits, ts, seed = 4, 0.25, 11  # block = round(0.25*20) = 5; 4*5 = 20 = exact partition
    held = set()
    block_sizes = []
    for cycle in range(n_splits):
        tr, te = rsg(groups, n_splits, ts, seed, cycle=cycle)
        tp = set(groups[te])
        assert tp.isdisjoint(held)                  # held out at most once (without replacement)
        assert set(groups[tr]).isdisjoint(tp)       # leakage-free
        block_sizes.append(len({int(g.split(":")[1]) for g in tp}))
        held |= tp
    assert held == set(groups)                      # every plot held out exactly once
    assert block_sizes == [5, 5, 5, 5]              # equal random blocks


def test_random_systematic_is_reproducible_and_order_independent():
    """Same seed → same partition; the partition depends on plots, not on the array's row order."""
    from ml.splitting import random_systematic_group_split_by_count as rsg

    groups = np.array([f"plot:{p}" for p in range(1, 13)])
    a1 = set(groups[rsg(groups, 3, 0.34, 5, cycle=0)[1]])
    a2 = set(groups[rsg(groups, 3, 0.34, 5, cycle=0)[1]])
    assert a1 == a2                                 # reproducible
    shuffled = np.array([f"plot:{p}" for p in [12, 3, 7, 1, 9, 5, 11, 2, 8, 4, 10, 6]])
    assert set(shuffled[rsg(shuffled, 3, 0.34, 5, cycle=0)[1]]) == a1  # order-independent
    # A different seed shuffles differently, so block 0 generally differs.
    assert set(groups[rsg(groups, 3, 0.34, 6, cycle=0)[1]]) != a1


def test_random_systematic_block_shrinks_so_requested_count_fits():
    """The requested split count is honoured: the block shrinks so all N disjoint blocks tile the field.

    round(0.3*10)=3 would only let ⌊10/3⌋=3 blocks fit, but the user asked for 5 — so the block shrinks
    to ⌊10/5⌋=2 plots and all 5 hold-outs tile the field without overlap (both ratio and count honoured),
    rather than the count being silently clamped to 3.
    """
    from ml.splitting import random_systematic_group_split_by_count as rsg

    groups = np.array([f"plot:{p}" for p in range(1, 11)])  # 10 plots
    held = set()
    sizes = []
    for cycle in range(5):
        tr, te = rsg(groups, 5, 0.3, 0, cycle=cycle)
        tp = set(groups[te])
        assert tp.isdisjoint(held)                  # without replacement across the 5 folds
        assert set(groups[tr]).isdisjoint(tp)       # leakage-free within each fold
        sizes.append(len(tp))
        held |= tp
    assert sizes == [2, 2, 2, 2, 2]                 # block shrunk 3 → 2 so 5 blocks fit
    assert held == set(groups)                      # exact partition: every plot held out once


def test_max_splits_for_test_size_is_the_continuous_cap():
    # The cap is the continuous splits·ratio ≤ 1 (⌊1/ratio⌋), NOT ⌊n/round(ratio·n)⌋: the engine shrinks
    # each block to fit the requested count, so the count is bounded only by the ratio.
    from ml.splitting import max_splits_for_test_size

    assert max_splits_for_test_size(80, 0.25) == 4   # ⌊1/0.25⌋ = 4
    assert max_splits_for_test_size(80, 0.20) == 5   # ⌊1/0.20⌋ = 5
    assert max_splits_for_test_size(80, 0.30) == 3   # ⌊1/0.30⌋ = 3
    # The reported bug: 150 plots at 0.25 — round(0.25·150)=38 would (wrongly) cap the old rule at
    # ⌊150/38⌋=3, but the continuous cap correctly allows the requested 4 (the block shrinks to 37).
    assert max_splits_for_test_size(150, 0.25) == 4
    assert max_splits_for_test_size(0, 0.25) == 1    # degenerate guard (no groups → never below 1)
    assert max_splits_for_test_size(10, 0.0) == 10   # zero ratio → unbounded (falls back to n_groups)


@pytest.mark.parametrize("mode", ["random", "random_systematic", "sequential", "systematic"])
@pytest.mark.parametrize("n_splits,test_size", [(1, 0.33), (2, 0.5), (3, 0.2), (4, 0.25)])
def test_grouped_cv_folds_never_leak_a_plot_across_a_fold(mode, n_splits, test_size):
    """Every fold (every mode, incl. the now-allowed single split) keeps a plot wholly on one side.

    This is THE leakage guard: a plot's rows must never appear in both the train and the validation of
    the same fold. It must hold for any count/ratio the GUI now permits (splits·ratio ≤ 1 for the
    partition modes, unbounded for random), including a single split.
    """
    from ml.optimize import grouped_cv_folds

    groups = np.array([f"plot:{p}" for p in range(1, 21) for _ in range(3)])  # 20 plots, 3 rows each
    folds = grouped_cv_folds(groups, n_splits=n_splits, split_mode=mode,
                             test_size=test_size, seed=7)
    assert len(folds) >= 1
    for tr, te in folds:
        tr_plots, te_plots = set(groups[tr]), set(groups[te])
        assert tr_plots.isdisjoint(te_plots)              # no plot on both sides of one fold
        assert len(te) > 0 and len(tr) > 0                # both sides non-empty


@pytest.mark.parametrize("mode", ["random_systematic", "sequential", "systematic"])
def test_grouped_cv_folds_partition_modes_never_hold_a_plot_out_twice(mode):
    """The three partition modes are without replacement: no plot is held out in two folds at once."""
    from ml.optimize import grouped_cv_folds

    groups = np.array([f"plot:{p}" for p in range(1, 21) for _ in range(2)])
    folds = grouped_cv_folds(groups, n_splits=4, split_mode=mode, test_size=0.25, seed=1)
    held = set()
    for _tr, te in folds:
        tp = set(groups[te])
        assert tp.isdisjoint(held)                        # held out at most once across the folds
        held |= tp


def test_grouped_cv_folds_random_systematic_is_a_partition():
    from ml.optimize import grouped_cv_folds

    groups = np.array([f"plot:{p}" for p in range(1, 21) for _ in range(2)])
    folds = grouped_cv_folds(groups, n_splits=4, split_mode="random_systematic",
                             test_size=0.25, seed=3)
    assert len(folds) == 4                                # exactly n_splits folds (cap satisfied)
    held = set()
    for tr, te in folds:
        tp = set(groups[te])
        assert tp.isdisjoint(held)                        # non-overlapping (without replacement)
        assert set(groups[tr]).isdisjoint(tp)             # leakage-free
        held |= tp
    assert held == set(groups)                            # every plot held out exactly once


def test_grouped_cv_folds_random_systematic_clamps_past_cap():
    """grouped_cv_folds never returns duplicate blocks past the splits·ratio cap."""
    from ml.optimize import grouped_cv_folds

    groups = np.array([f"plot:{p}" for p in range(1, 11)])  # block = round(0.3*10) = 3 → ⌊10/3⌋ = 3
    folds = grouped_cv_folds(groups, n_splits=5, split_mode="random_systematic",
                             test_size=0.3, seed=0)
    assert len(folds) == 3                                  # clamped to the 3 disjoint blocks
    held = set()
    for tr, te in folds:
        tp = set(groups[te])
        assert tp.isdisjoint(held)
        held |= tp


def test_search_space_text_reflects_hparams():
    from ml.optimize import classical_search_space_text

    text = classical_search_space_text("ridge")
    assert "Alpha" in text and "log scale" in text  # ridge alpha is a wide log-scale float
    # PLS n_components is clamped to the feature count when known.
    pls = classical_search_space_text("pls", n_features=4)
    assert "4" in pls


def test_make_xy_groups_by_plot_and_drops_missing_target():
    ds = make_dataset(n_plots=3, copies=2)
    # Blank one target value -> that row should be dropped.
    ds.frame.iloc[0, ds.frame.columns.get_loc("biomass")] = np.nan
    X, y, groups = make_xy(ds)
    assert len(X) == len(y) == len(groups) == 5         # 6 rows - 1 dropped
    assert all(g.startswith("plot:") for g in groups)


# --------------------------------------------------------------------------- #
# Dataset loading from two workbooks                                          #
# --------------------------------------------------------------------------- #

def test_load_dataset_joins_features_and_targets(tmp_path):
    idx = ["field_plot(1).laz", "field_plot(2).laz"]
    feats = pd.DataFrame({"f0": [1.0, 2.0], "f1": [3.0, 4.0]}, index=idx)
    feats.index.name = "filename"
    tgts = pd.DataFrame({"biomass": [10.0, 20.0]}, index=idx)
    tgts.index.name = "filename"
    fpath, tpath = tmp_path / "features.csv", tmp_path / "target.csv"
    feats.to_csv(fpath)
    tgts.to_csv(tpath)

    ds = load_dataset(fpath, tpath)
    assert ds.feature_columns == ["f0", "f1"]
    assert ds.target_column == "biomass"
    assert ds.frame.loc["field_plot(2).laz", "biomass"] == pytest.approx(20.0)


# --------------------------------------------------------------------------- #
# The run - every model fits and reports per-fold losses                      #
# --------------------------------------------------------------------------- #

def _run(ds, cfg, n_outer: int = 3, **kwargs):
    """The history a run produces: nested CV with fixed hyperparameters.

    :func:`ml.validate.validate_procedure` is the only thing that builds a model now, so these tests
    drive it directly with ``do_optimize=False`` — the fold loop, the augmented-data toggles and the
    metric cube are all exercised without paying for an Optuna search.
    """
    return validate_procedure(ds, cfg, do_optimize=False, n_outer_splits=n_outer, **kwargs).history


@pytest.mark.parametrize("mode", ["random_systematic", "sequential", "systematic"])
def test_run_produces_a_single_leakage_free_fold_in_every_mode(mode):
    """A single outer fold is allowed in every partition mode and produces one leakage-free split."""
    ds = make_dataset(n_plots=8, copies=3, n_features=4, seed=2)
    history = _run(
        ds, TrainConfig(model_key="ridge", test_size=0.25, seed=0),
        n_outer=1, outer_split_mode=mode,
    )
    assert len(history.cycles) == len(history.splits) == 1
    sm = history.splits[0]
    assert sm.train_plots.isdisjoint(sm.test_plots)   # no plot leaks across the one split
    assert sm.test_plots and sm.train_plots           # both sides populated


@pytest.mark.parametrize("model_def", MODELS, ids=[m.key for m in MODELS])
def test_every_model_runs_and_reports_losses(model_def):
    ds = make_dataset(n_plots=8, copies=3, n_features=4, seed=1)
    seen = []
    history = validate_procedure(
        ds,
        TrainConfig(model_key=model_def.key, seed=0),
        do_optimize=False,
        n_outer_splits=3,
        progress=lambda f, total, ho, tr, trm, tem: seen.append((f, tr, ho, trm, tem)),
    ).history
    assert len(history.cycles) == 3
    assert len(seen) == 3
    # The progress callback also receives the fold's full train/held-out metric dicts.
    for *_, trm, tem in seen:
        assert {"rrmse", "r2", "r", "mape"} <= trm.keys()
        assert {"rrmse", "r2", "r", "mape"} <= tem.keys()
    # Every fold is retained as a selectable model (not just the best one).
    assert len(history.splits) == 3
    assert history.active_split == history.best_split
    # The default model comes from the fold with the lowest held-out rRMSE.
    best = min(history.cycles, key=lambda r: r.val_loss)
    assert history.best_split == best.cycle
    assert history.final_metrics["rrmse"] == pytest.approx(best.val_loss)
    # R² is reported per fold (in the cycle and in the SplitModel).
    assert all(math.isfinite(c.val_r2) for c in history.cycles)
    assert all(math.isfinite(s.metrics["r2"]) for s in history.splits)
    for metric in ("rrmse", "mae", "r2"):
        assert math.isfinite(history.final_metrics[metric])
    assert not history.predictions.empty
    assert {"plot", "actual", "predicted"}.issubset(history.predictions.columns)


def test_every_fold_stores_a_full_per_plot_prediction_table():
    """Each fold carries predictions for EVERY row, so a saved bundle needs no source to show a map."""
    from ml import has_stored_predictions

    ds = make_dataset(n_plots=8, copies=3, n_features=4, seed=1)
    history = _run(ds, TrainConfig(model_key="ridge", seed=0))
    assert has_stored_predictions(history)
    for sm in history.splits:
        assert set(sm.plot_predictions.columns) == {"actual", "predicted", "plot", "aug"}
        assert len(sm.plot_predictions) == len(ds.frame)   # every row, not just the held-out ones


# --------------------------------------------------------------------------- #
# Per-feature normalisation                                                    #
# --------------------------------------------------------------------------- #

def test_build_scaling_modes_for_scale_sensitive_model():
    ridge = MODELS_BY_KEY["ridge"]  # needs_scaling = True
    cols = ["f0", "f1", "f2", "f3"]

    # None -> standardise everything (the original behaviour).
    all_scaled = ridge.build(normalize_columns=None, feature_columns=cols)
    assert isinstance(all_scaled.named_steps["scale"], StandardScaler)

    # A subset -> a ColumnTransformer scaling just those positions, others passthrough.
    subset = ridge.build(normalize_columns=["f0", "f2"], feature_columns=cols)
    ct = subset.named_steps["scale"]
    assert isinstance(ct, ColumnTransformer)
    assert ct.transformers[0][2] == [0, 2]

    # Empty list -> "normalize none": no scale step at all.
    none_scaled = ridge.build(normalize_columns=[], feature_columns=cols)
    assert "scale" not in none_scaled.named_steps


def test_build_never_scales_tree_model():
    rf = MODELS_BY_KEY["random_forest"]  # needs_scaling = False
    for normalize in (None, ["f0"], []):
        pipe = rf.build(normalize_columns=normalize, feature_columns=["f0", "f1"])
        assert "scale" not in pipe.named_steps


def test_partial_normalisation_trains_without_leakage():
    ds = make_dataset(n_plots=8, copies=2, n_features=4, seed=2)
    cfg = TrainConfig(
        model_key="ridge",
        feature_columns=["f0", "f1", "f2", "f3"],
        normalize_columns=["f0", "f2"],  # only some features standardised
        seed=0,
    )
    history = _run(ds, cfg)
    assert math.isfinite(history.final_metrics["rrmse"])
    # The fitted pipeline standardises exactly the requested columns.
    assert isinstance(history.model.named_steps["scale"], ColumnTransformer)


# --------------------------------------------------------------------------- #
# Augmented-data validation toggle                                            #
# --------------------------------------------------------------------------- #

def test_validation_excludes_augmented_rows_by_default():
    ds = make_dataset(n_plots=8, copies=3, seed=3)
    history = _run(ds, TrainConfig(model_key="ridge", seed=0), n_outer=4)
    # Default (validate_on_augmented=False): the held-out set is scored on originals only.
    assert "_aug(" not in " ".join(map(str, history.predictions.index))
    assert not history.predictions.empty


def test_validation_can_include_augmented_rows_when_enabled():
    ds = make_dataset(n_plots=8, copies=3, seed=3)
    history = _run(
        ds, TrainConfig(model_key="ridge", seed=0, validate_on_augmented=True), n_outer=4
    )
    # With the toggle on, augmented rows are allowed in the scored set.
    names = " ".join(map(str, history.predictions.index))
    assert "_aug(" in names


def test_run_raises_when_no_original_rows_and_default_validation():
    ds = make_dataset(n_plots=4, copies=2)
    # Make every row look augmented so there is nothing original to score the held-out fold on.
    ds.frame.index = [f"{Path(i).stem}_aug(9){Path(i).suffix}" for i in ds.frame.index]
    with pytest.raises(ValueError, match="original"):
        _run(ds, TrainConfig(model_key="ridge", seed=0), n_outer=2)


# --------------------------------------------------------------------------- #
# Augmented-data fitting toggle (main model)                                  #
# --------------------------------------------------------------------------- #

def test_train_on_augmented_false_fits_originals_only():
    ds = make_dataset(n_plots=8, copies=3, seed=6)  # 1 original + 2 augmented per plot
    history = _run(
        ds, TrainConfig(model_key="ridge", seed=0, train_on_augmented=False), n_outer=4
    )
    # Each plot contributes exactly one original row, so the fit set is one row per train plot.
    assert history.n_train == len(history.best_train_plots)


def test_train_on_augmented_true_fits_more_rows():
    ds = make_dataset(n_plots=8, copies=3, seed=6)
    with_aug = _run(ds, TrainConfig(model_key="ridge", seed=0, train_on_augmented=True), n_outer=4)
    no_aug = _run(ds, TrainConfig(model_key="ridge", seed=0, train_on_augmented=False), n_outer=4)
    # Same plots, but fitting on augmented rows means strictly more training rows.
    assert with_aug.n_train > no_aug.n_train


def test_run_raises_when_no_original_rows_to_fit():
    ds = make_dataset(n_plots=4, copies=2)
    ds.frame.index = [f"{Path(i).stem}_aug(9){Path(i).suffix}" for i in ds.frame.index]
    # Scoring allowed on augmented, but fitting forbidden -> the fit guard should fire.
    with pytest.raises(ValueError, match="fit"):
        _run(ds, TrainConfig(
            model_key="ridge", seed=0,
            train_on_augmented=False, validate_on_augmented=True,
        ), n_outer=2)


# --------------------------------------------------------------------------- #
# Precomputed metric cube (SplitModel.metric_variants)                        #
# --------------------------------------------------------------------------- #

def test_had_augmented_and_aug_toggles_meaningful():
    from ml import aug_toggles_meaningful

    # A run WITH augmented copies: cube present, had_augmented True, toggles meaningful.
    aug = _run(make_dataset(n_plots=8, copies=3, seed=11), TrainConfig(model_key="ridge"), n_outer=2)
    assert aug.had_augmented is True
    assert aug_toggles_meaningful(aug) is True
    # A run with ZERO augmented rows: cube still built, but had_augmented False, so the toggles are NOT
    # meaningful (flipping them can't change a metric) — the inconsistency the gate fixes.
    orig = _run(make_dataset(n_plots=8, copies=1, seed=11), TrainConfig(model_key="ridge"), n_outer=2)
    assert orig.had_augmented is False
    assert has_variant_cube(orig)                  # the cube exists...
    assert aug_toggles_meaningful(orig) is False   # ...but the toggles still shouldn't be enabled
    # Legacy fallback: a history predating the flag (had_augmented unknown/None) falls back to the cube
    # check, so old bundles never regress to disabled toggles.
    orig.had_augmented = None
    assert aug_toggles_meaningful(orig) is True


def test_metric_variants_present_and_shaped():
    ds = make_dataset(n_plots=8, copies=3, seed=11)
    history = _run(ds, TrainConfig(model_key="ridge", seed=0))
    assert has_variant_cube(history)
    for sm in history.splits:
        cube = sm.metric_variants
        assert set(cube["held_out"]) == {"orig_only", "with_aug"}
        assert set(cube["train"]) == {"orig_only", "with_aug"}
        assert set(cube["overall"]) == {"orig_orig", "aug_orig", "orig_aug", "aug_aug"}
        for cell in (cube["held_out"]["with_aug"], cube["train"]["with_aug"],
                     cube["overall"]["aug_aug"]):
            assert {"rrmse", "r2", "r", "mape", "mae", "n"} <= set(cell)


def test_orig_only_excludes_augmented_counts():
    # 1 original + 2 augmented per plot, so with_aug counts are 3x the orig_only counts per group.
    ds = make_dataset(n_plots=9, copies=3, seed=12)
    history = _run(ds, TrainConfig(model_key="ridge", seed=0))
    for sm in history.splits:
        ho = sm.metric_variants["held_out"]
        tr = sm.metric_variants["train"]
        assert ho["with_aug"]["n"] == 3 * ho["orig_only"]["n"]
        assert tr["with_aug"]["n"] == 3 * tr["orig_only"]["n"]
        # overall(orig,orig) counts only originals; overall(aug,aug) counts everything.
        assert sm.metric_variants["overall"]["aug_aug"]["n"] == (
            tr["with_aug"]["n"] + ho["with_aug"]["n"]
        )
        assert sm.metric_variants["overall"]["orig_orig"]["n"] == (
            tr["orig_only"]["n"] + ho["orig_only"]["n"]
        )


def test_overall_mixed_combo_matches_its_sides():
    # The default display state (train aug ON, val aug OFF) -> "aug_orig":
    # its sample count must be all training rows plus original held-out rows.
    ds = make_dataset(n_plots=9, copies=3, seed=13)
    history = _run(ds, TrainConfig(model_key="ridge", seed=0), n_outer=2)
    for sm in history.splits:
        cube = sm.metric_variants
        expected_n = cube["train"]["with_aug"]["n"] + cube["held_out"]["orig_only"]["n"]
        assert cube["overall"]["aug_orig"]["n"] == expected_n


def test_variant_metrics_falls_back_for_legacy_split():
    ds = make_dataset(n_plots=6, copies=2, seed=14)
    history = _run(ds, TrainConfig(model_key="ridge", seed=0), n_outer=2)
    sm = history.splits[0]
    sm.metric_variants = {}  # simulate a pre-cube bundle
    assert not has_variant_cube(history)
    assert variant_metrics(sm, "held_out", "orig_only") == sm.metrics
    assert variant_metrics(sm, "train", "with_aug") == sm.train_metrics
    assert variant_metrics(sm, "overall", "aug_aug") == {}


# --------------------------------------------------------------------------- #
# Training-split preview stats (ranges + direct R^2)                          #
# --------------------------------------------------------------------------- #

def test_training_split_stats_uses_train_rows_and_reacts_to_aug_toggle():
    # Each plot's augmented copy carries an extreme feature value (100), so including it
    # widens the observed range. Originals alone follow f0 = plot, y = 10*plot (R^2 = 1).
    idx, rows = [], []
    for plot in range(1, 6):
        idx.append(f"field_plot({plot}).laz")
        rows.append({"f0": float(plot), "y": float(10 * plot)})
        idx.append(f"field_plot({plot})_aug({plot}).laz")
        rows.append({"f0": 100.0, "y": float(10 * plot)})
    frame = pd.DataFrame(rows, index=idx)
    frame.index.name = "filename"
    ds = Dataset(frame=frame, feature_columns=["f0"], target_column="y")

    mn_no, mx_no, r2_no = training_split_stats(
        ds, ["f0"], test_size=0.25, seed=0, include_augmented=False)["f0"]
    mn_yes, mx_yes, r2_yes = training_split_stats(
        ds, ["f0"], test_size=0.25, seed=0, include_augmented=True)["f0"]

    assert mx_no < 100.0                     # originals only -> small range
    assert mx_yes >= 100.0                   # augmented value pulls the max up
    assert r2_no is None or 0.0 <= r2_no <= 1.0
    assert r2_yes is None or 0.0 <= r2_yes <= 1.0


# --------------------------------------------------------------------------- #
# Hyperopt augmented-data toggles                                             #
# --------------------------------------------------------------------------- #

def test_optimize_with_fit_on_augmented_off_runs():
    pytest.importorskip("optuna")
    ds = make_dataset(n_plots=8, copies=2, seed=7)
    cfg = TrainConfig(model_key="ridge", feature_columns=["f0", "f1", "f2", "f3"], seed=0)
    best = optimize_hyperparameters(
        ds, cfg, n_trials=3, fit_on_augmented=False, validate_on_augmented=False
    )
    assert set(best) >= {h.name for h in MODELS_BY_KEY["ridge"].hparams}


# --------------------------------------------------------------------------- #
# Hyperparameter optimisation (Optuna)                                        #
# --------------------------------------------------------------------------- #

def test_optimize_returns_params_within_ranges():
    pytest.importorskip("optuna")
    ds = make_dataset(n_plots=8, copies=2, seed=4)
    cfg = TrainConfig(model_key="ridge", feature_columns=["f0", "f1", "f2", "f3"], seed=0)
    best = optimize_hyperparameters(ds, cfg, n_trials=3)
    ridge = MODELS_BY_KEY["ridge"]
    for h in ridge.hparams:
        assert h.name in best
        if h.kind == "float":
            assert h.min <= best[h.name] <= h.max


def test_optimize_random_systematic_mode_runs_end_to_end():
    pytest.importorskip("optuna")
    ds = make_dataset(n_plots=12, copies=2, seed=4)
    cfg = TrainConfig(model_key="ridge", feature_columns=["f0", "f1", "f2", "f3"],
                      test_size=0.25, seed=0)
    # 12 plots, block 3 → up to 4 disjoint folds; ask for 4 to exercise an exact random partition.
    best = optimize_hyperparameters(ds, cfg, n_trials=3, n_cv_splits=4,
                                    split_mode="random_systematic")
    assert "alpha" in best


def test_optimize_logs_best_trial_cv_summary():
    pytest.importorskip("optuna")
    ds = make_dataset(n_plots=8, copies=2, seed=4)
    cfg = TrainConfig(model_key="ridge", feature_columns=["f0", "f1", "f2", "f3"], seed=0)
    notes: list[str] = []
    optimize_hyperparameters(ds, cfg, n_trials=3, note=notes.append)
    # The post-study quick-inspection summary reports the standard metric set across the CV folds.
    summary = [n for n in notes if "Best trial across" in n]
    assert len(summary) == 1
    assert all(tok in summary[0] for tok in ("mean rRMSE=", "R²=", "R=", "MAPE="))


def test_optimize_respects_augmented_validation_default():
    pytest.importorskip("optuna")
    ds = make_dataset(n_plots=8, copies=2, seed=5)
    # Default config validates on originals only; the search should still run end-to-end.
    cfg = TrainConfig(model_key="random_forest", seed=0)
    best = optimize_hyperparameters(ds, cfg, n_trials=3)
    assert set(best) >= {h.name for h in MODELS_BY_KEY["random_forest"].hparams}


def test_optimize_pls_clamps_n_components_to_feature_count():
    pytest.importorskip("optuna")
    # 4 features but the PLS HParam max is 20; the search must never suggest n_components > 4,
    # which would otherwise raise "n_components upper bound is 4".
    ds = make_dataset(n_plots=8, copies=2, n_features=4, seed=11)
    cfg = TrainConfig(
        model_key="pls", feature_columns=["f0", "f1", "f2", "f3"], seed=0
    )
    notes: list[str] = []
    best = optimize_hyperparameters(ds, cfg, n_trials=10, note=notes.append)
    assert 1 <= best["n_components"] <= 4
    assert any("capped at 4" in n for n in notes)


def test_optimize_threaded_folds_match_serial():
    pytest.importorskip("optuna")
    pytest.importorskip("joblib")
    # Threading the CV folds is purely a speedup: the folds are independent and each rebuilds its own
    # pipeline, so a deterministic (linear) model must return the same best params at any thread count.
    ds = make_dataset(n_plots=12, copies=2, n_features=4, seed=8)
    cfg = TrainConfig(model_key="ridge", feature_columns=["f0", "f1", "f2", "f3"], seed=0)
    serial = optimize_hyperparameters(ds, cfg, n_trials=8, n_cv_splits=4, seed=8, n_jobs=1)
    parallel = optimize_hyperparameters(ds, cfg, n_trials=8, n_cv_splits=4, seed=8, n_jobs=4)
    assert serial == parallel


def test_optimize_random_forest_threaded_folds_run_and_match():
    pytest.importorskip("optuna")
    pytest.importorskip("joblib")
    # RF builds with n_jobs=-1 for a single train, but the arbiter pins it single-threaded while the
    # folds fan out, so threading folds × forest can't oversubscribe — and the search still runs and
    # agrees with serial (RF predict jitters ~1e-15, so the same params are found).
    ds = make_dataset(n_plots=12, copies=2, n_features=4, seed=9)
    cfg = TrainConfig(model_key="random_forest", feature_columns=["f0", "f1", "f2", "f3"], seed=0)
    serial = optimize_hyperparameters(ds, cfg, n_trials=6, n_cv_splits=4, seed=9, n_jobs=1)
    parallel = optimize_hyperparameters(ds, cfg, n_trials=6, n_cv_splits=4, seed=9, n_jobs=4)
    assert set(serial) == set(parallel)
    assert serial == parallel

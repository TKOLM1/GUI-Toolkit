"""Unit tests for permutation feature importance (ml.explain)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml import Dataset, TrainConfig, permutation_importance, validate_procedure
from ml.trainer import active_split_model


def make_signal_dataset(n_plots: int = 12, copies: int = 2, seed: int = 0) -> Dataset:
    """A dataset where the target depends on f0 strongly, f1 weakly, and f2/f3 not at all.

    So a fitted model should rank f0 > f1 > {f2, f3} by permutation importance, giving the test a
    direction to assert rather than just a smoke check.
    """
    rng = np.random.default_rng(seed)
    feat_cols = ["f0", "f1", "f2", "f3"]
    target = "biomass"
    index, x_rows, y_vals = [], [], []
    for plot in range(1, n_plots + 1):
        base = rng.normal(size=4)
        true = float(10.0 * base[0] + 1.0 * base[1] + 50.0)  # f2, f3 carry no signal
        for c in range(copies):
            name = f"field_plot({plot}).laz" if c == 0 else f"field_plot({plot})_aug({c}).laz"
            index.append(name)
            x_rows.append(base + rng.normal(scale=0.01, size=4))
            y_vals.append(true + rng.normal(scale=0.1))
    frame = pd.DataFrame(x_rows, columns=feat_cols, index=index)
    frame[target] = y_vals
    frame.index.name = "filename"
    return Dataset(frame=frame, feature_columns=feat_cols, target_column=target)


def make_signal_history(dataset: Dataset, seed: int = 0, model_key: str = "random_forest"):
    """A TrainHistory to explain, built by the one run the app has: nested CV with fixed params."""
    config = TrainConfig(model_key=model_key, test_size=0.34, seed=seed)
    return validate_procedure(dataset, config, do_optimize=False, n_outer_splits=2).history


def test_importance_ranks_informative_feature_highest():
    dataset = make_signal_dataset()
    history = make_signal_history(dataset)
    result = permutation_importance(dataset, history, n_repeats=10, seed=0)
    assert not result.is_empty
    assert result.features[0] == "f0"                      # strongest driver ranks first
    top = dict(zip(result.features, result.mean))
    assert top["f0"] > top["f1"] > 0                       # f0 most important, f1 still positive
    assert top["f0"] > top["f2"] and top["f0"] > top["f3"]  # noise features matter far less


def test_importance_is_reproducible_with_seed():
    dataset = make_signal_dataset()
    history = make_signal_history(dataset)
    a = permutation_importance(dataset, history, n_repeats=8, seed=7)
    b = permutation_importance(dataset, history, n_repeats=8, seed=7)
    assert a.features == b.features
    assert np.allclose(a.mean, b.mean)                     # same seed -> identical shuffles -> identical ranking


def test_importance_threaded_matches_serial_exactly_for_deterministic_model():
    # A linear (ridge) model's predict is exactly deterministic, so the parallel shuffle path must
    # reproduce the serial ranking *bit-for-bit*: all permutations are drawn up front in the same
    # order, and each task scores against its own column copy (no shared-state race).
    import pytest
    pytest.importorskip("joblib")
    dataset = make_signal_dataset(seed=4)
    history = make_signal_history(dataset, seed=4, model_key="ridge")
    serial = permutation_importance(dataset, history, n_repeats=10, seed=2, n_jobs=1)
    parallel = permutation_importance(dataset, history, n_repeats=10, seed=2, n_jobs=4)
    assert serial.features == parallel.features
    assert np.array_equal(serial.mean, parallel.mean)
    assert np.array_equal(serial.std, parallel.std)


def test_importance_threaded_matches_serial_for_random_forest():
    # The default RF model's predict jitters by ~1e-15 (BLAS/threading), so this is the same allclose
    # agreement the serial path already has run-to-run — the threading adds no extra divergence.
    import pytest
    pytest.importorskip("joblib")
    dataset = make_signal_dataset(seed=6)
    history = make_signal_history(dataset, seed=6)
    serial = permutation_importance(dataset, history, n_repeats=10, seed=3, n_jobs=1)
    parallel = permutation_importance(dataset, history, n_repeats=10, seed=3, n_jobs=4)
    assert serial.features == parallel.features
    assert np.allclose(serial.mean, parallel.mean)
    assert np.allclose(serial.std, parallel.std)


def test_importance_repeats_yield_std_error_bars():
    dataset = make_signal_dataset()
    history = make_signal_history(dataset)
    result = permutation_importance(dataset, history, n_repeats=15, seed=1)
    assert result.n_repeats == 15
    assert result.std.shape == result.mean.shape
    assert (result.std >= 0).all()                         # std is a non-negative spread


def test_importance_held_out_uses_only_test_plots():
    dataset = make_signal_dataset()
    history = make_signal_history(dataset)
    active = active_split_model(history)
    n_test_plots = len(active.test_plots)
    result = permutation_importance(dataset, history, score_on="held_out", n_repeats=5, seed=0)
    # Every scored row belongs to a held-out plot, and each plot has `copies` rows.
    assert result.score_on == "held_out"
    assert 0 < result.n_scored <= n_test_plots * 2


def test_importance_all_scores_more_rows_than_held_out():
    dataset = make_signal_dataset()
    history = make_signal_history(dataset)
    held = permutation_importance(dataset, history, score_on="held_out", n_repeats=3, seed=0)
    everything = permutation_importance(dataset, history, score_on="all", n_repeats=3, seed=0)
    assert everything.n_scored > held.n_scored


def test_importance_excludes_aug_copies_when_toggle_off():
    """include_aug=False scores originals only — fewer rows than counting every copy."""
    dataset = make_signal_dataset(copies=3)  # one original + two aug copies per plot
    history = make_signal_history(dataset)
    with_aug = permutation_importance(dataset, history, score_on="held_out", include_aug=True,
                                      n_repeats=3, seed=0)
    orig_only = permutation_importance(dataset, history, score_on="held_out", include_aug=False,
                                       n_repeats=3, seed=0)
    assert orig_only.include_aug is False
    assert with_aug.include_aug is True
    # Originals-only keeps exactly one row per held-out plot; with-aug keeps all three copies.
    assert orig_only.n_scored < with_aug.n_scored
    assert with_aug.n_scored == orig_only.n_scored * 3


def test_importance_train_subset_also_respects_toggle():
    dataset = make_signal_dataset(copies=3)
    history = make_signal_history(dataset)
    with_aug = permutation_importance(dataset, history, score_on="train", include_aug=True,
                                      n_repeats=3, seed=0)
    orig_only = permutation_importance(dataset, history, score_on="train", include_aug=False,
                                       n_repeats=3, seed=0)
    assert orig_only.n_scored < with_aug.n_scored


def test_importance_empty_when_no_feature_columns():
    """A deep-style history (no tabular features) yields an empty, explained result, not a crash."""
    dataset = make_signal_dataset()
    history = make_signal_history(dataset)
    history.feature_columns = []                           # mimic a deep bundle
    result = permutation_importance(dataset, history, n_repeats=3, seed=0)
    assert result.is_empty
    assert result.notes                                    # carries an explanation for the UI

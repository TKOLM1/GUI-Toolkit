"""The log target transform: the round trip, the exponential trap, and the units every metric is in."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml import TrainConfig, TransformedTargetPipeline, build_estimator, validate_procedure
from ml.dataset import Dataset
from ml.models import MODELS_BY_KEY
from ml.target_transform import LOG, NAIVE, NONE, SMEARING, forward, inverse, smearing_factor
from tests.test_ml import make_dataset


def _lognormal_dataset(n_plots: int = 24, sigma: float = 0.4, seed: int = 0,
                       level: float = 5.5) -> Dataset:
    """A multiplicative, right-skewed target — the shape the log transform exists for.

    ``y = exp(linear(x) + N(0, sigma))``: log-linear in the features with constant *log*-scale noise,
    so on the raw scale the error grows with the plot. One row per plot (no augmented copies) so the
    fold arithmetic stays trivial and the test is about the transform only. ``level`` sets the log-scale
    intercept: the default puts the target around 250, a realistic biomass magnitude where ``log1p`` and
    ``log`` agree (see the note in :mod:`ml.target_transform` about targets of order 1).
    """
    rng = np.random.default_rng(seed)
    cols = ["f0", "f1"]
    x = rng.normal(size=(n_plots, 2))
    log_y = level + 0.8 * x[:, 0] - 0.5 * x[:, 1] + rng.normal(scale=sigma, size=n_plots)
    index = [f"field_plot({i + 1}).laz" for i in range(n_plots)]
    frame = pd.DataFrame(x, columns=cols, index=index)
    frame["biomass"] = np.exp(log_y)
    frame.index.name = "filename"
    return Dataset(frame=frame, feature_columns=cols, target_column="biomass")


def _config(**kw) -> TrainConfig:
    return TrainConfig(model_key="ridge", params={}, test_size=0.25, seed=0,
                       train_on_augmented=False, validate_on_augmented=False, **kw)


# --------------------------------------------------------------------------- #
# The transform itself                                                        #
# --------------------------------------------------------------------------- #

def test_round_trip_is_exact_without_a_correction():
    """``inverse(forward(y))`` returns y — the log1p/expm1 pair, with the factor at its neutral 1."""
    y = np.array([0.0, 0.5, 3.0, 120.0])
    assert np.allclose(inverse(forward(y), 1.0), y)


def test_log1p_accepts_exact_zeros_and_rejects_impossible_targets():
    """Zero biomass is a real measurement (plain log could not represent it); below -1 cannot be."""
    assert np.isfinite(forward(np.array([0.0, 1.0]))).all()
    with pytest.raises(ValueError, match="greater than -1"):
        forward(np.array([1.0, -2.0]))


def test_smearing_factor_recovers_the_lost_exponential_factor():
    """For log-normal residuals the smearing factor approximates ``exp(sigma^2/2)`` — the trap's size."""
    sigma = 0.5
    residuals = np.random.default_rng(0).normal(scale=sigma, size=20000)
    assert smearing_factor(residuals) == pytest.approx(np.exp(sigma**2 / 2), rel=0.02)


def test_smearing_factor_degrades_to_one_when_there_is_nothing_to_estimate():
    """No usable residuals ⇒ the neutral factor (the naive back-transform), never a NaN."""
    assert smearing_factor(np.array([])) == 1.0
    assert smearing_factor(np.array([np.nan, np.inf])) == 1.0


def test_naive_back_transform_under_predicts_and_smearing_corrects_it():
    """The exponential trap, measured: without the correction the fit is systematically low.

    Both estimators are fit on identical data; only the back-transform differs. The naive one predicts
    the conditional median, so its mean prediction falls short of the mean actual, while the smeared
    one lands close to it.
    """
    ds = _lognormal_dataset(n_plots=400, sigma=0.5)
    X = ds.frame[ds.feature_columns]
    y = ds.frame["biomass"]
    model_def = MODELS_BY_KEY["ridge"]
    kw = dict(normalize_columns=None, feature_columns=list(X.columns), target_transform=LOG)

    naive = build_estimator(model_def, {}, bias_correction=NAIVE, **kw).fit(X, y)
    smeared = build_estimator(model_def, {}, bias_correction=SMEARING, **kw).fit(X, y)

    ratio_naive = float(np.mean(naive.predict(X)) / y.mean())
    ratio_smeared = float(np.mean(smeared.predict(X)) / y.mean())
    assert ratio_naive < 0.95                                  # the median bias, plainly visible
    assert abs(ratio_smeared - 1.0) < abs(ratio_naive - 1.0)   # and materially reduced
    assert ratio_smeared == pytest.approx(1.0, abs=0.03)


def test_predictions_never_go_negative():
    """A back-transformed prediction is floored at 0 — negative biomass is not a value to propagate."""
    est = TransformedTargetPipeline(MODELS_BY_KEY["ridge"].build({}), bias_correction=NAIVE)
    est.smearing_ = 1.0
    assert np.all(inverse(np.array([-50.0, -1.0, 2.0]), 1.0) >= 0.0)


def test_wrapper_delegates_introspection_to_its_pipeline():
    """``named_steps`` falls through, so the Results tab reads a logged model's hyperparameters as usual."""
    est = TransformedTargetPipeline(MODELS_BY_KEY["ridge"].build({"alpha": 2.5}))
    assert est.named_steps["model"].alpha == 2.5


def test_build_estimator_only_wraps_when_the_transform_is_on():
    kw = dict(normalize_columns=None, feature_columns=["f0"])
    assert not isinstance(build_estimator(MODELS_BY_KEY["ridge"], {}, target_transform=NONE, **kw),
                          TransformedTargetPipeline)
    assert isinstance(build_estimator(MODELS_BY_KEY["ridge"], {}, target_transform=LOG, **kw),
                      TransformedTargetPipeline)


# --------------------------------------------------------------------------- #
# What a run reports                                                          #
# --------------------------------------------------------------------------- #

def test_every_reported_number_stays_in_the_targets_own_units():
    """Predictions, the metric cube and the stored per-plot table are all on the raw target's scale.

    The guard against the whole failure mode this feature could introduce: if anything leaked log
    units, the predictions would sit near ``log1p(y)`` (single digits) rather than near ``y``, and the
    stored ``actual`` column would no longer match the dataset's own values.
    """
    ds = _lognormal_dataset(n_plots=24)
    result = validate_procedure(ds, _config(target_transform=LOG), do_optimize=False,
                                n_outer_splits=3, outer_split_mode="sequential")
    raw = ds.frame["biomass"]

    for sm in result.history.splits:
        stored = sm.plot_predictions
        # ``actual`` is the untransformed target, row for row.
        assert np.allclose(stored["actual"].to_numpy(), raw.loc[stored.index].to_numpy())
        # ...and the predictions live on that same scale, not the log one.
        assert stored["predicted"].mean() > raw.mean() / 3
        assert np.all(stored["predicted"] >= 0)
        assert sm.metrics["rrmse"] == pytest.approx(
            100 * np.sqrt(np.mean((sm.predictions["actual"] - sm.predictions["predicted"]) ** 2))
            / abs(sm.predictions["actual"].mean())
        )
        for scope in ("held_out", "train"):
            for variant in ("orig_only", "with_aug"):
                assert sm.metric_variants[scope][variant]["n"] > 0


def test_run_records_its_target_handling_for_the_bundle_and_results_tab():
    """The history carries the choice, so a saved model's units are never ambiguous later."""
    ds = _lognormal_dataset(n_plots=16)
    history = validate_procedure(ds, _config(target_transform=LOG, target_bias_correction=NAIVE),
                                 do_optimize=False, n_outer_splits=2).history
    assert history.target_transform == LOG
    assert history.target_bias_correction == NAIVE


def test_a_default_run_is_untouched_by_the_feature():
    """With the transform off nothing wraps and nothing is recorded — old behaviour, bit for bit."""
    ds = make_dataset(n_plots=8, copies=3, n_features=4, seed=2)
    history = validate_procedure(ds, _config(), do_optimize=False, n_outer_splits=2).history
    assert history.target_transform == NONE
    assert not isinstance(history.splits[0].model, TransformedTargetPipeline)


def test_an_impossible_target_fails_before_any_fold_runs():
    """A target the transform cannot represent is caught up front, not three folds into a search."""
    ds = _lognormal_dataset(n_plots=12)
    ds.frame.iloc[0, ds.frame.columns.get_loc("biomass")] = -5.0
    with pytest.raises(ValueError, match="greater than -1"):
        validate_procedure(ds, _config(target_transform=LOG), do_optimize=False, n_outer_splits=2)


def test_the_log_fit_beats_the_raw_fit_on_a_multiplicative_target():
    """The reason the toggle exists: on a log-linear target the transform lowers the honest rRMSE."""
    ds = _lognormal_dataset(n_plots=40, sigma=0.35, seed=3)
    kw = dict(do_optimize=False, n_outer_splits=4, outer_split_mode="systematic")
    raw = validate_procedure(ds, _config(), **kw)
    logged = validate_procedure(ds, _config(target_transform=LOG), **kw)
    assert logged.mean_held_out_rrmse < raw.mean_held_out_rrmse

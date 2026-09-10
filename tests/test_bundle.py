"""Tests for the shared model bundle (ml.bundle) used by both learning tabs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ml
from ml import Dataset, TrainConfig, validate_procedure
from ml.results import compute_plot_predictions


def make_dataset(n_plots: int = 6, copies: int = 2, n_features: int = 3, seed: int = 0) -> Dataset:
    """A synthetic feature/target dataset, mirroring tests/test_ml.py's helper."""
    rng = np.random.default_rng(seed)
    feat_cols = [f"f{i}" for i in range(n_features)]
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
    frame = pd.DataFrame(x_rows, columns=feat_cols, index=index)
    frame["biomass"] = y_vals
    return Dataset(frame=frame, feature_columns=feat_cols, target_column="biomass")


def _run(ds: Dataset, cfg: TrainConfig, n_outer: int = 3):
    """The history a run produces — nested CV with fixed hyperparameters (the app's only path)."""
    return validate_procedure(ds, cfg, do_optimize=False, n_outer_splits=n_outer).history


def _trained(seed: int = 1):
    ds = make_dataset(seed=seed)
    cfg = TrainConfig(model_key="random_forest", feature_columns=ds.feature_columns, seed=seed)
    return ds, _run(ds, cfg)


def test_classical_bundle_round_trip_matches_compute_plot_predictions(tmp_path):
    ds, history = _trained()
    path = ml.save_bundle(history, tmp_path, model_key="random_forest")
    assert path.exists()

    loaded = ml.load_bundle(path)
    assert loaded.kind == "classical"
    assert loaded.model_key == "random_forest"
    assert loaded.target_column == "biomass"

    table = loaded.predict_table(ds)
    ref = compute_plot_predictions(ds, history)
    assert list(table.columns) == ["plot", "aug", "actual", "predicted", "error_pct", "role"]
    assert np.allclose(table["predicted"].to_numpy(), ref["predicted"].to_numpy())
    assert np.array_equal(table.index, ref.index)


def test_bundle_keeps_all_splits(tmp_path):
    ds, history = _trained()
    path = ml.save_bundle(history, tmp_path, model_key="random_forest")
    loaded = ml.load_bundle(path)
    assert len(loaded.history.splits) == len(history.splits) == 3
    # each split keeps its own held-out predictions (for the performance scatter)
    for sm in loaded.history.splits:
        assert {"rrmse", "mae", "r2"} <= set(sm.metrics)
        assert not sm.predictions.empty


def test_list_bundles_newest_first(tmp_path):
    ds, h1 = _trained(seed=1)
    _, h2 = _trained(seed=2)
    p1 = ml.save_bundle(h1, tmp_path, model_key="random_forest")
    # force a later timestamp on the second save
    import time
    time.sleep(1.1)
    p2 = ml.save_bundle(h2, tmp_path, model_key="pls")
    infos = ml.list_bundles(tmp_path)
    assert [i.path for i in infos] == [p2, p1]  # newest first
    assert infos[0].kind == "classical"
    assert "ML" in infos[0].label


def test_bundle_carries_validate_on_augmented_and_tags_label(tmp_path):
    ds = make_dataset()
    cfg = TrainConfig(
        model_key="random_forest", feature_columns=ds.feature_columns,
        seed=1, validate_on_augmented=True,
    )
    history = _run(ds, cfg)
    assert history.validate_on_augmented is True

    path = ml.save_bundle(history, tmp_path, model_key="random_forest")
    loaded = ml.load_bundle(path)
    assert loaded.history.validate_on_augmented is True

    info = ml.bundle_info(path)
    assert info.validate_on_augmented is True
    assert "(AUG)" in info.label


def test_bundle_carries_train_on_augmented(tmp_path):
    """The per-model train-on-augmented setting survives the bundle round trip (the Results page reads
    it from the loaded history to show the training-side badge). It defaults True; an explicit False
    must persist too."""
    ds = make_dataset()
    cfg = TrainConfig(
        model_key="random_forest", feature_columns=ds.feature_columns,
        seed=1, train_on_augmented=False,
    )
    history = _run(ds, cfg)
    assert history.train_on_augmented is False
    loaded = ml.load_bundle(ml.save_bundle(history, tmp_path, model_key="random_forest"))
    assert loaded.history.train_on_augmented is False

    # the default (fit on augmented rows too) round-trips as True
    _, default_history = _trained()
    assert default_history.train_on_augmented is True
    default_loaded = ml.load_bundle(
        ml.save_bundle(default_history, tmp_path, model_key="pls")
    )
    assert default_loaded.history.train_on_augmented is True


def test_bundle_label_has_no_aug_tag_by_default(tmp_path):
    _, history = _trained()
    assert history.validate_on_augmented is False
    info = ml.bundle_info(ml.save_bundle(history, tmp_path, model_key="random_forest"))
    assert info.validate_on_augmented is False
    assert "(AUG)" not in info.label


def test_nested_cv_bundle_is_tagged(tmp_path):
    """A bundle saved with vault_meta is tagged source='nested_cv', labelled '(nested CV)', and carries
    the meta back on load. A plain bundle stays source='train' with no tag."""
    _, history = _trained()
    meta = {"mean_held_out_rrmse": 12.3, "std_held_out_rrmse": 1.2, "n_outer": 4,
            "did_optimize": False,
            "fold_features": [["a"]], "fold_params": [{}]}
    path = ml.save_bundle(history, tmp_path, model_key="ridge", vault_meta=meta)

    loaded = ml.load_bundle(path)
    assert loaded.source == "nested_cv" and loaded.is_nested_cv
    assert loaded.vault_meta["mean_held_out_rrmse"] == 12.3

    info = ml.bundle_info(path)  # reads via the sidecar
    assert info.source == "nested_cv"
    assert "(nested CV)" in info.label

    plain = ml.bundle_info(ml.save_bundle(history, tmp_path, model_key="pls"))
    assert plain.source == "train"
    assert "(nested CV)" not in plain.label


def test_summary_meta_packs_a_plain_model_into_one_fold(tmp_path):
    """A regular (non-nested) bundle gets a one-"fold" stability summary: its feature set and the
    hyperparameters the estimator was actually fitted with, read off the pipeline (not form values)."""
    from gui.results.nested_cv_stability import summary_meta

    ds = make_dataset(seed=1)
    cfg = TrainConfig(model_key="ridge", feature_columns=ds.feature_columns, seed=1)
    loaded = ml.load_bundle(
        ml.save_bundle(_run(ds, cfg), tmp_path, model_key="ridge")
    )

    meta = summary_meta(loaded)
    assert not loaded.is_nested_cv
    assert meta["fold_features"] == [ds.feature_columns]  # one fold = the model's feature set
    assert meta["did_optimize"]
    # The fitted estimator's declared hparam (alpha) is read back off the active split, not a spread.
    active = ml.active_split_model(loaded.history)
    assert list(meta["fold_params"]) == [{"alpha": active.model.named_steps["model"].alpha}]
    assert summary_meta(None) is None


def test_bundle_format_is_4(tmp_path):
    import joblib

    _, history = _trained()
    path = ml.save_bundle(history, tmp_path, model_key="random_forest")
    payload = joblib.load(path)
    assert payload["format"] == 4


def test_stored_table_matches_compute_plot_predictions(tmp_path):
    from ml import has_stored_predictions

    ds, history = _trained()
    assert has_stored_predictions(history)
    path = ml.save_bundle(history, tmp_path, model_key="random_forest")
    loaded = ml.load_bundle(path)
    assert loaded.has_stored_predictions

    # Stored lookup must equal the live source-based computation for the active split…
    for split in (history.best_split, history.splits[0].split, history.splits[-1].split):
        loaded.history.active_split = split
        stored = loaded.stored_table()
        ref = compute_plot_predictions(ds, loaded.history)
        # Same rows; compare on a common sort.
        stored = stored.sort_index()
        ref = ref.sort_index()
        assert list(stored.columns) == list(ref.columns)
        assert np.allclose(stored["predicted"].to_numpy(), ref["predicted"].to_numpy())
        assert np.allclose(stored["error_pct"].to_numpy(), ref["error_pct"].to_numpy(),
                           equal_nan=True)
        assert list(stored["role"]) == list(ref["role"])  # role tracks the active split


def test_stored_table_covers_every_plot_and_aug(tmp_path):
    ds, history = _trained()
    for sm in history.splits:
        # full table covers all rows; held-out predictions are a subset
        assert len(sm.plot_predictions) >= len(sm.predictions)
        assert set(sm.plot_predictions.columns) == {"actual", "predicted", "plot", "aug"}


def test_legacy_bundle_without_stored_predictions_falls_back(tmp_path):
    import pandas as pd

    from ml import has_stored_predictions

    ds, history = _trained()
    for sm in history.splits:  # strip stored predictions -> simulate a format<=3 bundle
        sm.plot_predictions = pd.DataFrame()
    path = ml.save_bundle(history, tmp_path, model_key="random_forest")
    loaded = ml.load_bundle(path)
    assert not has_stored_predictions(loaded.history)
    assert not loaded.has_stored_predictions
    with pytest.raises(ValueError):
        loaded.stored_table()
    # the live path still works
    table = loaded.predict_table(ds)
    ref = compute_plot_predictions(ds, loaded.history)
    assert np.allclose(table.sort_index()["predicted"].to_numpy(),
                       ref.sort_index()["predicted"].to_numpy())


def test_bundle_round_trips_metric_variants(tmp_path):
    from ml import has_variant_cube

    ds, history = _trained()
    assert has_variant_cube(history)
    path = ml.save_bundle(history, tmp_path, model_key="random_forest")
    loaded = ml.load_bundle(path)
    assert has_variant_cube(loaded.history)
    for orig, got in zip(history.splits, loaded.history.splits):
        assert orig.metric_variants == got.metric_variants
        assert set(got.metric_variants["overall"]) == {
            "orig_orig", "aug_orig", "orig_aug", "aug_aug"
        }


def test_old_bundle_without_cube_still_loads(tmp_path):
    from ml import has_variant_cube
    from ml.trainer import variant_metrics

    ds, history = _trained()
    for sm in history.splits:  # strip the cube to mimic a pre-format-3 bundle
        sm.metric_variants = {}
    path = ml.save_bundle(history, tmp_path, model_key="random_forest")
    loaded = ml.load_bundle(path)
    assert not has_variant_cube(loaded.history)
    sm = loaded.history.splits[0]
    assert variant_metrics(sm, "held_out", "orig_only") == sm.metrics
    assert variant_metrics(sm, "overall", "aug_aug") == {}


def test_list_bundles_tolerates_foreign_joblib(tmp_path):
    import joblib

    joblib.dump({"not": "a bundle"}, tmp_path / "stray.joblib")
    ds, history = _trained()
    ml.save_bundle(history, tmp_path, model_key="random_forest")
    infos = ml.list_bundles(tmp_path)
    assert len(infos) == 1  # the stray file is skipped, the real bundle is listed

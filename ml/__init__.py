"""ml - classical machine-learning module for predicting biomass from plot features.

Public surface:
    MODELS / MODELS_BY_KEY - the model registry (drives the GUI).
    Dataset / load_dataset / make_xy - feature (X) + target (y) loading.
    grouped_split - leakage-free train/test split by plot number.
    TrainConfig / TrainHistory / SplitModel - a run's inputs and its per-fold outputs.
    validate_procedure / sweep_models - the nested-CV run, for one model or for every model.
    build_estimator / TransformedTargetPipeline - the model pipeline, optionally fit on a log target
        and back-transformed so every reported metric stays in the target's own units.
"""

from .models import MODELS, MODELS_BY_KEY, HParam, ModelDef
from .dataset import Dataset, load_dataset, make_xy
from .splitting import grouped_split, max_splits_for_test_size
from .metrics import mae, mape, r, r2, rmse, rrmse, standard_metrics
from .trainer import (
    CycleResult,
    SplitModel,
    TrainConfig,
    TrainControl,
    TrainHistory,
    TrainingStopped,
    active_split_model,
    aug_toggles_meaningful,
    average_split_metrics,
    average_split_train_metrics,
    has_stored_predictions,
    has_variant_cube,
    variant_metrics,
)
from .target_transform import TransformedTargetPipeline, build_estimator
from .optimize import optimize_hyperparameters
from .validate import VaultResult, sweep_models, validate_procedure
from .preview import training_split_stats
from .explain import ImportanceResult, permutation_importance
from .bundle import (
    BundleInfo,
    LoadedModel,
    bundle_info,
    delete_bundle,
    list_bundles,
    load_bundle,
    prune_orphan_info_sidecars,
    save_bundle,
)

__all__ = [
    "MODELS",
    "MODELS_BY_KEY",
    "HParam",
    "ModelDef",
    "Dataset",
    "load_dataset",
    "make_xy",
    "grouped_split",
    "max_splits_for_test_size",
    "rrmse",
    "rmse",
    "mae",
    "r2",
    "r",
    "mape",
    "standard_metrics",
    "TrainConfig",
    "TrainHistory",
    "CycleResult",
    "SplitModel",
    "TrainControl",
    "TrainingStopped",
    "active_split_model",
    "average_split_metrics",
    "average_split_train_metrics",
    "variant_metrics",
    "has_variant_cube",
    "aug_toggles_meaningful",
    "has_stored_predictions",
    "TransformedTargetPipeline",
    "build_estimator",
    "optimize_hyperparameters",
    "validate_procedure",
    "sweep_models",
    "VaultResult",
    "training_split_stats",
    "permutation_importance",
    "ImportanceResult",
    "save_bundle",
    "load_bundle",
    "list_bundles",
    "bundle_info",
    "delete_bundle",
    "prune_orphan_info_sidecars",
    "BundleInfo",
    "LoadedModel",
]

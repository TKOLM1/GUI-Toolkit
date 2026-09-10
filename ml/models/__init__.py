"""The model registry - the single source of truth for the ML model dropdown.

Each model lives in its own module and exports a :class:`ModelDef`; they are collected
here into :data:`MODELS` (display order) and :data:`MODELS_BY_KEY` (lookup). Adding a
new model is: drop in a new file exporting a ``ModelDef`` and add it to this list.
"""

from __future__ import annotations

from .base import HParam, ModelDef
from .random_forest import MODEL as RANDOM_FOREST
from .hist_gbt import MODEL as HIST_GBT
from .svr import MODEL as SVR_MODEL
from .knn import MODEL as KNN
from .linear import ELASTIC_NET, LASSO, RIDGE
from .pls import MODEL as PLS
from .gpr import MODEL as GPR

MODELS: list[ModelDef] = [
    RANDOM_FOREST,
    HIST_GBT,
    SVR_MODEL,
    KNN,
    RIDGE,
    LASSO,
    ELASTIC_NET,
    PLS,
    GPR,
]

MODELS_BY_KEY: dict[str, ModelDef] = {m.key: m for m in MODELS}

__all__ = [
    "MODELS",
    "MODELS_BY_KEY",
    "ModelDef",
    "HParam",
]

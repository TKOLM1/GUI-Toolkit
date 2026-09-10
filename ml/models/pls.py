"""Partial Least Squares regression."""

from __future__ import annotations

from sklearn.cross_decomposition import PLSRegression

from .base import HParam, ModelDef


def _factory(p: dict) -> PLSRegression:
    return PLSRegression(n_components=int(p["n_components"]))


MODEL = ModelDef(
    key="pls",
    label="Partial Least Squares",
    tooltip="Projects features onto a few latent components correlated with the target; "
            "good for many correlated features. Scale-sensitive.",
    hparams=(
        HParam("n_components", "Components", "int", 2, 1, 20,
               tooltip="Number of latent components (must be <= number of features)."),
    ),
    needs_scaling=True,
    factory=_factory,
)

"""Support Vector Regression."""

from __future__ import annotations

from sklearn.svm import SVR

from .base import HParam, ModelDef


def _factory(p: dict) -> SVR:
    return SVR(
        C=float(p["C"]),
        epsilon=float(p["epsilon"]),
        kernel=str(p["kernel"]),
        gamma=str(p["gamma"]),
    )


MODEL = ModelDef(
    key="svr",
    label="Support Vector Regression",
    tooltip="Kernel SVR; scale-sensitive, so features are standardised in the pipeline.",
    hparams=(
        HParam("C", "C (regularisation)", "float", 10.0, 0.01, 1000.0, step=1.0,
               tooltip="Penalty on errors; larger C fits the training data harder."),
        HParam("epsilon", "Epsilon (tube width)", "float", 0.1, 0.0, 10.0, step=0.05,
               tooltip="Width of the no-penalty tube around the regression line."),
        HParam("kernel", "Kernel", "choice", "rbf", choices=("rbf", "linear", "poly"),
               tooltip="Kernel function mapping inputs to a higher-dimensional space."),
        HParam("gamma", "Gamma", "choice", "scale", choices=("scale", "auto"),
               tooltip="Kernel coefficient for rbf/poly kernels."),
    ),
    needs_scaling=True,
    factory=_factory,
)

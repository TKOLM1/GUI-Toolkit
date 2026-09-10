"""Gaussian Process regression.

Note: GPR scales ~O(n^3) in the number of training rows, so it can be slow and
memory-heavy on large augmented datasets - the ML page warns when it is selected.
"""

from __future__ import annotations

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from .base import HParam, ModelDef


def _factory(p: dict) -> GaussianProcessRegressor:
    kernel = (
        ConstantKernel(1.0)
        * RBF(length_scale=float(p["length_scale"]))
        + WhiteKernel(noise_level=float(p["noise_level"]))
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=float(p["alpha"]),
        normalize_y=True,
        random_state=0,
    )


MODEL = ModelDef(
    key="gpr",
    label="Gaussian Process",
    tooltip="Non-parametric Bayesian regression with an RBF kernel; scale-sensitive and "
            "slow on large datasets (O(n^3)).",
    hparams=(
        HParam("length_scale", "RBF length scale", "float", 1.0, 0.01, 100.0, step=0.1,
               tooltip="Smoothness of the RBF kernel; larger = smoother fit."),
        HParam("noise_level", "Noise level (regularisation)", "float", 1.0, 0.0001, 100.0, step=0.1,
               tooltip="White-noise kernel level absorbing observation noise. This is the GPR's "
                       "regulariser: larger = smoother, more tolerant of noisy targets."),
        HParam("alpha", "Jitter (alpha)", "float", 1e-6, 1e-12, 1e-3, step=1e-6,
               tooltip="Tiny value added to the kernel diagonal for numerical stability only. "
                       "Regularisation is the noise-level knob above, not this one."),
    ),
    needs_scaling=True,
    factory=_factory,
)

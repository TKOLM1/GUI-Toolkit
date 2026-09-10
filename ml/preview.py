"""Per-feature preview stats for the ML page - computed on the *training split only*.

The ML page shows, next to each feature, its value range and a quick "direct R": the Pearson
correlation between that single feature and the target (sign preserved, so the user sees the
direction of the relationship). It is computed on a **representative training split only**
(never the held-out rows), so the preview can never leak test information, and it honours the
"use augmented data for fitting" toggle so the numbers reflect exactly the rows the model
would be fit on.

This is a descriptive aid, not part of the model: the values are display-only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common.naming import aug_number_from_name
from .dataset import Dataset, make_xy
from .metrics import r as pearson_r
from .splitting import grouped_split

# One feature's preview: (min, max, direct_r). Any field is None when undefined.
FeatureStat = tuple[float | None, float | None, float | None]


def _direct_r(x: np.ndarray, y: np.ndarray) -> float | None:
    """Signed Pearson correlation between a single feature and the target, or None."""
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    val = pearson_r(x, y)
    return val if np.isfinite(val) else None


def training_split_stats(
    dataset: Dataset,
    feature_cols: list[str],
    test_size: float,
    seed: int,
    include_augmented: bool,
) -> dict[str, FeatureStat]:
    """Return ``{feature: (min, max, direct_r)}`` over the representative training split.

    The split is the same grouped train/test split the trainer's first cycle uses
    (``grouped_split(groups, test_size, seed)``); only its **training** rows are used, with
    augmented rows dropped when ``include_augmented`` is False.
    """
    empty: dict[str, FeatureStat] = {c: (None, None, None) for c in feature_cols}
    if not feature_cols:
        return empty

    X, y, groups = make_xy(dataset, feature_cols)
    if len(X) < 2:
        return empty

    try:
        tr_idx, _ = grouped_split(groups, test_size, seed)
    except ValueError:
        tr_idx = np.arange(len(X))  # too few groups to split: fall back to all rows

    is_aug = np.array([aug_number_from_name(str(i)) is not None for i in X.index])
    sel = tr_idx if include_augmented else tr_idx[~is_aug[tr_idx]]
    if len(sel) == 0:
        sel = tr_idx  # no original rows in the train split: fall back so the preview is non-empty

    X_tr = X.iloc[sel]
    y_tr = pd.to_numeric(y.iloc[sel], errors="coerce").to_numpy(dtype=float)

    stats: dict[str, FeatureStat] = {}
    for col in feature_cols:
        values = pd.to_numeric(X_tr[col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        mn = float(values[finite].min()) if finite.any() else None
        mx = float(values[finite].max()) if finite.any() else None
        pair = finite & np.isfinite(y_tr)
        direct_r = _direct_r(values[pair], y_tr[pair])
        stats[col] = (mn, mx, direct_r)
    return stats

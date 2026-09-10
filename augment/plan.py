"""Turning the user's augmentation settings into a concrete per-sample plan.

:class:`AugmentConfig` is the whole configuration the GUI produces. For each augmented
output the runner asks :func:`sample_plan` to:

1. pick how many distinct methods to apply (a count drawn from ``[min_methods,
   max_methods]``), then
2. choose that many distinct methods from the selected set, and
3. draw concrete parameters for each from the user's ranges.

The chosen methods are returned in registry order so the geometric/noise/sampling
operations always compose in the same, predictable sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .transforms import AUG_BY_KEY, AUGMENTATIONS, Range

# The registry order, used to sort a sampled plan deterministically.
_ORDER = {a.key: i for i, a in enumerate(AUGMENTATIONS)}


@dataclass
class AugmentConfig:
    """Everything needed to drive an augmentation run."""

    selected: list[str]                              # augmentation keys that are enabled
    ranges: dict[str, dict[str, Range]]              # key -> {param_name: (min, max)}
    min_methods: int = 1                             # min distinct methods per augmented sample
    max_methods: int = 2                             # max distinct methods per augmented sample
    n_per_sample: int = 5                            # augmented copies per original file
    seed: int = 0                                    # master RNG seed (reproducible runs)
    height_channel: str = "RelativeHeight"               # channel jitter_z shifts together with z
    out_suffix: str = ".laz"                         # output file format

    def effective_method_bounds(self) -> tuple[int, int]:
        """Clamp the requested method-count bounds to what the selection allows."""
        n_sel = len(self.selected)
        lo = max(1, min(self.min_methods, n_sel))
        hi = max(lo, min(self.max_methods, n_sel))
        return lo, hi


# One applied transform: its key and the concrete parameters that were drawn.
AppliedAug = tuple[str, dict]


def sample_plan(config: AugmentConfig, rng: np.random.Generator) -> list[AppliedAug]:
    """Sample the list of (key, params) transforms for one augmented output."""
    if not config.selected:
        return []
    lo, hi = config.effective_method_bounds()
    k = int(rng.integers(lo, hi + 1))
    chosen = list(rng.choice(np.array(config.selected, dtype=object), size=k, replace=False))
    chosen.sort(key=lambda key: _ORDER[key])  # compose in registry order
    plan: list[AppliedAug] = []
    for key in chosen:
        ranges = config.ranges.get(key, {})
        params = AUG_BY_KEY[key].sample(ranges, rng)
        plan.append((key, params))
    return plan

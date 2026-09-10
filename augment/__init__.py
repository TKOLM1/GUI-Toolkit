"""augment - data augmentation for per-plot LiDAR clouds.

Public surface:
    AUGMENTATIONS     - the ordered registry of transforms (drives GUI + runner).
    AugmentDef        - one transform definition.
    AugmentConfig     - the whole run configuration.
    sample_plan       - draw the (key, params) transforms for one augmented output.
    run_augment       - batch: copies + augmented files + manifest.
    load_previous_output - reuse a previous output folder (skip re-augmenting).
"""

from .transforms import AUG_BY_KEY, AUGMENTATIONS, AugmentDef, AugParam
from .plan import AugmentConfig, sample_plan
from .runner import AugmentResult, run_augment
from .manifest import MANIFEST_NAME, PreviousOutput, load_previous_output

__all__ = [
    "AUGMENTATIONS",
    "AUG_BY_KEY",
    "AugmentDef",
    "AugParam",
    "AugmentConfig",
    "sample_plan",
    "run_augment",
    "AugmentResult",
    "load_previous_output",
    "PreviousOutput",
    "MANIFEST_NAME",
]

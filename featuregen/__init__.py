"""featuregen - LiDAR vegetation feature generator for individual plots.

Public surface:
    Config            - user-tunable parameters (channel name, class codes, bin size).
    PlotData          - the per-file arrays a metric needs, read once and cached.
    load_plot         - read a .las/.laz file into a PlotData.
    FEATURES          - the ordered registry of every feature (drives GUI + computation).
    compute_features  - run a selection of features over one PlotData.
    run_batch         - process many files into a .csv table.
"""

from .io_las import Config, PlotData, load_plot
from .features import FEATURES, FeatureDef, compute_features
from .external import (
    ExternalData,
    default_reference_dir,
    load_external,
    plot_number_from_name,
)
from .pipeline import run_batch

__all__ = [
    "Config",
    "PlotData",
    "load_plot",
    "FEATURES",
    "FeatureDef",
    "compute_features",
    "ExternalData",
    "load_external",
    "plot_number_from_name",
    "default_reference_dir",
    "run_batch",
]

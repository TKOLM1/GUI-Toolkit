"""The hand-off state the three modules share.

The shell owns a single :class:`Session`. Each module reads what it needs and, when
the user presses a "Go to ..." button, fills in the fields the next module consumes -
so augmentation can seed feature generation's inputs, and feature generation can seed
the ML module's feature/target tables. Every field is optional: each page also works
standalone with nothing pre-filled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.project import Project


@dataclass
class Session:
    """Mutable hand-off state passed between the Clip, Augmentation, Feature, ML and Results pages."""

    # The active project folder: the single source of truth for every stage's input/output
    # locations. None until the user creates or opens one on the first tab.
    project: Project | None = None

    # Clip / Augmentation -> Feature generation
    input_files: list[Path] = field(default_factory=list)  # files to feature-generate
    augment_output_dir: Path | None = None                 # the shared plots/ folder
    manifest_path: Path | None = None                      # the augmentation manifest .csv

    # Import -> Feature generation: the reference sheet. The Import tab owns getting the file in
    # and linking it to the plots (which column holds the plot number, how many plots matched);
    # the Feature tab owns what is done with it (which columns become features, which is the
    # target), and reads the loaded table from here rather than picking the file again.
    reference_path: Path | None = None                     # the chosen .csv/.xlsx
    reference_key: str | None = None                       # its plot-number column (None = first)
    external: Any | None = None                            # featuregen.external.ExternalData

    # Feature generation -> ML
    feature_table_path: Path | None = None                 # features.csv (X)
    targets_table_path: Path | None = None                 # target.csv (y)
    target_column: str | None = None                       # the chosen ground-truth column name

    # ML -> Results
    train_history: Any | None = None                       # ml.TrainHistory of the last training
    ml_dataset: Any | None = None                          # ml.Dataset the model was trained on

    def reset_handoff(self) -> None:
        """Clear every hand-off field (everything except :attr:`project`) back to its default.

        Called when a project is created or opened so a new project never inherits the previous
        one's input files, workbook paths, chosen target or trained-model state. The shell pairs
        this with each page's ``reset_for_project`` to wipe the per-page caches too.
        """
        self.input_files = []
        self.augment_output_dir = None
        self.manifest_path = None
        self.reference_path = None
        self.reference_key = None
        self.external = None
        self.feature_table_path = None
        self.targets_table_path = None
        self.target_column = None
        self.train_history = None
        self.ml_dataset = None

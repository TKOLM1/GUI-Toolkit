# Wheat Biomass Toolkit

A desktop application for predicting biomass from LiDAR
point clouds of field plots. It covers the whole path from raw scans to a
validated model: clip or import per-plot clouds, optionally augment them,
generate vegetation features, train and nested-cross-validate a range of
regressors, then explore the results against the field layout.

Built with PySide6 (Qt), with a Polyscope 3D viewer for inspecting individual
plots from the results map.

## Pipeline

The app is a nav bar over six pages, run left to right; each page hands its
output to the next through a shared session.

| Page | What it does |
| --- | --- |
| **Project** | Create or open a project folder, and pick the active preset. |
| **Import** | Get per-plot clouds into the project — either clip a whole-field cloud with a `.gpkg` of labelled plot polygons, or import a folder of already-separated `.las` / `.laz` / `.pcd` files. |
| **Augment** | Optional data augmentation: batch transforms over the per-plot clouds, tracked by a manifest so augmented copies stay tied to their source plot. |
| **Features** | Compute LiDAR vegetation features per plot (height percentiles, density and geometry metrics) into a feature table. |
| **ML** | Join features to ground-truth biomass and run nested CV over the model registry — ridge, lasso, elastic net, PLS, kNN, SVR, random forest, hist gradient boosting and GPR — with Optuna hyper-parameter search. |
| **Results** | Model comparison, performance and feature-importance plots, split-consistency and nested-CV stability views, and a field map you can click through into the 3D viewer. |

Plot numbering is derived from the source data rather than assigned, and
augmented copies keep a reference to their original plot, so train/test splits
are grouped by plot and stay leakage-free.

## Requirements

Python 3.12 (developed against 3.12.10).

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# .venv/bin/python -m pip install -r requirements.txt         # macOS / Linux
```

## Running

```bash
.venv/Scripts/python.exe main.py
```

## Tests

```bash
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
.venv/Scripts/python.exe -m pytest
```

The suite runs across all cores by default via `pytest-xdist`. Use `-n 0` to
run single-process when stepping through a failure in a debugger.

## Data

`Data/SGCBP/` ships the full dataset used to develop the toolkit — ground-truth
biomass tables and the LiDAR point clouds (~271 MB), so the pipeline can be run
end to end straight after cloning. The data is from a CC BY 4.0 CSIRO
collection; see [`Data/SGCBP/README.md`](Data/SGCBP/README.md) for the citation
and licence.

`presets.txt` contains ready-made configurations for the two SGCBP scan dates
(2019-08-28, stage Z31 and 2019-10-02, stage Z65).

## Licence

Source code is MIT licensed — see [`LICENSE`](LICENSE). Data under
`Data/SGCBP/` is licensed separately under CC BY 4.0.

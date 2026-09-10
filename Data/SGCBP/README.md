# SGCBP dataset

The data in this folder comes from a publicly released CSIRO data collection:

> Estavillo, Gonzalo; Anthony, Condon; Pan, Liyuan; Bull, Geoff; & Coe, Robert
> (2021): *Biomass and LiDAR data from wheat and triticale plots grown at Yanco
> (NSW) in 2019 to improve prediction of digital biomass.* v2. CSIRO. Data
> Collection. https://doi.org/10.25919/xv6v-6h56

Licensed under the
[Creative Commons Attribution 4.0 International Licence](https://creativecommons.org/licenses/by/4.0/)
(CC BY 4.0).

## What is included

The complete dataset ships with this repository — the ground-truth tables and
all the LiDAR point clouds (~271 MB). No separate download is needed to run the
toolkit end to end.

| Path | Contents |
| --- | --- |
| `original/Ground_truth_data_final.csv` | Dry-weight biomass keyed on `(runNo, rangeNo)`, both growth stages. |
| `original/train_list.txt`, `original/test_list.txt` | The published train/test split: `.pcd` path plus biomass value. |
| `original/Yanco_TC_2019_HI-pcd/` | 462 per-plot `.pcd` clouds (234 early + 228 late), one folder per driving run. |
| `reformatted/early (2019-08-28)/` | Z31 ground truth, 156 rows, plus 156 `.laz` clouds under `plots/`. |
| `reformatted/late (2019-10-02)/` | Z65 ground truth, 150 rows, plus 150 `.laz` clouds under `plots/`. |

## Changes made

As required by CC BY 4.0, the modifications are:

* **`original/`** — unmodified copies of the published files.
* **`reformatted/`** — restructured from the original: split by scan date
  (2019-08-28 / 2019-10-02), and re-keyed from `(runNo, rangeNo)` to this
  toolkit's canonical `plot(N)` names. The ground-truth CSVs gain `plot` and
  `source_laz` columns tying each row to its cloud; the clouds themselves are
  converted from `.pcd` to `.laz` and renamed to match. The biomass
  measurements are unchanged.

  This is a **subset**: only plots with matching ground truth for that growth
  stage are carried over, so 462 source clouds become 306 (156 early, 150
  late), one per row of the corresponding CSV.

This licence covers the data in this folder only. The toolkit's source code is
licensed separately — see the `LICENSE` file at the repository root.

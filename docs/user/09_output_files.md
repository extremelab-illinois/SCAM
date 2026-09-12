<!-- SPDX-License-Identifier: MIT -->
# Output files

Every CLI run writes its results to `output.path` in the case deck (default
`results/`), in the format given by `output.format` (default `csv`), unless
you pass `--no-output`. For the exhaustive column/dataset list, see
[`04_hdf5_and_csv_layout.md`](../reference/04_hdf5_and_csv_layout.md); this
page covers the practical points that table doesn't.

## CSV is the richer format

`format: hdf5` gives you one file with the scalar time history and the
temperature-profile array, which is convenient for large sweeps. But it
**does not** include char-fraction data — `char_fraction_profiles.csv` and
`comp{c}_beta_profiles.csv` are CSV-only. If you want to animate
decomposition fronts with `plot_results.py --slab`
(see [`10_plotting_results.md`](10_plotting_results.md)), use
`format: csv`.

## Depth axis: use `depth_profiles.csv`, not the profile file's own column

`temperature_profiles.csv` and `char_fraction_profiles.csv` both carry a
`depth_m` column in position 0 — but that column is always the **t=0** node
depth, frozen at simulation start. Under ALE recession
(`continuous_remap: true`), the actual node positions move every step, so
that fixed column silently drifts away from where the data really is.
`depth_profiles.csv` has the identical row/column shape but stores the
**actual** `y_nodes` at each saved snapshot — always use it as the depth
axis when animating a profile, which is exactly what `plot_results.py`
does internally.

## `NaN` padding after node drops

Under the default Lagrangian recession scheme (`continuous_remap: false`),
the node count shrinks over time as cells are dropped from the ablating
layer's back face. The profile CSVs are written as a rectangular matrix
sized to the run's largest node count, so later (shorter) snapshots are
**front-padded with `NaN`** to fill the missing rows. Any plotting or
post-processing code that reads these files directly should skip or mask
`NaN` rather than treating them as zero.

## Thermocouple columns only appear if you asked for them

`time_history.csv`'s `tc_{i}_at_{X.X}mm_K` columns (and HDF5's `/tc_temps`
dataset) only exist when `solver.tc_positions` is non-empty in the deck. An
empty list (the default) means no thermocouple columns at all, not columns
full of zeros.

## Getting `char_fraction_profiles.csv` from the Python API

The CLI always passes `stack` and `mat_cards` to `write_results`, so a CLI
run always gets `char_fraction_profiles.csv` and the per-component beta
files (for CSV output). If you call `write_csv`/`write_results` directly
from Python without `stack=`/`mat_cards=`, you'll get everything except
those two — pass them explicitly if you need char-fraction output from a
script; see [`11_python_api_tutorial.md`](11_python_api_tutorial.md).

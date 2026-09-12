<!-- SPDX-License-Identifier: MIT -->
# Output file layout

Written by `scam.io.writers.write_results`, called automatically by the CLI
(and available from the Python API). Format is `output.format` in the case
deck (`csv` default, or `hdf5`), overridable with `--format` on the CLI.

## CSV (`format: csv`, default)

All files are written under the run's `output.path` directory.

| File | Columns | Notes |
|---|---|---|
| `time_history.csv` | `time_s, T_wall_K, s_total_m, s_dot_m_s, q_cond_W_m2, m_dot_pyro_kg_m2_s, m_dot_char_kg_m2_s`, plus one `tc_{i}_at_{X.X}mm_K` column per `tc_positions` entry | Per-output-step scalars. |
| `temperature_profiles.csv` | column 0 = `depth_m` (t=0 node depth), then one `t=…s` column per snapshot | Front-padded with `NaN` after node drops (Lagrangian recession shrinks the node count). |
| `depth_profiles.csv` | same shape as `temperature_profiles.csv`, but each column is the **actual** `y_nodes` at that snapshot | Differs from `temperature_profiles.csv`'s depth column under ALE recession, where nodes move every step. **Always use this file, not the t=0 depth column, as the depth axis for profile animations.** |
| `char_fraction_profiles.csv` | same shape; β per node, `β = (ρ_virgin − ρ)/(ρ_virgin − ρ_char)`, clipped to `[0, 1]` | Written **only** when `stack` and `mat_cards` are passed to `write_results` — the CLI always passes them, so this file is present for every CLI run, but a bare Python-API call to `write_results` without `stack`/`mat_cards` omits it. |
| `comp{c}_beta_profiles.csv` | same shape; per-Arrhenius-component char fraction `β_c = (ρ_0_c − ρ_c)/(ρ_0_c − ρ_r_c)` | One file per component index `c` (0, 1, ...), up to the largest component count of any layer. Node values are volume-weighted averages over nodelets. Used by `plot_results.py --slab` to show individual decomposition fronts. |

## HDF5 (`format: hdf5`, requires `pip install -e ".[hdf5]"`)

A single file, `results.h5`, with these datasets (each carrying a `units`
attribute):

| Dataset | Shape | Units | Notes |
|---|---|---|---|
| `/time` | `(n_steps,)` | s | |
| `/T_wall` | `(n_steps,)` | K | |
| `/s_total` | `(n_steps,)` | m | |
| `/s_dot` | `(n_steps,)` | m/s | |
| `/q_cond` | `(n_steps,)` | W/m² | |
| `/m_dot_pyro` | `(n_steps,)` | kg/(m²·s) | |
| `/m_dot_char` | `(n_steps,)` | kg/(m²·s) | |
| `/tc_temps` | `(n_tc, n_steps)` | K | Only written when `tc_positions` is non-empty; carries a `tc_positions_m` attribute. |
| `/T_profiles` | `(n_steps, n_nodes)` | K | |
| `/initial_depth_m` | `(n_nodes,)` | m | t=0 node depth only. |

```{warning}
HDF5 output does **not** include the char-fraction or per-component β data
that the CSV path writes to `char_fraction_profiles.csv` and
`comp{c}_beta_profiles.csv`. If you need those fields, use `format: csv` (or
call `write_csv` directly from the Python API alongside `write_hdf5`).
```

An unrecognised `fmt` (anything other than `csv`, `hdf5`, or `h5`) raises
`ValueError` from `write_results`.

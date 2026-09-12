<!-- SPDX-License-Identifier: MIT -->
# YAML case deck schema

The case deck is the primary input format, parsed by
`scam.io.case_loader.load_case`. Only `stack` is required; every other
top-level section defaults to `{}` (its type's defaults) if omitted.

A **scalar BC** column value of "float or table" means the field accepts
either a constant number or a `[[t0, v0], [t1, v1], ...]` list, which the
loader turns into a linearly-interpolated callable `f(t)`.

```{warning}
`load_case` does not reject unrecognised keys. A typo or a field the loader
does not read (see the "not read" rows below) is silently dropped, with no
error and no warning. When in doubt, check the row for that key here rather
than the material or solver dataclass docstring — the dataclass documents a
wider surface than the loader actually parses.
```

## Top level

| Key | Required | Default | Notes |
|---|---|---|---|
| `name` | no | YAML filename stem | Printed at runtime; used as the HDF5 root name. |
| `geometry` | no | `{type: slab}` | See below. |
| `stack` | **yes** | — | `{layers: [...]}`. |
| `surface_bc` | no | `{type: energy_balance}` | See below. |
| `back_bc` | no | `{type: adiabatic}` | See below. |
| `solver` | no | `SolverOptions()` defaults | See [`03_solver_options.md`](03_solver_options.md). |
| `materials` | no | `[]` | List of material-card YAML paths, resolved relative to **the deck's own directory**, not the working directory. |
| `output` | no | `{path: results/, format: csv}` | `path` and `format` may be overridden by the CLI's `--output`/`--format`. |

## `geometry`

| Key | Type | Default | Notes |
|---|---|---|---|
| `type` | `slab` \| `hollow_cylinder` \| `tabulated` | `slab` | `slab`: constant area. `hollow_cylinder`: `A(y) = r_inner + y`. `tabulated`: arbitrary `A(y)` from `area_table`. |
| `r_inner` | float | `0.0` | Inner radius at the hot face. Only meaningful for `hollow_cylinder`. |
| `area_table` | `[[y_m, A_m2], ...]` | `None` | Only meaningful for `tabulated`. |

## `stack.layers[]`

One entry per material layer, **hot-face first** (`layers[0]` faces the
environment).

| Key | Required | Default | Notes |
|---|---|---|---|
| `material` | **yes** | — | Must match the `name` field of a card loaded via `materials`. |
| `thickness` | **yes** | — | Layer thickness [m]. |
| `n_nodes` | **yes** | — | Finite-volume node count in this layer. |
| `n_subcells` | no | `4` | Nodelets per node for the decomposition sub-grid. |
| `contact_resistance` | no | `0.0` | Thermal contact resistance at this layer's back face [m²·K/W]. |
| `grading` | no | `1.0` | Node-spacing grading ratio within the layer. |

## `surface_bc`

| Key | Type | Default | Notes |
|---|---|---|---|
| `type` | `energy_balance` \| `prescribed_flux` \| `prescribed_temp` | `energy_balance` | |
| `alpha_conv` | float or table | `0.0` | Temperature-based convection coefficient [W/(m²·K)]. Ignored once `rhoUeCH > 0`. |
| `T_aw` | float or table | `300.0` | Adiabatic wall temperature [K] for the temperature-based mode. |
| `rhoUeCH` | float or table | `0.0` | Enthalpy-based mode coefficient `ρ_e·u_e·C_H` [kg/(m²·s)]. `> 0` overrides `alpha_conv`/`T_aw`. |
| `h_r` | float or table | `0.0` | Freestream recovery enthalpy [J/kg], same reference as the material's `h_g` table. |
| `T_rad_in` | float or table | `0.0` | Incoming-radiation source temperature [K]. |
| `rho_e_u_e` | float or table | `0.0` | Freestream mass flux [kg/(m²·s)], used to form `B'_g` and to scale `m_dot_char = rho_e_u_e · C_M · B'_c`. |
| `C_M` | float or table | `0.0` | Mass-transfer Stanton number. |
| `p_e` | float or table | `101325.0` | Edge pressure [Pa] for the B′ table lookup. |
| `emissivity` | float only | `-1.0` | `-1` uses the material card's value. |
| `view_factor` | float only | `1.0` | Geometric view factor to the radiation sink. |
| `lambda_blowing` | float only | `0.5` | Blowing correction exponent. **Note:** typed as float-or-callable on the `SurfaceBCConfig` dataclass, but the loader always coerces it to `float` — a `[[t,v],...]` table here is silently truncated to its first parse, not time-varying. |
| `q_prescribed` | float or table | — | Net flux [W/m²] into the material. Only used when `type: prescribed_flux`. |
| `T_prescribed` | float or table | — | Surface temperature [K]. Only used when `type: prescribed_temp`. |

```{danger}
**`blowing_model`, `stanton_wall_correction`, and
`apply_wall_correction_to_heat` are documented on the `SurfaceBCConfig`
dataclass and set in `examples/templates/ablation2_template.yaml`
(`blowing_model: "lees"`), but `_parse_surface_bc` never reads any of the
three keys from the YAML.** Setting `blowing_model` in a deck today has no
effect — the solver silently uses the dataclass default (`"rational"`)
regardless of what the YAML says. This is a known loader gap (tracked for a
fix), not a documentation error. To use `"lees"` or `"kays"` today, construct
`SurfaceBCConfig` from Python instead of a YAML deck.
```

## `back_bc`

| Key | Type | Default | Notes |
|---|---|---|---|
| `type` | `adiabatic` \| `prescribed_temp` \| `prescribed_flux` \| `radiation` | `adiabatic` | |
| `T_back` | float or table | `300.0` | Used by `prescribed_temp`. |
| `q_back` | float or table | `0.0` | Used by `prescribed_flux`, positive into the material. |
| `emissivity_back` | float only | `0.0` | Used by `radiation`; `0` disables the radiative term. |
| `view_factor_back` | float only | `1.0` | Used by `radiation`. |
| `T_env_back` | float or table | `300.0` | Enclosure/environment temperature for `radiation`. |
| `h_back` | float or table | `0.0` | Optional convective film added to `radiation` [W/(m²·K)]. |

`radiation` requires at least one of `emissivity_back` or `h_back` to be set
in the YAML — with both at their zero defaults the loader raises
`SCAMInputError` rather than silently behaving like `adiabatic`.

## `solver`

See [`03_solver_options.md`](03_solver_options.md) for the full field table,
including the fields that exist on `SolverOptions` but are **not** parsed
from YAML at all (`array_backend`, `s_dot_prescribed`, `stanton_wall_correction`).

## `materials`

A list of material-card YAML file paths, e.g.:

```yaml
materials:
  - ../../scam/materials/ablative_organic/tacot_v3.0.yaml
```

Paths are resolved relative to **the case deck's own directory**. Every
material named in `stack.layers[].material` must appear in this list (by its
card's `name` field, not its filename), or `load_case` raises
`SCAMInputError` naming the loaded set.

## `output`

| Key | Type | Default | Notes |
|---|---|---|---|
| `path` | string | `results/` | Output directory. |
| `format` | `csv` \| `hdf5` | `csv` | `hdf5` requires `pip install -e ".[hdf5]"`. |

Both are overridable from the CLI (`--output`, `--format`) without editing
the deck; see [`03_cli_reference.md`](../user/03_cli_reference.md).

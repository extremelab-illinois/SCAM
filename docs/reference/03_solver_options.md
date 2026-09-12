<!-- SPDX-License-Identifier: MIT -->
# `SolverOptions` reference

`SolverOptions` (`scam/config/solver.py`) configures the numerical solver:
timestep limits, SEB convergence, output interval, thermocouple positions,
and the physics-model switches (element transport, ALE recession, live
Cantera chemistry). The YAML `solver:` section maps onto a subset of its
fields; the table below marks which are actually read by
`case_loader.py::_parse_solver`.

| Field | YAML? | Type | Default | Meaning |
|---|:---:|---|---|---|
| `t_start` | ✅ | float | `0.0` | Simulation start time [s]. |
| `t_end` | ✅ | float | `10.0` | Simulation end time [s]. |
| `dt_init` | ✅ | float | `1e-3` | Initial timestep [s]. |
| `dt_min` | ✅ | float | `1e-7` | Minimum allowed timestep [s]. |
| `dt_max` | ✅ | float | `1.0` | Maximum allowed timestep [s]. |
| `dt_max_dT` | ✅ | float | `50.0` | Max `\|ΔT\|` per step in any cell [K], an adaptive-timestep criterion. |
| `dt_max_drho_frac` | ✅ | float | `0.05` | Max `\|Δρ/ρ\|` per step in any nodelet [-], an adaptive-timestep criterion. |
| `max_seb_iter` | ✅ | int | `50` | Max Newton iterations for the surface energy balance. |
| `seb_tol` | ✅ | float | `1.0` | SEB residual convergence tolerance [W/m²]. |
| `output_dt` | ✅ | float | `0.1` | Interval between saved snapshots [s]. |
| `node_drop_threshold` | ✅ | float | `0.1` | Trigger a node drop when the shrinking surface cell's thickness falls below this fraction of the nominal spacing. |
| `tc_positions` | ✅ | list[float] | `[]` | Thermocouple probe depths [m] from the original (t=0) front face; interpolated at each output step. |
| `allow_recession` | ✅ | bool | `True` | Set `False` to suppress surface recession (in-depth pyrolysis only — pyrolysis gas then exits via Darcy flow rather than driving recession). |
| `use_rho_old` | ✅ | bool | `True` | Energy-storage formulation. `True`: exact `d(ρh)/dt` with `h_old` at pre-decomposition `ρ_old` (implicitly carries the sensible-enthalpy term of decomposition). `False`: PATO's `ρ·dh/dt ≈ ρ·cp·dT/dt` approximation (no density-change term). See [`04_energy_formulation_comparison.md`](../theory/04_energy_formulation_comparison.md). |
| `element_transport` | ✅ | bool | `False` | Enable the `Z_elem` (C/H/O/N) transport PDE. Requires `pyro_elem_fracs` on every ablating layer's material card. |
| `bprime_runtime` | ✅ | bool | `False` | Use live Cantera B′ evaluation instead of the pre-computed YAML table. Only active together with `element_transport=True` and a configured `BprimeEvaluator`. Adds ~10–100 ms per SEB Newton call — reference runs only. |
| `max_picard` | ✅ | int | `8` | Max Picard (fixed-point) iterations on the in-depth tridiagonal solve per timestep. `1` reproduces the old single-pass behaviour. |
| `picard_tol` | ✅ | float | `0.05` | Picard convergence criterion: max `\|T_new − T_k\|` [K]. |
| `continuous_remap` | ✅ | bool | `False` | Recession mesh scheme. `False`: Lagrangian fixed grid — the surface cell shrinks and is dropped/merged one cell at a time, which puts a small (~±6 K) sawtooth on `T_wall`. `True`: continuous ALE moving mesh — nodes are redistributed every step between the receding surface and the fixed back face with conservative re-interpolation; node count stays fixed, `T_wall` is smooth. ALE also needs (and gets, automatically) the recession-CFL limit `dt ≤ 0.1·h/s_dot`. See [`08_recession_and_mesh_motion.md`](../theory/08_recession_and_mesh_motion.md). |
| `array_backend` | ❌ **Python API only** | str | `"numpy"` | Experimental kernel backend; `"jax"` requires `pip install -e ".[jax]"`. Not in `_parse_solver`'s key list — set it by constructing `SolverOptions` from Python. |
| `s_dot_prescribed` | ❌ **Python API only** | float \| None | `None` | Prescribe a fixed surface recession rate [m/s], overriding the surface-mass-balance-computed `s_dot`. Used for verification cases with an analytically known recession rate. Not parseable from the YAML deck today. |

```{note}
Every field above with a ✅ can be set under the deck's `solver:` section.
The two ❌ rows are real dataclass fields with no YAML path — construct
`SolverOptions(...)` directly if you need them (see
[`11_python_api_tutorial.md`](../user/11_python_api_tutorial.md)).
```

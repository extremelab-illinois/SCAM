<!-- SPDX-License-Identifier: MIT -->
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See `.claude/skills/handoff/SKILL.md` for handoff procedures.

## Commands

```bash
# Install in editable mode (required before first use)
pip install -e ".[dev]"

# For HDF5 output support
pip install -e ".[hdf5]"

# For live Cantera surface chemistry
pip install -e ".[bprime]"

# Run all tests
~/.local/bin/pytest tests/ -q

# Run a single test file
~/.local/bin/pytest tests/unit/test_decomposition.py -v

# Run a single test by name
~/.local/bin/pytest tests/unit/test_tridiagonal.py::test_thomas_simple -v

# Run examples (must use Agg backend to avoid blocking on plt.show())
MPLBACKEND=Agg python3 examples/other_examples/flat_plate_no_ablation.py
MPLBACKEND=Agg python3 examples/other_examples/full_ablation_tacot.py

# CLI entry point
python3 -m scam examples/other_examples/tacot_arcjet.yaml

# Run annotated input-deck templates (examples/templates/)
python3 -m scam examples/templates/tacot_arcjet_template.yaml
python3 -m scam examples/templates/ablation1_template.yaml
python3 -m scam examples/templates/ablation2_template.yaml

# Plot results from any SCAM CSV output directory
MPLBACKEND=Agg python3 examples/templates/plot_results.py results/ablation2_template/
MPLBACKEND=Agg python3 examples/templates/plot_results.py results/ablation2_template/ --video --speed 30 --beta 0.5
MPLBACKEND=Agg python3 examples/templates/plot_results.py results/ablation2_template/ --slab --speed 30
```

## Architecture

SCAM is a 1-D Backward Euler FVM ablation solver (CMA-style). Each timestep follows a fixed sequence defined in `solvers/indepth_solver.py::step()`:

1. **Decomposition** — Arrhenius ODE integrated analytically per nodelet (`physics/decomposition.py`)
2. **Pyrolysis gas flux** — accumulated from in-depth density change (`physics/pyrolysis_gas.py`)
3. **Tridiagonal assembly** — FVM coefficients A/B/C/D with conductance coupling (`numerics/assembly.py`)
4. **F_cond reduction** — backward elimination reduces the system to `q_cond = α_F·T_w + β_F`; this linear relationship is what the SEB Newton solver uses (`numerics/assembly.py::compute_F_cond`)
5. **SEB Newton solve** — scalar Newton iterate on T_wall; F_cond is the coupling to in-depth (`solvers/surface_solver.py`)
6. **Tridiagonal solve** — Thomas algorithm with q_cond inserted at surface node (`numerics/tridiagonal.py`)

The outer time loop lives in `solvers/material_response.py::run()`, which also handles recession (`mesh/receding.py`), node drops (`mesh/remap.py`), and adaptive timestep (`numerics/time_integration.py`).

### Critical numerical details

**F_cond sign** — The backward elimination in `assembly.py::compute_F_cond` uses `D_sweep[n-1] -= factor * D_sweep[n]` (MINUS). The formula is `D_eff[n-1] = D[n-1] - C[n-1]·D[n]/B[n]`; since `factor = C[n-1]/B[n]` is negative (C is negative), using `+=` silently subtracts and gives a wrong beta_F. This was a critical bug that caused Newton non-convergence.

**Nodelet subgrid** — Each thermal node contains J nodelets (default 4) for decomposition. Nodelet temperatures are interpolated using cumulative *volume* as the coordinate (not distance) — critical for cylindrical geometry. Node density is the volume-weighted average of nodelet densities. **Nodal density formula**: `rho = fiber + sum_i(vol_avg(rho_i))` where `fiber = rho_char − sum_i(rho_r_i)`. Do NOT use `rho_char + sum(rho_0_i)` — that double-counts the residual density when any `rho_r > 0` (e.g. TACOT v2.2 has `sum(rho_r)=60 kg/m³`, giving 340 vs correct 280 kg/m³ initially). TACOT v3.0 has all `rho_r=0` so the old formula was accidentally correct there. See `physics/decomposition.py::nodal_density_from_nodelets` and `mesh/grid.py`.

**Convective correction in ALE mode** — `update_nodelet_densities` in `physics/decomposition.py` accepts `s_dot` to apply an Eulerian convective correction `v·∂ρ/∂y` converting Lagrangian density rates to fixed-y rates. In the **discrete Lagrangian scheme** (`continuous_remap=False`) nodelet positions are fixed, so this correction is needed. In the **ALE scheme** (`continuous_remap=True`), `apply_recession_ale` already re-interpolates `rho_components` onto new nodelet positions at the end of every step, handling the advective transport. Passing `s_dot` to the correction on top of the ALE re-interpolation double-counts the same advection. Fix: `indepth_solver.py` passes `s_dot_decomp = 0.0 if options.continuous_remap else s_dot_prev`. This ensures `drho_dt_y` and the density update use the pure Lagrangian decomposition rate in ALE mode, which is what pyrolysis gas flux and energy source should use (true mass conversion rate). Without the fix: m_dot_g is artificially 2.5% high during heating and shows a step-drop at t=60s when s_dot→0.

**Node drop direction** — Nodes are always dropped from the BACK face of the ablating layer (deeper into material), not the front. The merging is conservative: volume-weighted average of temperature and density.

**Recession scheme** — Two options (see `config/solver.py::SolverOptions.continuous_remap`). Default (`False`) is the Lagrangian fixed grid: the surface cell shrinks and is dropped/merged one cell at a time — discrete, and it puts a ~±6 K sawtooth on `T_wall` (each drop exposes the next, cooler node). `True` selects a continuous moving-mesh (ALE) scheme (`mesh/receding.py::apply_recession_ale`): the ablating layer's nodes are redistributed between the receding surface and the fixed back face every step with conservative field re-interpolation — node count fixed, no drops, smooth `T_wall`. ALE additionally needs the recession-CFL limit in `time_integration.py` (`dt ≤ 0.1·h/s_dot`) to avoid re-interpolation wobble. See `docs/verification/pato_validation.md`.

**Blowing on m_dot_char** — The blowing factor `1/(1+λ·B_total)` reduces BOTH `q_conv` AND `m_dot_char` (Reynolds analogy: C_H and C_M drop together). `B_total = B'_c + B'_g` is formed on the unblown basis so the algebraic correction stays bounded under heavy injection. See `physics/surface_energy.py::seb_residual` and `docs/verification/pato_validation.md`.

**SEB advective terms** — PATO's `Bprime` BC includes the advective enthalpy of the ablating mass: `qAdvPyro = mDotGw·(h_g − h_w)` and `qAdvChar = mDotCw·(h_c − h_w)` (the char term is the surface oxidation heat release). `seb_residual` adds these (`q_adv`) when the chemistry backend exposes `surface_enthalpies(T_w,p,Z_C_pyro)→(h_g,h_c)` on the same reference as `h_wall`. This is available from the live `BprimeEvaluator` and from Cantera-generated B′ tables carrying optional pretabulated `h_g`/`h_c`; legacy tables omit the term. Closed ~40 K of the ablation2 surface gap. See `docs/verification/pato_validation.md` §8.

**Pyrolysis gas `h_g` enthalpy reference — `h_g_abs_offset`** — YAML material cards store the pyrolysis-gas enthalpy table on the **sensible** reference (`h_g(298 K) ≈ 0`). PATO's gasProperties file and Cantera use the **absolute** (formation-enthalpy) reference (`h_g(298 K) ≈ −7.09 MJ/kg` for TACOT). The optional `MaterialCard.h_g_abs_offset` field (e.g. `−7.093e6` J/kg for TACOT) converts sensible→absolute. `numerics/assembly.py::_build_system` applies the offset to every node's `h_g_cache` when it is not `None`. The surface node gas outflow is explicit (`mg_out[0] = m_dot_g[0]`); combined with SEB `q_adv_pyro = m_dot_g·(h_g_abs(T_w) − h_wall)`, the net surface gas energy ≈ `−m_dot_g·h_wall` (PATO-consistent). Scripts that replace `h_g_table` with already-absolute data (PATO gasProperties or Cantera) **must** also clear `h_g_abs_offset=None` in the `dataclasses.replace()` call to prevent double-application. See `docs/verification/pato_validation.md` §11.

**Cooldown SEB reference consistency** — In PATO 2.x, cooldown maps `rhoUeCH = 0.3e-2` kg/m²/s and `h_r=0`, but `chemistryOn=0` disables the B′ lookup and the Bprime temperature BC ignores `rhoUeCH/h_r`; it uses `hconv*(Tedge-T)` instead. When the B′ lookup is skipped, `h_wall` falls back to `enthalpy_char` (298-ref) instead of the equilibrium wall enthalpy. `q_adv` is therefore gated on `_bprime_ran` (only applied when `h_wall` is on the same reference as `h_g`/`h_c`) — otherwise the `h_g − h_wall` reference mismatch injects a large spurious cooling sink during cooldown. The enthalpy-BC branch uses B′ `h_wall` only when `_bprime_ran`; if a Bprime backend exists but chemistry is off, `seb_residual` falls back to the temperature branch with no advective or mass-removal sink. See `physics/surface_energy.py::seb_residual`.

**Coordinate convention** — `MeshState.y_nodes` is measured from the ORIGINAL front face (surface node at `y_nodes[0] == s_total`; back face fixed). Depth below the current surface is `y_nodes − y_nodes[0]`. Never add `s_total` to `y_nodes`.

**Element transport flux** — `Z_elem` transport (`physics/element_transport.py`) must advect with the pyrolysis gas flux `m_dot_g_nodes` (mass-consistent with its `−dρ/dt` source), NOT the Darcy expansion flux — otherwise each `Z_i` diverges to 1.0. Columns are renormalised to sum to 1 in `indepth_solver.py`.

**Energy storage — `ρ·h(T)` formulation** — The tridiagonal RHS uses sensible enthalpy `h(T) = ∫₀ᵀ cp dT'` rather than `cp·T`. `h_virgin_sensible` / `h_char_sensible` tables are built by `build_sensible_enthalpy_table` in `properties.py` and stored in `MaterialCard` (built from cp tables on load; or on-the-fly in tests). `mixture_enthalpy_array` evaluates the blended mixture `h(T,ρ)` node-wise. Sensible enthalpy interpolation uses **linear extrapolation** (`_interp_linear_extrap`) beyond the table range — flat `np.interp` extrapolation broke the Picard invariant when temperatures exceeded the table range (caused ~50% error before this fix).

**Picard iteration** — `indepth_solver.py` wraps the tridiagonal solve in a fixed-point loop (`SolverOptions.max_picard=8`, `picard_tol=0.05 K`). Each iteration reassembles with `cp(T^k)` at the current iterate; `h_old` is fixed throughout. Converges in ≤ 2 iterations for constant-cp inert cases; more for strong cp(T) variation. `max_picard=1` reproduces old single-pass behaviour. `PRESCRIBED_TEMP` BC bypasses the loop.

**`h_old` at `ρ_old` — exact FVM storage** — `h_old_arr` in `indepth_solver.py` is computed as `mixture_enthalpy_array(state.T, state.rho, ...)` using **pre-decomposition** density `ρ_old`. In `_build_system`: `Dc_thermal = M·T^k + (ρ_old·h_old − ρ_new·h_k)·A·Δ/dt`. This is the exact FVM conservation `(ρ_new·h_new − ρ_old·h_old)/dt = fluxes + Q_chemical`. The density-change term `(ρ_old−ρ_new)·h_old/dt` naturally provides the implicit `π·h̄_sensible` contribution (sensible enthalpy carried away by decomposing mass). Q_vol therefore carries only `h_bar_chemical` (= `h_bar_absolute − h̄_sensible`) for table materials, or `h_decomp` (pure chemical `hp`) for tacot-style materials. `h̄_sensible` for tacot-style materials was previously missing from the energy balance; it is now captured implicitly. `properties.py` provides `h_bar_chemical_array` and `h_bar_sensible_array` helpers. **`SolverOptions.use_rho_old`** (default `True`) is now wired: `False` passes `h_old=None, T_old_rhs=state.T` to the assembly, giving `Dc_thermal = M·T^n` (PATO `ρ·dh/dt` approximation, Picard-stable because T^n is fixed). Verified mean density comparison vs PATO massLoss: exact `d(ρh)/dt` undershoots by 0.76 kg/m³ at t=60 s but converges to 0.08 kg/m³ by t=120 s; the `ρ·dh/dt` diagnostic matches at t=60 s (+0.006) but diverges by +0.99 kg/m³ during cooldown. Production ablation2 comparisons use exact storage; `compare_pato_ablation2_beta.py` is the diagnostic for the storage approximation. See `docs/verification/pato_validation.md` §10.

**Pyrolysis energy (`h_bar` = PATO's `hs`)** — PATO's `Pyrolysis` energy model (what `AblationTestCase_2.x*` use) has `pyrolysisFlux = −π·(hs + hp)`: `hs` = virgin/char enthalpy difference (== SCAM's `h_bar`), `hp` = per-reaction heat of pyrolysis (TACOT: −4 MJ/kg). SCAM splits the same energy: the `ρ_old·h_old` storage term implicitly carries `h̄_sensible` (the `hs` piece); `Q_vol` carries only `h_bar_chemical = h_bar_absolute − h̄_sensible` for materials with explicit enthalpy tables, and `h_decomp` (pure chemical `hp`) for materials without explicit enthalpy tables. Total in-depth decomposition energy = `h̄_sensible` (implicit) + `h_bar_chemical` (explicit) = `h_bar_absolute`. TACOT 3.0 uses explicit virgin/char enthalpy tables, so it follows the `h_bar_chemical` path. See `docs/verification/pato_validation.md` §7.

**Performance-sensitive assembly path** — `numerics/assembly.py` intentionally evaluates `k`, `cp`, `h_bar`, and gas enthalpy by material/layer using vectorized property helpers. `compute_F_cond` returns `(alpha_F, beta_F, TridiagSystem)` so `indepth_solver.py` can reuse the pre-built system and only patch `D[0]` after the SEB solve. `physics/pyrolysis_gas.py` uses reverse cumulative sums for `m_dot_g_nodes`, and `physics/darcy_flow.py` vectorizes constant-pressure gas expansion. `physics/element_transport.py` pre-computes `storage` and the `Z_pyro (4,N)` matrix layer-wise before the element loop, assembles the tridiagonal via NumPy face-flux operations, and uses `np.where` for degenerate nodes — the former per-node Python loop (4×N scalar calls per timestep) was the dominant cost for equilibriumElementConservation cases. Avoid reintroducing per-node property loops or duplicate assembly/gas-flux passes.

**Irreversibility** — Density never increases: `ρ_new = max(min(ρ_new, ρ_old), ρ_residual)` is enforced unconditionally after every Arrhenius step.

### Module dependency order (strict, no upward imports)

```text
core/constants, core/errors
  → config/*
    → core/state, core/results
      → geometry/*, mesh/*, io/
        → physics/*
          → numerics/*
            → solvers/*
              → cli/
```

### Key dataclasses

- `SimState` (`core/state.py`) — nodal T, nodal ρ, s_dot, T_wall, q_cond, mesh reference
- `MeshState` (`core/state.py`) — y_nodes, delta_nodes, layer_id, s_total, nodelet geometry
- `SimCase` (`config/case.py`) — top-level simulation case composed of `StackConfig`, `GeometryConfig`, `SurfaceBCConfig`, `BackBCConfig`; parsed by `io/case_loader.py`
- `StackConfig` / `LayerConfig` (`config/stack.py`) — ordered list of layers; `layers[0]` is the hot face. Each `LayerConfig` holds `material_name`, `thickness`, `n_nodes`, `n_subcells`, and optional `contact_resistance`.
- `SurfaceBCConfig` / `BackBCConfig` (`config/boundary.py`) — boundary conditions; scalar parameters accept either a `float` or a callable `f(t) -> float`. `eval_bc(param, t)` resolves either form.
- `GeometryConfig` (`config/geometry.py`) — area function A(y): `SLAB` (flat plate), `HOLLOW_CYLINDER` (r_inner + y), or `TABULATED`.
- `MaterialCard` / `ComponentCard` (`config/material.py`) — all material properties; loaded from YAML by `io/material_loader.py`. Surface emissivity is resolved by `properties.py::surface_emissivity(mat, T, rho)`: optional `emissivity_virgin_table`/`emissivity_char_table` (`[[T,ε],…]`) give temperature dependence, else the scalar `emissivity`/`emissivity_char`; the virgin and char values are blended by the local char fraction (so a charred surface emits at the char value).
- `SolverOptions` (`config/solver.py`) — timestep limits, SEB convergence, output interval, TC positions, `element_transport` (Z_elem PDE on/off), `continuous_remap` (ALE vs Lagrangian recession), `bprime_runtime` (live Cantera B′ backend)
- `Results` (`core/results.py`) — time-series snapshots accumulated by `material_response.py`

### I/O

`io/case_loader.py` parses YAML input decks into `SimCase`. The private helper `_make_scalar_bc(raw_val)` converts a raw YAML value to the `ScalarBC` type: a plain number becomes a `float`; a `[[t, v], ...]` list becomes a `lambda t: float(np.interp(t, ts, vs))` callable. Every BC field in `_parse_surface_bc` and `_parse_back_bc` goes through it, so time-varying BCs can be specified directly as tables in the YAML deck with no Python code required.

`io/writers.py` handles CSV and optional HDF5 output (requires `h5py`; install with `pip install -e ".[hdf5]"`). CSV output produces four files:

- `time_history.csv` — per-step scalars (time, T_wall, s_total, s_dot, q_cond, m_dot_pyro, m_dot_char, TC temperatures)
- `temperature_profiles.csv` — T(y) at each snapshot; first column is the initial node depth [m]
- `depth_profiles.csv` — actual `y_nodes` [m] at each snapshot; differs from the temperature-profile depth column under ALE recession (where nodes move every step). **Always use this file as the depth axis for profile animations.**
- `char_fraction_profiles.csv` — β(y,t) = (ρ_v − ρ)/(ρ_v − ρ_c) ∈ [0,1] at each snapshot; written when `stack`+`mat_cards` provided (automatic via CLI).
- `comp{c}_beta_profiles.csv` — per-Arrhenius-component char fraction β_c(y,t) = (ρ_0_c − ρ_c)/(ρ_0_c − ρ_r_c), one file per component index. Node densities are volume-weighted averages over nodelets. Used by `--slab` animation to show individual decomposition fronts.

### Templates and plotter

`examples/templates/` contains three heavily annotated YAML input decks and a standalone plotter:

- `tacot_arcjet_template.yaml` — full ablation, temperature-based convection with tabulated `alpha_conv`, 51 nodes, 120 s
- `ablation1_template.yaml` — prescribed-T surface, no recession, 501 nodes; PATO AblationTestCase_1.x equivalent
- `ablation2_template.yaml` — full SEB + B′ chemistry + ALE recession, enthalpy-based convection; PATO AblationTestCase_2.x equivalent
- `plot_results.py` — reads any SCAM CSV results directory and produces a static 3-panel PNG and optional animations. Two animation modes:
  - `--video` — 1-D temperature profile sweeping with depth; red dashed = receding surface, blue dashed = char-fraction isoline at `--beta LEVEL` (default 0.5)
  - `--slab` — 2-D slab visualization: the 1-D profiles are extruded to a thin slab (width = total_depth/4) and rendered as coloured `imshow` panels: Temperature (plasma), β total (tan→char-brown), β per Arrhenius component (Blues, Oranges, …). Ablated zone shown in light gray. Requires `comp{c}_beta_profiles.csv` (written automatically by the CLI alongside `char_fraction_profiles.csv`).

  Both modes accept `--fps N` and `--speed N` (sim-seconds per real-second). Output is mp4 if ffmpeg is available, otherwise gif.

### Blowing correction

`physics/blowing.py` reduces the effective convective Stanton number when pyrolysis gas and char products are injected: `α_eff = α_conv / (1 + λ·B')`. `λ` defaults to 0.5 (laminar). The individual B' components (`B'_g`, `B'_c`) feed into both the blowing correction and the B' table chemistry lookup.

### Surface chemistry backends (B' table vs live Cantera)

The surface solver needs an object with a `lookup(T_wall, p_e, B'_g, Z_C_pyro=None) -> (B'_c, h_wall)` method; two interchangeable backends implement it and are passed in the `b_prime_tables` dict (keyed by material name):

- **`BPrimeTable`** (defined in `io/material_loader.py`, re-exported by `physics/chemistry.py`) — a `RegularGridInterpolator` over a pre-computed YAML table. TACOT 3.0 uses `scam/materials/ablative_organic/tacot_v3.0_bprime_air.yaml`; element-transport cases use live Cantera with the transported `Z_C_pyro`. Tables are generated by `scam/tools/generate_bprime.py` (run with `python3 scam/tools/generate_bprime.py --help`). Fast; no Cantera at runtime.
- **`BprimeEvaluator`** (`physics/bprime_evaluator.py`) — live Cantera equilibrium, a drop-in for `BPrimeTable` (same `lookup`). Build with `BprimeEvaluator.from_config("scam/materials/ablative_organic/tacot_v3.0_bprime_config.yaml")`. Warm-started (~0.3 ms/call in the current TACOT verification path; about 3–4 s total for ~11.6k lookups). Requires Cantera (`pip install -e ".[bprime]"`). **Only** this backend exposes `surface_enthalpies(...)`, which activates the SEB advective terms (see above). `examples/verification/ablation2/compare_pato_ablation2.py` uses it for live chemistry modes; use `chemistry_mode="table"` for the explicit table-only diagnostic. **CRITICAL:** `_TACOT_PYRO_X_NOMINAL` and `tacot_v3.0_bprime_config.yaml` must use `pyro_y: "CH4:0.5551,CO:0.2418,H2O:0.2031"` — this is the ONLY composition that matches PATO's tacot26 elemental fracs (C:0.206, H:0.679, O:0.115). The old preset `CH4:0.4600,...` gave C:0.227/H:0.622/O:0.151 and over-predicted B'_c by 15–140% depending on B'_g. TACOT v2.2 and v3.0 share identical pyro gas composition; `tacot_v2.2_bprime_config.yaml` is therefore equivalent.

### Material data

TACOT 3.0 properties for the PATO verification cases are in `scam/materials/ablative_organic/tacot_v3.0.yaml`; `tacot_v3.0_3rxn.yaml` keeps the exact 3-reaction kinetics variant. TACOT 2.2 (Goldstein kinetics) is in `tacot_v2.2.yaml`.

B′ companion file naming convention for TACOT (and similarly for other materials):

- `*_bprime_from_refXLS.TSV` — raw copy-paste from the reference XLS dataset
- `*_bprime_from_refXLS.yaml` — same data converted to SCAM `BPrimeTable` format
- `*_bprime_air.yaml` — Cantera-generated pre-computed table used in simulations
- `*_bprime_config.yaml` — config for `generate_bprime.py` to regenerate the Cantera B′ table

TACOT v2.2 and v3.0 share the same pyrolysis gas elemental composition (C:0.206, H:0.679, O:0.115); their bprime configs and `_bprime_air.yaml` files are chemically equivalent. The XLS reference tables differ because v2.2 was computed with Mutation++ / CEA thermodynamic database (25-species, 1 atm only) while v3.0 used Mutation++ at 4 pressure levels.

Materials are organized into subdirectories by class: `ablative_organic/` (TACOT, PICA, AVCOAT, HEEET, cork, carbon phenolic, ASTERM, ZURAM, NorCoat Liège), `ablative_carbon/` (FiberForm), `ablative_hybrid/` (SLA-561V), `ablative_silicone/` (RTV560, SIRCA, SLA-220), `ablative_silica/` (see README), `subsurface/` (see README). Each ablative material may have a companion `*_bprime_air.yaml` B′ table and `*_bprime_config.yaml` for the live Cantera backend. `material_loader.py` resolves companion paths relative to the material file's directory.

**Versioning fields** (metadata only, not used by solver): every card carries `card_version` (SCAM-internal revision, start `"1.0"`), `status` (`verified` | `provisional` | `estimate`), and optionally `dataset_version` (upstream release, e.g. `"3.0"` for TACOT 3.0 — omitted when no single authoritative version exists). **Naming convention**: `{material}_v{X}.yaml` when multiple dataset versions coexist; `_{variant}` suffix for model variants of the same dataset (e.g. `tacot_v3.0_3rxn.yaml`). Full inventory: `scam/materials/MATERIALS.md`.

### Additional physics modules

- `physics/surface_energy.py` — SEB residual and Jacobian for the Newton solve
- `physics/recession.py` — surface recession rate from the surface mass balance (SMB)
- `physics/charring_energy.py` — in-depth chemical energy source `Q_decomp = -(dρ/dt)·h̄(T)`
- `physics/gas_enthalpy.py` — linearly interpolated pyrolysis gas enthalpy `h_g(T)` from material table
- `physics/darcy_flow.py` — gas expansion flux and energy source for porous materials with `gas_porosity > 0`
- `physics/pressure_darcy.py` — pressure-driven Darcy flow: solves 1-D gas pressure field and returns gas mass flux + energy source. Key entry points: `solve_pressure_and_flux(...)` (combined call; preferred) and `pressure_darcy_energy_source(...)`. Permeability at each node is computed by `_node_permeability()`: blends virgin (`permeability_virgin`) and char (`permeability`) values by local virgin fraction, then applies the Klinkenberg slip correction `K_app = K·(1 + klinkenberg_b/p)` when `klinkenberg_b > 0`. Both new fields are in `MaterialCard` and loaded from YAML; TACOT v3 carries estimated values (`permeability_virgin=1.6e-11`, `permeability=2.0e-11`, `klinkenberg_b=6000 Pa`). **Pressure-aware gas properties (stage 1):** when `MaterialCard.gas_properties_pT` is set (full PATO gasProperties (p,T) table: axes `p`/`T` + `M`/`h_g`/`mu` arrays; bilinear `properties.py::interp_gas_pT`, shared with the 2D path), μ and M use bilinear (T,p) lookups and ρ_g uses the local solved pressure; otherwise the legacy Sutherland-air μ / scalar `gas_molar_mass` path is used and results are bit-identical. The surface-face flux gradient now uses the passed `p_surface` (was a hardcoded 101325 Pa — only visible for sub-atmospheric runs). **Stage 2 (h_g):** `gas_enthalpy.py::pyrolysis_gas_enthalpy_abs(mat, T, p=None)` is the single reference-safe entry point for absolute h_g — pT table → bilinear h_g(T,p) with NO `h_g_abs_offset` re-application (the table is already absolute; double-application trap, pato_validation.md §11); else sensible table + offset (legacy, bit-identical). Used by the 1D assembly `h_g_cache` and gas-storage term (at the card's ambient `gas_pressure` — consistent with how those terms treat ρ_g) and by the SEB `q_adv` at `(T_w, p_e)` (PATO's boundary h_g field). `pressure_darcy_energy_source` transports `h_abs(T,p_local) − h0` when the table is present. Acceptance: pT 1-atm slice matches card `h_g_table+offset` to 0.24 %, and ablation2 (1 atm) with the table injected shifts T_wall by < ~1 K.

**`gas_properties_pT` is loaded routinely from the material card.** `io/material_loader.py::_load_gas_properties_pT` reads an optional `gas_properties_pT: <file>.yaml` companion-file key (same resolution convention as `b_prime_table`: relative to the card's directory) into the dict `{"p", "T", "M", "h_g", "mu"}` consumed by `interp_gas_pT`. `tacot_v3.0.yaml` carries `gas_properties_pT: tacot_v3.0_gasProperties_pT.yaml` — a repo-committed conversion of PATO's `data/Materials/Composites/TACOT/gasProperties` (5 pressure levels × 152 temperatures), so it works without the PATO-dev tree present. `tacot_v3.0_3rxn.yaml` and `tacot_v2.2.yaml` do NOT carry it (they also lack the 1-atm `gas_molar_mass_table`/`gas_viscosity_table` slices — no regression risk since their behavior is unchanged either way). Verified safe for every consumer of `tacot_v3.0.yaml`: full test suite unchanged (504 passed); `test_tacot_benchmark.py` (sub-atmospheric, p_e=10 kPa) shifts T_wall by 0.6 K and stays within its plausibility bounds; ablation1 (prescribed-T, no SEB) TC probes shift ≤0.5 K except the deepest (24 mm) at 6 K — small, physically expected, and none of these are regression-pinned to exact values.
- `numerics/nonlinear.py` — scalar Newton and bisection solvers (used by `surface_solver.py`)
- `geometry/area.py` — area/conductance helpers: `build_area_function`, harmonic conductivity, interface conductance, nodelet cumulative volume (used by `numerics/assembly.py` and `mesh/`)
- `geometry/fvm.py` — FVM face-area averaging used by the assembly to handle non-slab geometries

### Test layout

- `tests/unit/` — single-function correctness checks (geometry, Thomas algorithm, Arrhenius, properties)
- `tests/integration/` — physics-level: erfc analytical solution, energy conservation drift
- `tests/verification/` — analytical verification cases V1–V5: constant/variable conductivity conduction, multilayer, prescribed-T decomposition, surface balance
- `tests/validation/` — full TACOT arcjet run with `scope="module"` fixture (runs once per session); bounds are physics-calibrated, not arbitrary

### PATO verification & docs

PATO comparison scripts live in `examples/verification/ablation1/` and `examples/verification/ablation2/` (helpers in `examples/verification/ablation2/_pato2_common.py`; PATO reference data under the local `pato-3.1` tree). Run with `MPLBACKEND=Agg`; each saves a multi-panel PNG alongside the script. The `ablation2*` cases are the full SEB/recession regime. When changing surface-energy, recession, or pyrolysis physics, regenerate these and check against the references.

Ablation1 variants: `_grading` (approximates depth-varying density via a multilayer stack), `_multiPorousMat` (pyrolyzing TACOT porous layer over inert cork; PATO `porousMat1` uses `PyrolysisType LinearArrhenius`, so do not model it as inert char), `_function` (time-varying BC via Python callables), `_equilibriumElementConservation` (element transport with 4-D B′ table). All ablation1 scripts load material from `scam/materials/ablative_organic/tacot_v3.0.yaml` via `load_material()`; no hardcoded material tables. The base script is `compare_pato_ablation1.py` (PATO AblationTestCase_1.0, prescribed surface T, no ablation). Ablation2 variants: `_multiMat` (3-layer TACOT + inert sublayers), `_chemistryOff` (no B′ lookup), `_equilibriumElementConservation` (Cantera + element transport), `_beta` (diagnostic: per-component β_i and bulk β vs depth profiles + mean density vs time comparing exact `d(ρh)/dt` storage vs PATO-style `ρ·dh/dt` approximation; uses `SolverOptions.use_rho_old=False` for the latter). `compare_tacot_v3_vs_3rxn.py` — kinetics sensitivity study: runs TACOT v3.0 2-rxn and 3-rxn on the ablation2 setup; confirms they differ by < 4.5 K surface and < 0.004 mm recession (h_bar is flat with T so timing differences in dρ/dt wash out).

Conduction verification scripts are in `examples/verification/conduction/` (V1–V3). Pyropy kinetics verification scripts are in `examples/verification/pyropy/`; pyropy is developed by Francisco Torres Herrador ([github.com/Fratorhe/pyropy](https://github.com/Fratorhe/pyropy)) and is a separate package — not a SCAM dependency. Run `python3 examples/clean_outputs.py` to remove all generated PNGs and results directories.

Key design docs (read before touching the matched physics):

- `docs/verification/pato_validation.md` — the SCAM↔PATO fixes term by term (blowing, emissivity, h_bar/pyrolysis energy §7, SEB advective terms §8, recession), with the residual-gap analysis. Numbered sections are referenced from the code comments.
- `docs/verification/verification.md` — the V1–V5 verification ladder rationale and per-rung criteria.
- `docs/reference/pato_material_recession_summary.md`, `docs/reference/pato_receding_pyrolyzing_internal_energy_balance.md` — extracted PATO formulations (recession `s_dot=ṁ_char/ρ_s`; the `Pyrolysis` energy model the `h_bar` term must match). SCAM is verified consistent with these.
- `docs/physics/energy_formulation_comparison.md` — side-by-side comparison of Amar (2006), PATO, and SCAM energy equations; explains why `h_decomp` and enthalpy fields cannot be transplanted between codes without conversion.
- `docs/physics/tacot_kinetics_2rxn_vs_3rxn.md` — derivation of PATO's 3-reaction kinetics (from `constantProperties`) and the correct merge to 2 components (`tacot_v3.0.yaml`); explains the 9× rate error that results from misinterpreting the inert 60 kg/m³ matrix residual as a pyrolysis product.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

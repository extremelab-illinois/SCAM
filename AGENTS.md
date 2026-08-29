<!-- SPDX-License-Identifier: MIT -->
# CLAUDE.md

This file provides guidance to Codex when working with code in this repository.

## Commands

```bash
# Install in editable mode (required before first use)
pip install -e ".[dev]"

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

# Regenerate B' tables (must use module form, not direct script path)
python3 -m scam.tools.generate_bprime scam/materials/ablative_carbon/calcarb_bprime_mp_config.yaml
python3 -m scam.tools.generate_bprime scam/materials/ablative_carbon/calcarb_bprime_ace_config.yaml
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

**Nodelet subgrid** — Each thermal node contains J nodelets (default 4) for decomposition. Nodelet temperatures are interpolated using cumulative *volume* as the coordinate (not distance) — critical for cylindrical geometry. Node density is the volume-weighted average of nodelet densities.

**Node drop direction** — Nodes are always dropped from the BACK face of the ablating layer (deeper into material), not the front. The merging is conservative: volume-weighted average of temperature and density.

**Recession scheme** — Two options (see `config/solver.py::SolverOptions.continuous_remap`). Default (`False`) is the Lagrangian fixed grid: the surface cell shrinks and is dropped/merged one cell at a time — discrete, and it puts a ~±6 K sawtooth on `T_wall` (each drop exposes the next, cooler node). `True` selects a continuous moving-mesh (ALE) scheme (`mesh/receding.py::apply_recession_ale`): the ablating layer's nodes are redistributed between the receding surface and the fixed back face every step with conservative field re-interpolation — node count fixed, no drops, smooth `T_wall`. ALE additionally needs the recession-CFL limit in `time_integration.py` (`dt ≤ 0.1·h/s_dot`) to avoid re-interpolation wobble. See `docs/verification/pato_validation.md`.

**Blowing on m_dot_char** — The blowing factor `1/(1+λ·B_total)` reduces BOTH `q_conv` AND `m_dot_char` (Reynolds analogy: C_H and C_M drop together). `B_total = B'_c + B'_g` is formed on the unblown basis so the algebraic correction stays bounded under heavy injection. See `physics/surface_energy.py::seb_residual` and `docs/verification/pato_validation.md`.

**SEB advective terms** — PATO's `Bprime` BC includes the advective enthalpy of the ablating mass: `qAdvPyro = mDotGw·(h_g − h_w)` and `qAdvChar = mDotCw·(h_c − h_w)` (the char term is the surface oxidation heat release). `seb_residual` adds these (`q_adv`) when the chemistry backend exposes `surface_enthalpies(T_w,p,Z_C_pyro)→(h_g,h_c)` on the same reference as `h_wall`. This is available from the live `BprimeEvaluator` and from Cantera-generated B′ tables carrying optional pretabulated `h_g`/`h_c`; legacy tables omit the term. Closed ~40 K of the ablation2 surface gap. See `docs/verification/pato_validation.md` §8.

**Pyrolysis gas `h_g` enthalpy reference — `h_g_abs_offset`** — YAML material cards store the pyrolysis-gas enthalpy table on the **sensible** reference (`h_g(298 K) ≈ 0`). PATO's gasProperties file and Cantera use the **absolute** (formation-enthalpy) reference (`h_g(298 K) ≈ −7.09 MJ/kg` for TACOT). The optional `MaterialCard.h_g_abs_offset` field (e.g. `−7.093e6` J/kg for TACOT) converts sensible→absolute. `numerics/assembly.py::_build_system` applies the offset to every node's `h_g_cache` when it is not `None`. The surface node gas outflow is explicit (`mg_out[0] = m_dot_g[0]`); combined with SEB `q_adv_pyro = m_dot_g·(h_g_abs(T_w) − h_wall)`, the net surface gas energy ≈ `−m_dot_g·h_wall` (PATO-consistent). Scripts that replace `h_g_table` with already-absolute data (PATO gasProperties or Cantera) **must** also clear `h_g_abs_offset=None` in the `dataclasses.replace()` call to prevent double-application. See `docs/verification/pato_validation.md` §11.

**Cooldown SEB reference consistency** — In PATO 2.x, cooldown maps `rhoUeCH = 0.3e-2` kg/m²/s and `h_r=0`, but `chemistryOn=0` disables the B′ lookup and the Bprime temperature BC ignores `rhoUeCH/h_r`; it uses `hconv*(Tedge-T)` instead. When the B′ lookup is skipped, `h_wall` falls back to `enthalpy_char` (298-ref) instead of the equilibrium wall enthalpy. `q_adv` is therefore gated on `_bprime_ran` (only applied when `h_wall` is on the same reference as `h_g`/`h_c`) — otherwise the `h_g − h_wall` reference mismatch injects a large spurious cooling sink during cooldown. The enthalpy-BC branch uses B′ `h_wall` only when `_bprime_ran`; if a Bprime backend exists but chemistry is off, `seb_residual` falls back to the temperature branch with no advective or mass-removal sink. See `physics/surface_energy.py::seb_residual`.

**Coordinate convention** — `MeshState.y_nodes` is measured from the ORIGINAL front face (surface node at `y_nodes[0] == s_total`; back face fixed). Depth below the current surface is `y_nodes − y_nodes[0]`. Never add `s_total` to `y_nodes`.

**Element transport flux** — `Z_elem` transport (`physics/element_transport.py`) must advect with the pyrolysis gas flux `m_dot_g_nodes` (mass-consistent with its `−dρ/dt` source), NOT the Darcy expansion flux — otherwise each `Z_i` diverges to 1.0. Columns are renormalised to sum to 1 in `indepth_solver.py`.

**Energy storage — `ρ·h(T)` formulation** — The tridiagonal RHS uses sensible enthalpy `h(T) = ∫₀ᵀ cp dT'` rather than `cp·T`. `h_virgin_sensible` / `h_char_sensible` tables are built by `build_sensible_enthalpy_table` in `properties.py` and stored in `MaterialCard`. `mixture_enthalpy_array` evaluates the blended mixture `h(T,ρ)` node-wise. Sensible enthalpy interpolation uses **linear extrapolation** (`_interp_linear_extrap`) beyond the table range — flat `np.interp` extrapolation broke the Picard invariant when temperatures exceeded the table range.

**Picard iteration** — `indepth_solver.py` wraps the tridiagonal solve in a fixed-point loop (`SolverOptions.max_picard=8`, `picard_tol=0.05 K`). Each iteration reassembles with `cp(T^k)` at the current iterate; `h_old` is fixed throughout. `max_picard=1` reproduces old single-pass behaviour. `PRESCRIBED_TEMP` BC bypasses the loop.

**`h_old` at `ρ_old` — exact FVM storage** — `h_old_arr` in `indepth_solver.py` is `mixture_enthalpy_array(state.T, state.rho, ...)` using **pre-decomposition** density `ρ_old`. `_build_system` uses `Dc_thermal = M·T^k + (ρ_old·h_old − ρ_new·h_k)·A·Δ/dt` — the exact FVM form `(ρ_new·h_new − ρ_old·h_old)/dt = fluxes + Q_chemical`. The density-change implicit term `(ρ_old−ρ_new)·h_old/dt` provides `π·h̄_sensible`. Q_vol must therefore carry only `h_bar_chemical = h_bar_absolute − h̄_sensible` (not `h_bar_absolute`) for table materials — the two changes are coupled. `SolverOptions.use_rho_old=True` is the production default. Setting it `False` activates the diagnostic PATO-style storage approximation `ρ·dh/dt` (`Dc_thermal=M·T^n`): it matches PATO mean density near 60 s but diverges during cooldown, so keep it out of production comparisons except in `compare_pato_ablation2_beta.py`.

**Pyrolysis energy (`h_bar` split)** — PATO's `Pyrolysis` energy model has `pyrolysisFlux = −π·(hs + hp)`: `hs` = `h̄_sensible` (carried implicitly by `ρ_old·h_old` LHS), `hp` = chemical part. SCAM: Q_vol carries `h_bar_chemical = h_bar_absolute − h̄_sensible` (via `h_bar_chemical_array`) for table materials; `h_decomp` per component is the pure chemical `hp` for materials without explicit enthalpy tables. Do NOT add `hp` on top of `h_bar_chemical` for table materials — `h_bar_chemical` already captures only the chemical part. TACOT 3.0 uses explicit virgin/char enthalpy tables, so it follows the `h_bar_chemical` path. See `docs/verification/pato_validation.md` §7.

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

`io/case_loader.py` parses YAML input decks into `SimCase`. `io/writers.py` handles CSV and optional HDF5 output (requires `h5py`; install with `pip install -e ".[hdf5]"`).

### Blowing correction

`physics/blowing.py` reduces the effective convective Stanton number when pyrolysis gas and char products are injected: `α_eff = α_conv / (1 + λ·B')`. `λ` defaults to 0.5 (laminar). The individual B' components (`B'_g`, `B'_c`) feed into both the blowing correction and the B' table chemistry lookup.

### Surface chemistry backends (B' table vs live Cantera)

The surface solver needs an object with a `lookup(T_wall, p_e, B'_g, Z_C_pyro=None) -> (B'_c, h_wall)` method; two interchangeable backends implement it and are passed in the `b_prime_tables` dict (keyed by material name):

- **`BPrimeTable`** (defined in `io/material_loader.py`, re-exported by `physics/chemistry.py`) — a `RegularGridInterpolator` over a pre-computed YAML table. TACOT 3.0 uses the 3-D table `scam/materials/ablative_organic/tacot_v3.0_bprime_air.yaml`; element-transport cases use live Cantera with the transported `Z_C_pyro`. Tables are generated by `scam/tools/generate_bprime.py` . Fast; no Cantera at runtime.
- **`BprimeEvaluator`** (`physics/bprime_evaluator.py`) — live Cantera equilibrium, a drop-in for `BPrimeTable` (same `lookup`). Build with `BprimeEvaluator.from_config("scam/materials/ablative_organic/tacot_v3.0_bprime_config.yaml")`. Warm-started (~0.3 ms/call in the current TACOT validation path; about 3–4 s total for ~11.6k lookups). Requires Cantera. **Only** this backend exposes `surface_enthalpies(...)`, which activates the SEB advective terms (see above). `examples/verification/ablation2/compare_pato_ablation2.py` uses it for live chemistry modes; use `chemistry_mode="table"` for the explicit table-only diagnostic. **CRITICAL:** `_TACOT_PYRO_X_NOMINAL` and `tacot_v3.0_bprime_config.yaml` must use `pyro_y: "CH4:0.5551,CO:0.2418,H2O:0.2031"` — this is the ONLY composition that matches PATO's tacot26 elemental fracs (C:0.206, H:0.679, O:0.115). The old preset `CH4:0.4600,...` gave C:0.227/H:0.622/O:0.151 and over-predicted B'_c by 15–140% depending on B'_g.

### Material data

TACOT 3.0 properties for the PATO validation cases are in `scam/materials/ablative_organic/tacot_v3.0.yaml`; `tacot_v3.0_3rxn.yaml` keeps the exact 3-reaction kinetics variant. `ablative_organic/pica.yaml` and `ablative_carbon/fiberform.yaml` are structural templates with representative values only.

Carbon-carbon material for Amar (2006) §8.7: `ablative_carbon/carbon_carbon_amar2006.yaml` (Table 8.12 properties, `calcarb_bprime_air_mp.yaml` — full Cantera CNO C/C2/C3, 6 pressures 0.001–5 atm) and `carbon_carbon_amar2006_ace.yaml` (ACE-style: `calcarb_bprime_ace_air.yaml`, gri30 C-only no C2/C3, same pressures). The two B′ variants exist because Cantera/JANAF predicts carbon sublimation onset at ~3200 K via C2/C3 species, while ACE (used by CMA/SODDIT/Amar) keeps B′_c on the diffusion plateau (~0.175) to ~3600 K. Cantera is thermodynamically more complete; ACE better matches experiments because real sublimation is kinetically suppressed. Use the ACE-style table for code-to-code comparison against CMA/SODDIT.

**B′_c table pressure coverage matters for carbon ablation.** The original `calcarb_bprime_air.yaml` covered only 1 atm; `BPrimeTable` silently clamps above the table edge. Peak edge pressures in ballistic reentry cases reach ~1.6 atm (Amar §8.7), so always use `calcarb_bprime_air_mp.yaml` (covers 5 atm) when the BC pressure can exceed 1 atm.

### Additional physics modules

- `physics/surface_energy.py` — SEB residual and Jacobian for the Newton solve
- `physics/recession.py` — surface recession rate from the surface mass balance (SMB)
- `physics/charring_energy.py` — in-depth chemical energy source `Q_decomp = -(dρ/dt)·h̄(T)`
- `physics/gas_enthalpy.py` — linearly interpolated pyrolysis gas enthalpy `h_g(T)` from material table
- `physics/darcy_flow.py` — gas expansion flux and energy source for porous materials with `gas_porosity > 0`
- `physics/pressure_darcy.py` — pressure-driven Darcy flow: solves 1-D gas pressure field, computes `pressure_driven_mass_flux` and `pressure_darcy_energy_source`
- `numerics/nonlinear.py` — scalar Newton and bisection solvers (used by `surface_solver.py`)
- `geometry/area.py` — area/conductance helpers: `build_area_function`, harmonic conductivity, interface conductance, nodelet cumulative volume (used by `numerics/assembly.py` and `mesh/`)
- `geometry/fvm.py` — FVM face-area averaging used by the assembly to handle non-slab geometries

### Test layout

- `tests/unit/` — single-function correctness checks (geometry, Thomas algorithm, Arrhenius, properties)
- `tests/integration/` — physics-level: erfc analytical solution, energy conservation drift
- `tests/verification/` — analytical verification cases V1–V5: constant/variable conductivity conduction, multilayer, prescribed-T decomposition, surface balance
- `tests/validation/` — full TACOT arcjet run with `scope="module"` fixture (runs once per session); bounds are physics-calibrated, not arbitrary

### PATO validation & docs

PATO comparison scripts live in `examples/verification/ablation1/` and `examples/verification/ablation2/` (helpers in `examples/verification/ablation2/_pato2_common.py`; PATO reference data under the local `pato-3.1` tree). Run with `MPLBACKEND=Agg`; each saves a multi-panel PNG alongside the script. The `ablation2*` cases are the full SEB/recession regime. When changing surface-energy, recession, or pyrolysis physics, regenerate these and check against the references.

Ablation1 variants: `_grading` (approximates depth-varying density via a multilayer stack), `_multiPorousMat` (pyrolyzing TACOT porous layer over inert cork; PATO `porousMat1` uses `PyrolysisType LinearArrhenius`, so do not model it as inert char), `_function` (time-varying BC via Python callables), `_equilibriumElementConservation` (element transport with 4-D B′ table). Ablation2 variants: `_multiMat` (3-layer TACOT + inert sublayers), `_chemistryOff` (no B′ lookup), `_equilibriumElementConservation` (Cantera + element transport).

Conduction verification scripts are in `examples/verification/conduction/` (V1–V3). Run `python3 examples/clean_outputs.py` to remove all generated PNGs and results directories.

Key design docs (read before touching the matched physics):

- `docs/verification/pato_validation.md` — the SCAM↔PATO fixes term by term (blowing, emissivity, h_bar/pyrolysis energy §7, SEB advective terms §8, recession), with the residual-gap analysis. Numbered sections are referenced from the code comments.
- `docs/verification/verification.md` — the V1–V5 verification ladder rationale and per-rung criteria.
- `docs/reference/pato_material_recession_summary.md`, `docs/reference/pato_receding_pyrolyzing_internal_energy_balance.md` — extracted PATO formulations (recession `s_dot=ṁ_char/ρ_s`; the `Pyrolysis` energy model the `h_bar` term must match). SCAM is verified consistent with these.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

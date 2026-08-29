<!-- SPDX-License-Identifier: MIT -->
# PROJECT_CONTEXT.md

Living reference for human collaborators and AI assistants working on SCAM.
Update this file when focus shifts, bugs are resolved, or conventions change.

---

## Project Overview

**SCAM** (Simple Code for Ablative Materials) is a research-grade **1-D charring ablation solver** written in Python. It models the thermal response of ablative heat shields (e.g. carbon-phenolic composites like TACOT) under aerothermal heating — the physics that governs spacecraft re-entry thermal protection systems.

**What it solves:**
- In-depth heat conduction through a multi-layer porous solid (FVM, Backward Euler)
- Pyrolysis decomposition of the virgin material (Arrhenius kinetics, per-nodelet subgrid)
- Surface energy balance (SEB): radiation, convection, char oxidation, pyrolysis gas blowing
- Surface recession via the char surface mass balance (CMA model)
- Optional: element transport (Z_elem PDE), Darcy flow, pressure-driven gas flow

**Reference solver:** PATO (Porous material Analysis Toolbox, OpenFOAM-based). SCAM is validated against PATO's `AblationTestCase_1.x` and `2.x` benchmark suite.

**Stack:** Pure Python 3.10+, NumPy/SciPy for numerics, PyYAML for input, Matplotlib for output. Optional Cantera for live equilibrium chemistry.

**Current state:** Physics-complete and PATO-verified through the full AblationTestCase_2.x benchmark family (surface T within 2 K, recession within 1% of PATO at t=60 s across all variants). The main TACOT 3.0 material (`scam/materials/ablative_organic/tacot_v3.0.yaml`) is the canonical TACOT validation card; `tacot_v3.0_3rxn.yaml` keeps the exact 3-reaction kinetics variant. `ablative_organic/pica.yaml` and `ablative_carbon/fiberform.yaml` are structural templates with placeholder values only. Amar (2006) §8.7 carbon-carbon verification is underway: material cards `ablative_carbon/carbon_carbon_amar2006.yaml` and `carbon_carbon_amar2006_ace.yaml` are in place with dual B′ tables (full Cantera CNO and ACE-style gri30); §8.8 carbon-phenolic script skeleton exists, but reference figures are not yet digitized.

---

## Architecture Notes

### Solver sequence (one timestep)
Defined in `solvers/indepth_solver.py::step()`:
1. Decomposition — Arrhenius ODE per nodelet (analytic integration)
2. Pyrolysis gas flux — from `−dρ/dt` integrated over nodes
3. FVM tridiagonal assembly — A/B/C/D coefficients with harmonic conductance
4. F_cond reduction — backward elimination → `q_cond = α_F·T_w + β_F`
5. SEB Newton solve — scalar iterate on `T_wall`; uses `α_F`/`β_F` as the in-depth coupling
6. Tridiagonal solve — Thomas algorithm with `q_cond` inserted at node 0

The outer time loop is `solvers/material_response.py::run()`, which wraps the above with recession, node management, and adaptive timestep.

### Module dependency order (enforced, no upward imports)
```
core/constants, core/errors
  → config/*
    → core/state, core/results
      → geometry/*, mesh/*, io/
        → physics/*
          → numerics/*
            → solvers/*
              → cli/
```

### Key design decisions

**Two recession schemes** (`SolverOptions.continuous_remap`):
- `False` (default): Lagrangian — surface cell shrinks until dropped/merged. Simple but produces ~±6 K sawtooth on `T_wall`.
- `True`: ALE continuous remap — nodes redistributed every step between receding surface and fixed back face. Smooth `T_wall`, requires recession-CFL limit (`dt ≤ 0.1·h/s_dot`).

**Two surface chemistry backends** (both implement `.lookup(T_wall, p_e, B'_g, Z_C_pyro=None) → (B'_c, h_wall)`):
- `BPrimeTable` — pre-computed `RegularGridInterpolator` over a YAML table (3-D or 4-D with element transport). Fast, no Cantera needed.
- `BprimeEvaluator` (`physics/bprime_evaluator.py`) — live Cantera equilibrium. Slower, but the **only** backend that exposes `surface_enthalpies()`, which activates the SEB advective terms `q_adv`. Required to close the ~40 K surface gap vs PATO ablation2.

**Nodelet subgrid:** Each thermal node contains J nodelets (default 4) for decomposition. Temperature is interpolated using cumulative *volume* as coordinate (not distance) — important for non-slab geometries.

**Energy storage `ρ·h(T)` and Picard iteration (exact FVM):** The tridiagonal RHS stores sensible enthalpy `h(T)=∫cp dT` (not `cp·T`). `h_old = mixture_enthalpy_array(T^n, ρ_old)` is computed at **pre-decomposition** density and fixed for the Picard loop. `_build_system` computes `Dc_thermal = M·T^k + (ρ_old·h_old − ρ_new·h_k)·A·Δ/dt` — this is the exact FVM form `(ρ_new·h_new − ρ_old·h_old)/dt = fluxes + Q_chemical`. The density-change term `(ρ_old−ρ_new)·h_old/dt` provides implicit `π·h̄_sensible`; Q_vol carries only `h_bar_chemical = h_bar_absolute − h̄_sensible` (for table materials) or `h_decomp` (pure chemical `hp`). The Picard loop re-evaluates `cp(T^k)` each iteration until `max|ΔT| < picard_tol` (default 8 iterations, 0.05 K). Sensible enthalpy tables extrapolate linearly beyond their range via `_interp_linear_extrap` — flat extrapolation breaks Picard convergence when T exceeds the table bound.

**h_bar_chemical (pyrolysis energy):** Q_vol carries `h_bar_chemical = h_bar_absolute − h̄_sensible` (computed by `h_bar_chemical_array` in `properties.py`) for materials with explicit enthalpy tables. The sensible piece `h̄_sensible` is captured implicitly by the `ρ_old·h_old` storage term. For tacot-style materials (no explicit tables), `h_decomp` represents the pure chemical `hp`; `h̄_sensible` is now also implicitly present via the exact LHS (previously missing). Do NOT add `hp` on top of `h_bar_chemical` for table materials — `h_bar_chemical` already strips the sensible, so adding `hp` again would double-count the chemical part.

### Folder structure rationale

| Folder | Role |
|---|---|
| `scam/core/` | Primitives: constants, errors, `SimState`, `MeshState`, `Results` |
| `scam/config/` | Dataclasses for all user-facing inputs (parsed from YAML) |
| `scam/geometry/` | Area/conductance helpers; FVM face-area averaging for non-slab geometries |
| `scam/io/` | YAML case loader, material loader, CSV/HDF5 writers |
| `scam/mesh/` | Grid construction, ALE recession, node remap/drop |
| `scam/physics/` | Individual physics sub-models (each file = one concept) |
| `scam/numerics/` | FVM assembly, Thomas solver, nonlinear solvers, timestepping |
| `scam/solvers/` | Orchestration: in-depth step, surface solve, outer time loop |
| `scam/materials/` | YAML material cards + B′ tables |
| `scam/tools/` | Offline utilities (B′ table generation) |
| `docs/` | Design rationale, PATO validation record, verification criteria |
| `examples/` | Runnable scripts; standalone demos are in `examples/other_examples/`; subfolders below |
| `examples/verification/conduction/` | V1–V3 conduction verification scripts and output PNGs |
| `examples/verification/ablation1/` | `compare_pato_ablation1*.py` scripts and output PNGs |
| `examples/verification/ablation2/` | `compare_pato_ablation2*.py` scripts, `_pato2_common.py`, output PNGs |
| `tests/` | unit / integration / verification / validation |

---

## Coding Conventions

**Python style:**
- Dataclasses for all config and state objects (`@dataclass`, no mutable defaults)
- Type annotations throughout; `from __future__ import annotations` in all modules
- NumPy arrays are 1-D unless geometry requires otherwise; shapes are documented in docstrings
- Physical quantities use SI units throughout; variable names include units in comments where non-obvious

**Naming patterns:**
- Physics quantities follow the CMA/PATO convention: `m_dot_g`, `B_prime_c`, `q_cond`, `rho_e_u_e`, `s_dot`
- Node arrays are length `n_nodes`; nodelet arrays are `(n_nodes, n_nodelets)`
- Layer-indexed arrays use `layer_id` (int array, same length as `y_nodes`)
- Config classes: `*Config` suffix; result containers: `*Results`
- Backend duck-typing: surface chemistry objects just need `.lookup(...)` — no ABC enforced

**Boundary condition parameters** accept either `float` or `Callable[[float], float]`; always resolve via `eval_bc(param, t)` from `config/boundary.py`.

**No silent fallbacks in physics.** If a material property is missing, raise `MaterialError` rather than substituting a default. The surface chemistry backends are passed explicitly through the call stack — never looked up globally.

**Tests:**
- Unit tests are self-contained and fast (no file I/O)
- Verification tests (`tests/verification/`) compare against analytical solutions; tolerances are in `conftest.py` with documented physical rationale
- The validation fixture (`tests/validation/`) runs the full TACOT arcjet once per session (`scope="module"`)
- Always run `MPLBACKEND=Agg` when examples invoke `plt.show()`

---

## Current Focus

Core physics and PATO validation are complete.

Recently completed:

- **TC2.2 / h_g reference fix** — YAML material cards store sensible `h_g`; PATO/Cantera use absolute. Added `MaterialCard.h_g_abs_offset` field and apply it in `assembly.py::_build_system` to convert sensible→absolute before Q_adv. Surface node gas outflow is now explicit (`mg_out[0] = m_dot_g[0]`); old SEB cooldown compensation block removed. Scripts replacing `h_g_table` with absolute data must clear `h_g_abs_offset=None`. TC2.2 (TACOT v2.2, Mutation++ B′, enthalpy BC): SCAM 1568.1 K vs PATO 1568.4 K (Δ = −0.3 K) at t = 60 s. Ablation2 base: +0.8 K. See `docs/verification/pato_validation.md` §11.
- **Exact FVM energy formulation** — `ρ·h(T)` storage with `h_old` at `ρ_old`; Picard iteration; linear enthalpy extrapolation. Production ablation2 comparisons match PATO to ~+0.8 K at 60 s.
- **Performance pass** — all inner loops vectorised; 501-node live-Cantera run ~11 s (element transport ~75 s).
- **Examples reorganised** — `examples/verification/` subtree; `clean_outputs.py`.

Likely next areas:

- Finish Amar §8.7 debug (close remaining +7% recession overshoot)
- Digitize Amar §8.8 reference figures and run carbon-phenolic comparison
- Additional materials (PICA, FiberForm — currently placeholder properties)

---

## Known Issues / Things to Avoid

**`y_nodes` coordinate:** measured from the ORIGINAL front face; `y_nodes[0] == s_total` as the surface recedes. Depth below current surface is `y_nodes − y_nodes[0]`. Never add `s_total` to `y_nodes` — that double-counts recession.

**F_cond sign:** The backward elimination sweep in `assembly.py::compute_F_cond` must use `-=`, not `+=`. The factor `C[n-1]/B[n]` is negative (C is negative), so `+=` silently goes the wrong way and causes Newton non-convergence. This was a critical historical bug.

**Cantera pyrolysis composition:** `tacot_v3.0_bprime_config.yaml` and `_TACOT_PYRO_X_NOMINAL` must use `CH4:0.5551, CO:0.2418, H2O:0.2031`. The old `CH4:0.4600,...` preset gave wrong elemental fractions (C/H/O off by 10–20%) and over-predicted `B'_c` by 15–140%.

**Cooldown enthalpy references:** The SEB advective terms (`h_g − h_wall`, `h_c − h_wall`) must be gated on `_bprime_ran`. In PATO 2.x cooldown, `rhoUeCH` drops to `0.3e-2` kg/m²/s and `h_r=0`, but `chemistryOn=0` skips B′ and the Bprime temperature BC ignores `rhoUeCH/h_r`; SCAM must therefore set the chemistry driver `rho_e_u_e` to zero and fall back to `hconv*(Tedge-T)`. When B′ is skipped while a Bprime backend exists, do not use material `h_gas(T_w)`, equilibrium `h_wall`, advective terms, or a mass-removal sink.

**Ablation1 multiPorousMat:** PATO `AblationTestCase_1.0_multiPorousMat` has a pyrolyzing TACOT `porousMat1` over inert cork. `porousMat1Properties` uses `PyrolysisType LinearArrhenius`; treating it as inert char makes the 5 mm probe run 70-90 K too hot after the initial transient. The SCAM comparison enables the reduced TACOT two-resin decomposition model, with the largest remaining 5 mm mismatch concentrated near the early heat-front arrival.

**Element transport flux:** `Z_elem` advection must use `m_dot_g_nodes` (pyrolysis flux, mass-consistent with `−dρ/dt`), NOT the Darcy expansion flux. Using the wrong flux causes each element fraction to drift to 1.0.

**Node drop direction:** Nodes are always merged from the BACK face of the ablating layer (deeper side), not the front surface. Merging is volume-weighted for both T and ρ.

**`pyropy/` is a separate project** (pyrolysis kinetics parameter optimization) developed by Francisco Torres Herrador, available at [github.com/Fratorhe/pyropy](https://github.com/Fratorhe/pyropy). It lives at a different path and is not part of the SCAM package — do not import or add it as a dependency.

**`h_old` uses `ρ_old` (exact FVM):** `h_old_arr = mixture_enthalpy_array(state.T, state.rho, ...)` in `indepth_solver.py`. The implicit `π·h̄_sensible` from the LHS density-change term is intentional — Q_vol carries only `h_bar_chemical` (not `h_bar_absolute`) to balance it. If you revert Q_vol to `h_bar_absolute` while keeping `ρ_old` in h_old, you double-count the sensible piece. If you revert h_old to `ρ_new` while keeping `h_bar_chemical` in Q_vol, you get a zero sensible source (missing `h̄_sensible`). The two changes are coupled; always change both together.

**Picard enthalpy extrapolation:** `_interp_linear_extrap` in `properties.py` must be used (not bare `np.interp`) for `mixture_enthalpy_array`. Flat extrapolation at the table boundary breaks the Picard invariant for high-flux cases where T exceeds the cp table range (caused 49% error in v1b transient before fix).

**`pica.yaml` / `fiberform.yaml`** have placeholder material properties only. Do not use them for quantitative validation.

**B′_c table pressure clamping:** `BPrimeTable` silently clamps to the table edge when `p_e` exceeds the table maximum. The original `calcarb_bprime_air.yaml` covers only 1 atm; peak edge pressure in Amar §8.7 reaches 1.624 atm. Always use `calcarb_bprime_air_mp.yaml` (covers 5 atm) for carbon ablation cases where BC pressure can exceed 1 atm.

**Cantera vs ACE B′_c for carbon sublimation:** Cantera with full CNO species (C, C2, C3) predicts sublimation onset at ~3200 K, giving B′_c well above the diffusion-controlled plateau at peak heating. ACE (used by CMA, SODDIT, Amar) lacks C2/C3 and keeps B′_c near 0.175 to ~3600 K. For code-to-code comparison against CMA/SODDIT, use the ACE-style table (`calcarb_bprime_ace_air.yaml`, gri30 mechanism). For thermodynamically rigorous predictions, use the full CNO table (`calcarb_bprime_air_mp.yaml`). The ACE behavior is physically justifiable because real graphite sublimation is kinetically suppressed (evaporation coefficient 0.1–0.3).

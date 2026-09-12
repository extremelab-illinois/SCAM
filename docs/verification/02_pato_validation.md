<!-- SPDX-License-Identifier: MIT -->
# SCAM ↔ PATO Verification: Fixes & Findings

This document records the physics and numerics fixes made while matching SCAM to
the PATO `AblationTestCase_2.x_equilibriumElementConservation` benchmark (5 cm
TACOT slab, enthalpy-based convective BC `rhoUeCH = 0.3`, `h_r = 1.5e6`,
heating 0.1–60 s, cooling 60.1–120 s). In the PATO boundary table, cooldown
sets `rhoUeCH = 0.3e-2` kg/m²/s and `h_r=0`, but `chemistryOn=0` disables B′
chemistry and the Bprime temperature BC ignores the `rhoUeCH/h_r` columns in
that branch. The driver is
[`examples/verification/ablation2/compare_pato_ablation2_equilibriumElementConservation.py`](../../examples/verification/ablation2/compare_pato_ablation2_equilibriumElementConservation.py).

Each section gives the **symptom**, the **root cause**, the **fix** (with the
files/functions touched), and the **result**.

For SCAM this means `rhoUeCH` and the B′ chemistry driver `rho_e_u_e` are not
identical during cooldown: retain the mapped `rhoUeCH` value for diagnostics,
but set `rho_e_u_e=0` so the B′ lookup and Cantera advective terms are skipped.
When the B′ lookup is skipped while a Bprime backend exists,
[`physics/surface_energy.py::seb_residual`](../../scam/physics/surface_energy.py)
falls back to the temperature convection branch (`hconv*(Tedge-T)`) with no
advective or mass-removal sink.

PATO reference at t = 60 s: surface `T_wall = 1568.5 K`, recession ≈ 11.8 mm
(inferred from the probe-ablation times, since the reference has no mass file).

---

## 1. Blowing correction on the char mass flux (Reynolds analogy)

**Symptom.** SCAM over-predicted recession (14.3 mm vs PATO 11.8 mm) and ran
~48 K too cold at the surface. Reducing char ablation moved *all* metrics —
surface T, recession, and the whole in-depth profile — toward PATO at once,
identifying char ablation rate as the single controlling lever.

**Root cause.** In [`physics/surface_energy.py::seb_residual`](../../scam/physics/surface_energy.py),
the blowing correction `1/(1+λ·B)` was applied **only** to the convective heat
flux (`rhoUeCH_eff`), while the char mass flux used the *unblown* transfer
coefficient `m_dot_char = B'_c · ρ_e u_e C_M`. By the Reynolds analogy, blowing
reduces the heat **and** mass transfer coefficients by the same factor, so this
was inconsistent. (The B′ table itself is correct: its oxidation plateau
`B'_c ≈ 0.175` is the textbook diffusion-limited carbon-in-air value
`Y_O·M_C/M_O = 0.233·12/16`.)

**Fix.** Apply the same `blow_factor` to `m_dot_char`:
`m_dot_char = m_dot_char_unblown · blow_factor`. The blowing parameter
`B_total = B'_c + B'_g` is formed on the **unblown** basis so the algebraic
correction stays bounded under heavy injection (a self-consistent loop that
rescales `B_g` by the blown coefficient diverges when `m_dot_pyro/denom > 1/λ`,
e.g. the low-`C_M = 0.01` verification cases).

**Result.** Surface-T gap 48 K → 31 K; recession 14.3 → 12.85 mm. Tests that
rebuild the SEB ([`tests/verification/test_v5_surface_balance.py`](../../tests/verification/test_v5_surface_balance.py))
were updated to apply `blow_factor` to `m_dot_char` as well.

---

## 2. Depth coordinate convention (`y_nodes`)

**Symptom.** The comparison plot showed a ~900 K in-depth error at t = 30 s,
12 mm (SCAM ≈ 1500 K vs PATO ≈ 600 K), suggesting the transient was completely
wrong.

**Root cause — diagnostic only, not the solver.** `MeshState.y_nodes` is the
absolute coordinate measured from the **original front face**: the back face is
fixed and the surface node sits at `y_nodes[0] == s_total` as the front recedes.
But `T_at_original_depths` and the profile plot computed `y_orig = y_nodes +
s_total`, **double-counting** the recession and shifting SCAM's whole profile
inward by one full recession depth (~6.4 mm at t = 30 s). The physics was fine.

**Fix.** Use `y_nodes` directly as depth-from-original-surface; a probe at depth
`d` is ablated once `d < y_nodes[0] (== s_total)`. Touched
[`examples/verification/ablation2/_pato2_common.py::T_at_original_depths`](../../examples/verification/ablation2/_pato2_common.py)
and the profile/Z_C panels of the comparison script.

**Result.** Max in-depth error 946 K → ~200 K. Recession-aligned, the profiles
agree within ~20–50 K (e.g. t = 30 s / 12 mm: SCAM 628 K vs PATO 610 K).

**Rule of thumb.** `y_nodes` is from the original front face. Depth below the
*current* surface is `y_nodes − y_nodes[0]`. **Never add `s_total` again.**

---

## 3. Element-transport flux consistency (Z_C → 1.0 divergence)

**Symptom.** With element transport enabled, the gas-phase element fractions
`Z_elem` (C/H/O/N) diverged — every `Z_i` clipped to 1.0 (column sum → 3).

**Root cause.** In [`solvers/indepth_solver.py::step`](../../scam/solvers/indepth_solver.py)
the transport PDE was advected with the Darcy **thermal-expansion** flux while
its source was the **pyrolysis production** rate `(−dρ/dt)`. These are
mass-inconsistent, so the carrier-gas continuity
`f_out·A − f_in·A = (−dρ/dt)·V` did not hold, and the convex-combination
property that keeps each `Z_i` a bounded weighted average of the pyrolysis
fractions was lost.

**Fix.** Advect with the **pyrolysis gas mass flux** `m_dot_g_nodes` (the
reverse-cumulative sum of `(−dρ/dt)·A·δ`, hence exactly mass-consistent with the
source); the face-flux density is `m_dot_g_nodes[j] / area_nodes[j]`. Then
renormalise the columns of `Z_elem` to sum to 1 **at the integration point**
(not inside `element_transport.solve_element_transport`, whose operator stays
pure so the isolation unit tests keep their inputs).

**Result.** `Z_C` at the surface ≈ 0.49, matching the pyrolysis carbon fraction
(0.495); columns sum to 1. Regression guard:
`tests/unit/test_element_transport.py::test_mass_consistent_flux_bounds_Z_to_pyro_fraction`.

---

## 4. T_wall sawtooth → continuous moving-mesh (ALE) recession

**Symptom.** The solved surface temperature showed a ~±6 K sawtooth in time.

**Root cause (two mechanisms).**

1. *Dominant, periodic.* The Lagrangian fixed grid drops/merges the surface cell
   one cell at a time ([`mesh/remap.py`](../../scam/mesh/remap.py)); each drop
   discretely exposes the next, cooler sub-surface node, so `T_wall` steps down
   ~11 K then re-heats. Period = one cell of recession.
2. *Residual.* The adaptive-timestep controller was bang-bang (halve / ×1.2),
   making `dt` oscillate → backward-Euler truncation wobble.

**Fix (all opt-in, default behaviour unchanged).**

- `SolverOptions.continuous_remap` (default `False`). When `True`, recession
  uses a continuous moving-mesh (ALE) scheme,
  [`mesh/receding.py::apply_recession_ale`](../../scam/mesh/receding.py): the
  ablating layer's nodes are redistributed between the receding surface and the
  fixed back face every step (node count fixed, **no discrete drops**), with
  conservative re-interpolation of `T`, `Z_elem`, and the per-nodelet
  `rho_components`.
- A **smooth proportional** dt controller in
  [`numerics/time_integration.py`](../../scam/numerics/time_integration.py): scale
  `dt` to keep the worst normalised change ≈ 0.9 of its limit, with the per-step
  change capped to ×0.5 … ×1.2.
- A **recession-CFL** limit (only under `continuous_remap`):
  `dt ≤ 0.1·h_surface / s_dot`, so the surface advances < 0.1 cell per step —
  otherwise linear re-interpolation of the steep near-surface front injects a
  few-K wobble.

**Result.** Sawtooth max step 11.7 K → 0.08 K (std 5.8 → 0.013 K); `T_wall` and
recession unchanged. The Lagrangian scheme remains the default and is still
covered by the test suite.

---

## 5. CMA vs EquilibriumElement modes

`run_scam(mode=...)` in the comparison script runs two surface-chemistry
models:

- **`"equilibrium"`** (EquilibriumElement, PATO-equivalent): the in-depth
  element-conservation PDE tracks `Z_C(y,t)` and the surface carbon fraction
  feeds the live Cantera backend (`element_transport=True`).
- **`"cma"`** (classic CMA): no element conservation; the injected pyrolysis gas
  composition is fixed at the material/Cantera nominal TACOT pyrolysis-gas carbon
  fraction (`Z_C_pyro≈0.494984`, `element_transport=False`).

Both use the continuous ALE recession so `T_wall` is sawtooth-free.

| Model | `T_wall(60 s)` | Δ vs PATO | Recession(60 s) | Heating max\|ΔT\| | Cooling max\|ΔT\| |
| --- | --- | --- | --- | --- | --- |
| PATO | 1568.5 K | — | ≈ 11.8 mm | — | — |
| SCAM EquilibriumElement | 1563.1 K | −5.3 K | 12.11 mm | 30.7 K | 37.4 K |
| SCAM CMA | 1562.8 K | −5.6 K | 12.08 mm | 32.7 K | 37.3 K |

The two SCAM curves nearly overlay because the fixed CMA pyrolysis-gas carbon
fraction is essentially the same value the EquilibriumElement transport reaches
at the surface. That is the correct CMA comparison: fixed material gas
composition, no transported element field. The older raw-Cantera diagnostic
(`Z_C_pyro=None`) was useful for debugging a composition mismatch, but it is not
the PATO-equivalent CMA physics.

---

## 6. Comparison metric (flux-cutoff / startup sampling artifact)

**Symptom.** A spurious ~315 K "surface error" in the reported max\|ΔT\|.

**Root cause — metric, not physics.** The flux is cut off (near-instantaneously)
at t = 60 s, and the surface shoots 300 → ~1270 K within the first second at
startup. The PATO reference is only **1 s-resolved**, so it cannot represent
those sub-second transients; comparing SCAM's sub-step-resolved curve against a
linear interpolation of PATO's coarse samples across those cliffs inflates the
error.

**Fix.** Report the **heating** and **cooling** phases separately, exclude the
startup ramp (t < 2 s) and the flux-cutoff transition window
([59.9, 61.1] s) from the surface metric, and mask that window in the
difference panel (`_phase_max_dT` and `_masked_diff` in the comparison script).

**Result.** Surface differences at aligned times: heating ≤ 39 K, cooling ≤ 49 K
(EquilibriumElement). The cooling difference is the same modest model gap that
persists during heating, not a transient artifact.

---

## 7. In-depth pyrolysis energy: the `h_bar` (`hs`) term is correct

> **Correction.** An earlier revision of this note claimed PATO has *no* solid
> pyrolysis-energy term and that SCAM's `h_bar` should be removed. That was based
> on reading the wrong energy model (`ForchheimerPyrolysisEnergyModel.C`). The
> `AblationTestCase_2.x*` cases actually select `EnergyType Pyrolysis`
> (`porousMatProperties`), i.e. `PyrolysisEnergyModel.C`, which **does** carry a
> solid pyrolysis term. The conclusion below supersedes it.

**PATO's term.** The `Pyrolysis` energy model assembles
`rho_s*cp*ddt(T) + pyrolysisFlux_ + (gas storage/transport) − laplacian(k,T)`,
with (`PorousMaterialPropertiesModel.C`)

```text
pyrolysisFlux_ = − Σ_i piPyroReac_i · (hp_i + hs_)
hs_ = (ρ_v·[h_v(T)−h_v(298)] − ρ_c·[h_c(T)−h_c(298)])/(ρ_v−ρ_c)      ← == SCAM's h_bar
hp_i = per-reaction heat of pyrolysis (TACOT constantProperties: h[2][i] = −4e6 J/kg)
```

So the solid pyrolysis source has two parts: `hs` (the virgin/char enthalpy
difference — identical to SCAM's `h_bar`) and `hp` (the reaction enthalpy of the
produced gas).

**How SCAM maps to it.** SCAM splits the same energy differently: the `hs` part
is the `h_bar` term in [`numerics/assembly.py`](../../scam/numerics/assembly.py),
and the `hp`/gas part is carried by SCAM's pore-gas energy terms (the
gas-storage `cp`-correction + the `h_g` advection). This was confirmed by direct
test on `AblationTestCase_2.x` (recession-aligned in-depth at t = 60 s, PATO
probe at ~4 mm below the surface = 842 K):

| SCAM solid pyrolysis term | surface | 4 mm-zone |
| --- | --- | --- |
| none | 1523 K | 962 K (**+120**, far too hot) |
| `hs` only (`h_bar`) | 1497 K | **827 K (−15, matches PATO)** |
| `hs + hp` (literal PATO) | 1457 K | 707 K (−135, double-counted) |

`hs` alone reproduces PATO's in-depth pyrolysis zone; adding `hp` on top
double-counts (because SCAM's gas terms already carry it). **So `h_bar` is the
correct and consistent term and is kept** (always applied when virgin/char
enthalpy tables are present). Materials using only the component `h_decomp`
mechanism, with no virgin/char enthalpy tables, keep `h_bar = 0` and are
unaffected by the enthalpy-table path.

**Emissivity.** Independently, PATO's TACOT_v3 char file specifies ε = 0.9
(virgin 0.8); the comparison examples now use the material's blended ε (via
`emissivity = −1` in the BC) instead of a hardcoded 0.8 override — see §6 of
`docs/verification/verification.md`-adjacent material notes and `properties.py::surface_emissivity`.

## 8. Surface energy balance: advective enthalpy terms (PATO `qAdvPyro`/`qAdvChar`)

For a standalone implementation reference for SCAM's SEB signs, branches,
enthalpy references, table-vs-Cantera behavior, and cooldown gates, see
[`docs/theory/02_surface_energy_balance.md`](../theory/02_surface_energy_balance.md).

A term-by-term comparison against PATO's `Bprime` boundary condition
(`BprimeBoundaryConditions.C`) revealed SCAM's SEB was **missing the advective
enthalpy of the ablating mass**. PATO's surface flux balance is

```text
qConv      = ρeUeCH·F_b·(h_r − h_w)
qAdvPyro   = mDotGw·(h_g − h_w)              ← pyrolysis gas leaving at h_w
qAdvChar   = mDotCw·(h_c − h_w)              ← char leaving at h_w (oxidation release)
qRadEm     = −εσ·(T_w⁴ − T_bg⁴)
q_cond     = qConv + qAdvPyro + qAdvChar + qRadEm
```

SCAM kept only `q_conv = ρUeCH·(h_r − h_wall)` and folded mass removal into
`h_wall` (`q_mass_removal = 0`) — i.e. it dropped `qAdvPyro` and `qAdvChar`. With
all enthalpies on one Cantera reference (`h_w = +42`, `h_g = −3576`, `h_c = +2069`
kJ/kg at 1568 K), the missing terms net to **≈ +49 kW/m²** — a surface energy
source dominated by the char-oxidation release `qAdvChar ≈ +96 kW/m²` (partly
offset by the pyrolysis-gas sink `qAdvPyro ≈ −47 kW/m²`). Omitting it left the
surface tens of K too cold.

**Fix.** `seb_residual` adds `q_adv = mDotGw·(h_g − h_wall) +
mDotCw·(h_c − h_wall)` when the chemistry backend can supply `h_g`/`h_c` on the
same reference as `h_wall`.  The live Cantera backend provides these through
`BprimeEvaluator.surface_enthalpies`.  Cantera-generated B′ tables can now store
the same enthalpies: fixed-composition tables use `h_g(T,p)` / `h_c(T,p)`, while
element-transport tables use `h_g(T,p,Z_C_pyro)` / `h_c(T,p)`.  Legacy tables
without these optional arrays retain `q_adv = 0`.

**Result** on `AblationTestCase_2.x` (Cantera backend): surface `T_wall(60 s)` rose
from ~1507 K to **1547 K** (gap −61 → **−22 K**), with the in-depth match
preserved (4 mm: 827 vs 842 K; 12 mm: 407 vs 417 K). The test suite protects
the updated balance.

### Full PATO-equivalent stack (equilibrium element conservation)

`compare_pato_ablation2_equilibriumElementConservation.py` combines all the
matched physics for the case PATO runs with `GasPropertiesType Equilibrium` +
`MaterialChemistryType EquilibriumElement`: (1) the in-depth **element-transport**
PDE (`Z_C_pyro`), (2) the in-depth **Cantera equilibrium gas** `h_g(T)` (live
`equilibrate('TP')` of the TACOT pyrolysis-gas elements; `_pato2_common.equilibrium_hg_table`),
and (3) the **live Cantera surface backend** with the advective SEB terms. After
fixing the live Cantera `Z_C_pyro -> pyro_x` reconstruction to preserve PATO
TACOT's pyrolysis zeta H/O ratio, the EquilibriumElement mode matches PATO's
surface to **−5.3 K** at 60 s (`T_wall(60 s)=1563.1 K` vs 1568.5; heating
max\|ΔT\| 30.7 K). The script's CMA mode disables element transport but uses the
same fixed TACOT pyrolysis-gas carbon fraction as the base CMA model, so it is
only **−5.6 K** at 60 s and visually overlays the EquilibriumElement curve.
The remaining spread is small because this case's transported surface `Z_C`
stays close to the nominal TACOT pyrolysis value.

The base `compare_pato_ablation2.py` intentionally remains the PATO base 2.x
model: `GasPropertiesType Tabulated` and no in-depth element-transport PDE. The
PATO base material still defines a fixed pyrolysis elemental composition
(`zeta[C/H/O] = 0.494996/0.136912/0.368092`), so the SCAM base comparison now
passes that fixed carbon fraction to the live Cantera surface backend when element
transport is off. This imports the useful EquilibriumElement surface-composition
lesson without switching the base case's in-depth gas model.

### Surface chemistry backend (Cantera)

`compare_pato_ablation2.py` now drives surface chemistry through the **live
Cantera equilibrium backend** (`physics/bprime_evaluator.py::BprimeEvaluator`,
a drop-in for `BPrimeTable`) instead of the precomputed table. It is warm-started
(~ms per call) and falls back to the YAML table if Cantera is unavailable. The
live backend gives a more-negative `h_wall` than the 3-D table (≈ −15 vs +60
kJ/kg at 1524 K) → higher `q_conv`, and uniquely provides the consistent
`h_g`/`h_c` needed for the advective terms above.

## 9. Cantera pyrolysis gas elemental composition (pyro_y)

**Symptom.** Live Cantera `B'_c` was 15–140% higher than PATO's Mutation++
tacot26 table depending on the pyrolysis blowing ratio `B'_g` (15% at `B'_g=0.1`,
2.4× at `B'_g=0.45`).

**Root cause.** The Cantera preset `tacot` used `pyro_y: "CH4:0.4600,CO:0.3506,H2O:0.1894"`.
Converting to elemental mole fractions gives C:0.227, H:0.622, O:0.151 — but
PATO's `tacot26.xml` specifies C:0.206, H:0.679, O:0.115. The mismatch injects
a more carbon-rich, oxygen-rich pyrolysis gas than PATO does, shifting the
equilibrium surface composition toward higher char oxidation and inflating `B'_c`.

**Fix.** Compute the CH4/CO/H2O mole fractions that exactly reproduce the PATO
elemental fractions:

```text
C: a + b = 0.206 Z         (Z = total atoms/molecule)
H: 4a + 2c = 0.679 Z
O: b + c = 0.115 Z
a + b + c = 1              → Z = (6 - 4·d) / 1.551 with d=0 for pure CH4/CO/H2O
```

Solving: **CH4:0.5551, CO:0.2418, H2O:0.2031**. Updated in:

- `scam/materials/ablative_organic/tacot_v3.0_bprime_config.yaml` — uses explicit
  `edge_x`/`pyro_y`
- `scam/physics/bprime_evaluator.py::_TACOT_PYRO_X_NOMINAL` — updated fallback

**Result.** Cantera `B'_c` now matches PATO within **0.03–0.7%** across
T = 1500–2100 K and B'g = 0.1–0.45. Recession improved from ~12.8 mm to
**12.08 mm** vs PATO 11.87 mm. Surface `T_wall(60 s)` gap is **−4.5 K**. Note:
the pre-computed 3-D TACOT 3.0 YAML table (`tacot_v3.0_bprime_air.yaml`) carries the
same chemistry data and is the table backend used by the ablation2 scripts.

---

## Residual gap

With all physics corrections (§1–11: blowing, h_bar, ALE recession, SEB advective
terms, cooldown reference gates, Cantera pyro_y fix, T_rad_in=300 K, surface
pyrolysis-gas advection), the base `AblationTestCase_2.x` Cantera comparison now
matches:

- Surface `T_wall(60 s)` = **1563.9 K** vs PATO 1568.5 K (gap -4.6 K)
- Recession = **12.08 mm** vs PATO 11.87 mm at 60 s (gap +0.21 mm)
- Heating max|dT| = **34.2 K** (mostly startup/ramp)
- Cooldown max|dT| = **5.6 K** (was 55.9 K before §11)

For `AblationTestCase_2.x_equilibriumElementConservation`, SCAM's equilibrium
and CMA variants cool almost identically.

### History: the cooldown gap (now closed by §11)

Before the §11 fix, SCAM cooled ~56 K faster than PATO after flux-off.
Investigations at the time incorrectly attributed this to in-depth
storage/redistribution differences or PATO reference-model spread, and proposed
candidates including emissivity tuning and recession rescaling. None of those
was the root cause. The actual driver was the uncompensated pyrolysis-gas outflow
at the surface node — identified and quantified by fresh energy-balance analysis
in June 2026 (see §11 and `docs/physics/ablation2_cooldown_gas_advection.md`). The fix
brings the cooldown max|ΔT| from 55.9 K to **5.6 K** with no changes to
emissivity, recession, or in-depth terms.

### Base 2.x comparison-script hygiene

`compare_pato_ablation2.py` reports surface-temperature error with the same
phase-aware metric as the EquilibriumElement comparison: the startup ramp
(`t < 2 s`) and the flux-cutoff sampling window (`59.9–61.1 s`) are excluded
from the max-error statistic because the PATO reference is only 1 s-resolved.
The temperature-profile panel also plots `mesh.y_nodes` directly; the coordinate
is already measured from the original front face (`y_nodes[0] == s_total`), so
adding `s_total` again would double-count recession.

The base comparison has four useful surface-chemistry diagnostics:

| Backend | `T_wall(60 s)` gap | recession(60 s) gap | What it shows |
| --- | ---: | ---: | --- |
| `live_cantera` raw nominal pyro gas | −22.7 K | +1.02 mm | diagnostic mode; nominal Cantera pyro gas leaves the surface too cold |
| `live_cantera_zc_default` Cantera-derived Z_C default | −4.1 K | +0.21 mm | current default; T_rad_in=300 K matches PATO Tbackground |
| legacy table only (without `h_g`/`h_c`) | −72.2 K | +0.20 mm | table `B'_c` matches PATO recession, but lacks `qAdvPyro/qAdvChar` |
| table + Cantera advective differences | −35.6 K | +0.20 mm | recession stays matched; remaining surface gap is enthalpy/SEB coupling |

At PATO's own `T_wall` and inferred unblown `B'_g`, the checked-in 3-D B′ table
gives `B'_c ≈ 0.165` at 60 s, matching PATO's mass output. The live Cantera result
only matched recession after the Cantera-derived nominal pyrolysis-gas carbon
fraction (`Z_C_pyro≈0.49499`) was mapped to the full TACOT zeta family instead
of degenerating to pure CO. This separates the old
base-case mismatch into (1) a surface-composition error (`Z_C_pyro=None` used a
too-carbon-poor nominal gas), and (2) table-vs-live enthalpy support (`h_wall`,
`qAdv*`).

## 10. Energy formulation: d(ρh)/dt vs ρ·dh/dt — effect on mean slab density

**Background.** SCAM's in-depth FVM uses the exact energy storage
`d(ρh)/dt = (ρ_old·h(T^n,ρ_old) − ρ_new·h(T^k,ρ_new)) / Δt` (controlled by
`SolverOptions.use_rho_old=True`, the default). PATO's solver uses the
approximation `ρ·dh/dt ≈ ρ·cp·dT/dt`. The difference is the density-change term
`(ρ_old − ρ_new)·h̄_sensible / Δt` which appears implicitly on the LHS in SCAM
but is absent in PATO.

**Quantification.** The volume-averaged slab density (= `Σρᵢδᵢ / Σδᵢ` over the
remaining slab, identical between SCAM and PATO by definition) was compared using
`SolverOptions.use_rho_old=True/False` against the PATO `massLoss` file
(AblationTestCase_2.x, live Cantera backend, 201 nodes, ALE remap):

| t (s) | PATO ref | SCAM exact d(ρh)/dt | SCAM ρ·dh/dt (PATO approx) |
| ----: | -------: | ------------------: | -------------------------: |
| 10 | 277.867 | 277.709 (−0.158) | 277.950 (+0.083) |
| 20 | 276.922 | 276.597 (−0.325) | 276.973 (+0.051) |
| 40 | 275.727 | 275.149 (−0.578) | 275.735 (+0.008) |
| 60 | 274.786 | 274.004 (−0.782) | **274.774 (−0.012)** |
| 90 | 272.905 | 272.615 (−0.290) | 273.645 (+0.740) |
| 120 | 272.682 | 272.410 (−0.272) | 273.511 (+0.829) |

**Key findings:**

1. The exact d(ρh)/dt formulation produces ~0.78 kg/m³ lower mean density than
   PATO at t=60 s (peak ablation). The extra implicit `h̄_sensible` on the LHS
   routes more energy into in-depth decomposition, marginally lowering the slab
   density. The gap narrows slightly to 0.27 kg/m³ by t=120 s.

2. The `ρ·dh/dt` approximation (`use_rho_old=False`) nearly eliminates the
   **density** gap during ablation (−0.012 kg/m³ at t=60 s), but diverges
   after flux-off (+0.829 kg/m³ at t=120 s). However, it carries a **persistent
   ~20 K surface-temperature deficit** throughout the entire run
   (T_wall(60 s) = 1544.7 K vs PATO 1568.5 K = −23.8 K; this remains −18.8 K
   at t=120 s). The T_wall deficit arises because the `ρ·dh/dt` assembly
   (`Dc_thermal = M·T^n`) omits the `(ρ_old·h_old − ρ_new·h_k)·A·Δ/dt` term —
   a positive RHS contribution from the sensible enthalpy of the decomposing
   solid. Without it, less heat is available at each in-depth node, so the
   surface runs cooler by ~15 K. That deficit originates in the heating phase
   and propagates unchanged into cooldown, making `ρ·dh/dt` a poor match for
   PATO's T_wall even though it matches PATO's density at t=60 s.

3. By contrast, the exact d(ρh)/dt formulation tracks PATO T_wall to within
   ±7 K throughout the full run (heating peak −8.7 K; cooldown +6.6 K just
   after flux-off, settling to −1.2 K at t=120 s).

4. Follow-up diagnostics on the density divergence ruled out the simple suspects:
   - adding a lagged sensible `hs` source to `ρ·dh/dt` collapses back onto the
     exact curve;
   - using a lagged full PATO-default `h_bar*piTotal` source also collapses
     back onto the exact curve;
   - disabling the existing SCAM pyrolysis moving-mesh density-gradient
     correction makes density too high; flipping its sign is catastrophically
     wrong. SCAM's current ALE correction has the right sign and is not the
     missing PATO behavior.

5. **T_wall and mean density are governed by independent mechanisms** and cannot
   both match PATO simultaneously with either formulation.  T_wall is a surface
   quantity set by the SEB balance (radiation, Cantera h_wall, gas exit); it matches
   PATO when the surface chemistry is correct, regardless of the in-depth
   formulation.  Mean density is a volume integral of the Arrhenius decomposition
   field, controlled by the in-depth energy storage.  The exact d(ρh)/dt
   formulation includes the sensible enthalpy of the pyrolysing mass
   [(ρ_old·h_old − ρ_new·h_k)·A·Δ/dt] as a positive source in every decomposing
   node, raising local pyrolysis-zone temperatures slightly → more decomposition →
   ~0.76 kg/m³ lower mean density than PATO.  The ρ·dh/dt approximation omits
   this term → density matches PATO at t=60 s (+0.006 kg/m³) but T_wall is
   ~20 K too cold throughout the run.  The density gap in exact d(ρh)/dt is
   0.27 % of ρ_v — physically negligible; the T_wall gap in ρ·dh/dt is not.

6. **The exact formulation remains the better SCAM default.** Peak mean-density
   error is 0.27 % of ρ_v, and it avoids the large T_wall and cooldown-density
   errors of the ρ·dh/dt approximation. The `ρ·dh/dt` option is kept as a
   diagnostic comparison mode, not promoted to the production default.

7. The `SolverOptions.use_rho_old` flag is now wired into `indepth_solver.py`.
   Setting `False` passes `h_old=None, T_old_rhs=state.T` to the assembly,
   activating `Dc_thermal = M·T^n` (stable Picard, T^n fixed). The default
   (`True`) remains exact d(ρh)/dt with h_old at ρ_old.

The cleaned β-formulation comparison intentionally plots only exact `d(ρh)/dt`,
storage-only `ρ·dh/dt`, and the PATO reference; the closed-out diagnostic sweeps
are documented above rather than left as permanent plot clutter. The script is
[`examples/verification/ablation2/compare_pato_ablation2_beta.py`](../../examples/verification/ablation2/compare_pato_ablation2_beta.py).

## 11. Surface pyrolysis-gas advection and h_g enthalpy reference

**Background.** YAML material cards store pyrolysis-gas enthalpy `h_g(T)` on the
**sensible** reference (`h_g(298 K) ≈ 0`). PATO's gasProperties file and Cantera
use the **absolute** (formation-enthalpy) reference (`h_g(298 K) ≈ −7.09 MJ/kg`
for TACOT). The SEB advective term `q_adv_pyro = m_dot_g·(h_g(T_w) − h_wall)`
requires `h_g` and `h_wall` on the **same** reference, and `h_wall` comes from the
B′ lookup (Cantera/Mutation++ absolute reference). If `h_g` stays on the sensible
reference, the ~7 MJ/kg offset produces a large spurious heating source.

**`h_g_abs_offset` conversion.** `MaterialCard` carries an optional field
`h_g_abs_offset` (e.g. `−7.093e6` J/kg for TACOT), set in the material YAML.
`numerics/assembly.py::_build_system` applies this offset to every node's `h_g_cache`
entry when it is not `None`, converting the table from sensible to absolute reference
before the Q_adv computation and gas-storage source term. Scripts that replace
`h_g_table` with already-absolute data (e.g. PATO gasProperties or live Cantera)
**must** also set `h_g_abs_offset=None` to prevent double-application; all
`compare_pato_ablation*.py` scripts that call `dataclasses.replace(mat, h_g_table=...)`
include this clearance.

**Explicit surface outflow.** `_build_system` sets `mg_out = m_dot_g.copy()` (no
surface-node zeroing). The surface-node Q_adv is therefore:

```text
Q_adv[0] = m_dot_g[1]·h_g_abs(T[1]) − m_dot_g[0]·h_g_abs(T[0])
```

Combined with the SEB term `q_adv_pyro = m_dot_g·(h_g_abs(T_w) − h_wall)`, the net
surface energy contribution from the pyrolysis gas is:

```text
Q_adv_vol[0]·V[0] + q_adv_pyro ≈ m_dot_g[1]·h_g_abs(T[1]) − m_dot_g[0]·h_wall
```

At quasi-steady (T[0] ≈ T_w, T[1] ≈ T_w) this collapses to `−m_dot_g·h_wall`,
exactly matching PATO's `mDotGw·(h_g(T_w) − h_wall)` SEB term. The old SEB
`q_mass_removal` compensation block (applied when `not _bprime_ran`) is no longer
needed and was removed.

**Verification.** Running the TC2.2 benchmark (TACOT v2.2, Mutation++ B′ tables,
enthalpy BC with `rhoUeCH = 0.3` kg/m²/s, `h_r = 1.5` MJ/kg):

| t (s) | SCAM T_w (K) | PATO T_w (K) | Δ (K) |
|------:|-------------:|-------------:|------:|
| 10    | 1509.2       | 1511.9       | −2.7  |
| 30    | 1554.5       | 1555.5       | −1.0  |
| 60    | 1568.1       | 1568.4       | −0.3  |

Ablation2 base (live Cantera): T_w(60 s) = 1569.3 K vs PATO 1568.5 K (+0.8 K).
326 unit/integration/verification/validation tests pass.

Full derivation of the earlier cooldown gap (superseded fix): `docs/physics/ablation2_cooldown_gas_advection.md`.

---

### Performance notes

Runtime has been progressively optimised through two passes. The main changes are:

- `numerics/assembly.py` evaluates `k`, `cp`, `h_bar`, and gas enthalpy by
  material/layer using vectorized property functions instead of per-node Python
  loops; Q_darcy and gas-storage source terms are also vectorised by layer.
- `compute_F_cond` returns the pre-built `TridiagSystem`; `indepth_solver.py`
  patches `D[0]` with the converged `q_cond` instead of assembling the same
  system twice.
- `physics/pyrolysis_gas.py` uses a reverse cumulative sum for
  `m_dot_g_nodes`.
- `physics/darcy_flow.py` vectorizes the constant-pressure gas expansion flux
  and energy source by layer.
- `physics/element_transport.py` pre-computes `storage` and the `Z_pyro (4,N)`
  elemental-fraction matrix layer-wise before the element loop, assembles the
  tridiagonal via NumPy face-flux operations, and uses `np.where` for degenerate
  nodes — eliminating the former 4×N per-node Python loop.

Measured on the default 501-node live-Cantera cases: base `compare_pato_ablation2.py`
runs in ~11 s; `compare_pato_ablation2_equilibriumElementConservation.py` (Cantera +
element transport) runs in ~75 s (was ~152 s before element-transport vectorisation).
Current exact-storage results are unchanged by the optimisation itself: base
`T_wall(60 s)=1563.9 K`, recession `12.08 mm`; EquilibriumElement
`T_wall(60 s)=1563.1 K`, recession `12.11 mm`.

### Ablation1 multiPorousMat 5 mm probe

[`compare_pato_ablation1_multiPorousMat.py`](../../examples/verification/ablation1/compare_pato_ablation1_multiPorousMat.py)
matches PATO's `AblationTestCase_1.0_multiPorousMat`: a 1 cm TACOT porous layer
at the hot face over a 1 cm inert cork layer. The important PATO detail is that
`porousMat1Properties` uses `PyrolysisType LinearArrhenius`; the first layer is
not inert char. Treating it as inert made SCAM's 5 mm TACOT probe run 70-90 K
too hot after the initial heat-front transient.

SCAM now enables the reduced TACOT two-resin decomposition model for
`porousMat1` (matrix fractions 0.25 and 0.19+0.06, density 280 kg/m3 -> 220
kg/m3) with explicit gas storage, while leaving the cork layer inert. This
collapses the broad late-time 5 mm discrepancy: representative differences are
about -13 K at 60 s, -8 K at 90 s, and -4 K at 120 s. The reported full-run
max remains about 73 K because the largest remaining discrepancy is the early
arrival transient near 10 s, where PATO heats the 5 mm probe faster. Mesh
refinement, timestep caps, scalar density changes, gas-storage toggles, and
char/virgin property overrides did not remove that early mismatch as cleanly as
enabling the missing decomposition.

### Multi-material 2.x stack

[`compare_pato_ablation2_multiMat.py`](../../examples/verification/ablation2/compare_pato_ablation2_multiMat.py)
extends the same corrected 2.x surface physics to PATO's layered
`AblationTestCase_2.x_multiMat`: a 5.80 cm TACOT_v3 ablating layer over two
inert Fourier sublayers (0.14 cm and 1.27 cm), with the same
`rhoUeCH = 0.3`, `h_r = 1.5e6`, `Tbackground = 300 K`, Darcy flow, recession,
and adiabatic back face.

The SCAM deck mirrors the PATO layer order with `StackConfig.layers[0]` as the
hot-face TACOT layer, then the two Fourier layers. TACOT is loaded from
`scam/materials/ablative_organic/tacot_v3.0.yaml`, and the inert Fourier sublayers
are loaded from `scam/materials/subsurface/fourier.yaml`. It uses the live Cantera
backend with the Cantera-derived nominal `Z_C_pyro` fallback, the SEB advective
enthalpy terms, material-blended TACOT emissivity, and continuous ALE recession.
The sub-layer thermocouples are compared at fixed original-surface depths
58.7 mm and 62.1 mm, matching PATO's bottom-origin probe coordinates.

Current command:

```bash
MPLBACKEND=Agg python3 examples/verification/ablation2/compare_pato_ablation2_multiMat.py
```

Current result (2026-06-13 run): SCAM reaches `T_wall ≈ 1568.0 K` and
`s ≈ 12.04 mm` at 60 s, versus PATO `T_wall = 1568.57 K`; final recession is
`12.06 mm`. The full-run max in-depth differences are largest in the TACOT
pyrolysis zone during cooldown (about 60–85 K at the 1–16 mm probes), while the
deep TACOT probe and both Fourier sublayer probes remain essentially matched
(≤ 1 K reported for 45.3 mm, subMat1, and subMat2). The regenerated plot is written to
`examples/verification/ablation2/compare_pato_ablation2_multiMat.png`
(not tracked in git; regenerate it by running the command above).

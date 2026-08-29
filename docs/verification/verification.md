<!-- SPDX-License-Identifier: MIT -->
# SCAM Verification Plan

This document describes the **verification ladder** in `tests/verification/`: a
sequence of whole-solver runs of increasing physical complexity, each checked
against a closed-form or independently-computed reference. Every rung adds
exactly one physics mechanism on top of the one below it, so a failure localizes
the regression to a single mechanism.

```
V1  bare conduction, constant properties        → erfc / linear analytical
V2  conduction, temperature-dependent k, cp      → Kirchhoff steady profile
V3  multi-material stack (+ contact resistance)  → series thermal resistance
V4  in-depth decomposition / pyrolysis            → invariants + scipy kinetics
V5  coupled surface mass & energy balance          → SEB closure + audits
```

Run the whole ladder, in order:

```bash
pip install -e ".[dev]"
MPLBACKEND=Agg pytest tests/verification/ -v
```

Run a single rung, e.g. V3:

```bash
pytest tests/verification/test_v3_multilayer.py -v
```

The ladder is self-contained: shared input builders and analytical-reference
functions live in `tests/verification/conftest.py`; nothing about the solver is
re-implemented there.

---

## A note on discretization error

SCAM places the surface and back-face nodes **on** the boundary as half-cells
(thickness `h/2`). This is convenient for applying boundary conditions but makes
the boundary treatment **first-order accurate**: the steady conduction profile
carries a small error at the two nodes adjacent to each boundary that **halves
when the grid is halved** (verified directly in V1/V2). The interior is second
order. Tolerances below distinguish this expected, convergent discretization
error from a genuine bug (which would not converge, or would be far larger).

---

## V1 — Conduction, constant properties

**Mechanism isolated:** the bare heat equation — FVM assembly, Thomas solve,
backward-Euler time integration. No decomposition, recession, or SEB.

**Configuration:** single inert layer with constant `k`, `rho`, `cp`
(`make_inert_material`), slab geometry, adiabatic back unless noted.

| Sub-case | BC | Reference | Criterion |
|----------|----|-----------|-----------|
| Semi-infinite, prescribed flux | `PRESCRIBED_FLUX` `Q0` | `T = T0 + (2Q0/k)√(αt/π)e^{−ξ²} − (Q0 y/k)erfc ξ`, `ξ = y/(2√(αt))` | max rel. err < 4% over nodes with `ΔT > 50 K` |
| Semi-infinite, prescribed temp | `PRESCRIBED_TEMP` `T_w` | `T = T0 + (T_w−T0)erfc ξ` | surface pinned to 0.01 K; max rel. err < 5% (steeper near-surface gradient) |
| Finite slab, both ends fixed | `PRESCRIBED_TEMP` front & back | linear profile `T(y)=T_f+(T_b−T_f)y/L` | fine-grid err < 3 K **and** error ratio ≈ 2 on grid halving (first-order convergence) |

The transient tolerances (4–5%) reflect backward-Euler diffusivity at large
Fourier number; they tighten with smaller `dt_max`.

---

## V1b — Transient conduction (constant properties, multi-snapshot)

**What it adds over V1:** V1 compares only the *final* state; V1b tracks the
full transient by comparing the temperature profile against the analytical
solution at four time snapshots (`t = 2, 5, 10, 20 s`). This makes the
backward-Euler phase error and its decay with Fourier number visible.

**Script:** `examples/v1b_conduction_transient.py`

**Configuration:**

| Parameter | Value |
| --------- | ----- |
| k | 1.0 W/m/K |
| ρ | 180 kg/m³ |
| cₚ | 710 J/kg/K |
| α = k/ρcₚ | 7.84 × 10⁻⁶ m²/s |
| Slab thickness | 0.10 m (semi-infinite throughout the run) |
| Grid | 201 nodes, Δy ≈ 0.5 mm |
| dt_max | 0.05 s |
| T₀ | 300 K |

**Analytical references (Carslaw & Jaeger):**

*Sub-case 1 — Prescribed surface flux Q₀:*

$$T(y,t) = T_0 + \frac{2Q_0}{k}\sqrt{\frac{\alpha t}{\pi}}\,e^{-\xi^2} - \frac{Q_0\,y}{k}\,\mathrm{erfc}(\xi), \qquad \xi = \frac{y}{2\sqrt{\alpha t}}$$

*Sub-case 2 — Prescribed surface temperature T_w (step change):*

$$T(y,t) = T_0 + (T_w - T_0)\,\mathrm{erfc}\!\left(\frac{y}{2\sqrt{\alpha t}}\right)$$

Both are valid as long as the thermal penetration depth `2√(αt)` stays well
within the slab. At `t = 20 s`: `2√(αt) ≈ 25 mm` vs 100 mm slab — the back
face is unaffected.

**Error metric:** pointwise relative error `|T_num − T_ana| / (T_ana − T₀)`,
evaluated only where `T_ana − T₀ > 50 K` (masking the undisturbed region where
the denominator is negligible and relative error is meaningless).

**Acceptance criteria:**

| Sub-case | t [s] | Max rel. err |
| -------- | ----- | ------------ |
| Flux BC | 2 | ≤ 6% |
| Flux BC | ≥ 5 | ≤ 5% |
| Temp BC | 2 | ≤ 8% |
| Temp BC | ≥ 5 | ≤ 5% |

**Error budget — spatial dominates.**  
The erfc temperature front spans roughly `4√(αt)` in depth. At `t = 2 s` that
is ~14 mm = ~28 nodes, so the gradient is steep relative to the grid and the
half-cell boundary artifact is the leading error. Doubling the node count (101 →
201) halves the early-time error from ~16% to ~8%; halving `dt_max` has no
measurable effect. This is expected first-order spatial convergence.

At later times the front broadens, more nodes resolve it, and the error falls
below 3% even at the coarser grid.

---

## V2 — Conduction, temperature-dependent properties

**Mechanism added:** `k(T)` table interpolation and the virgin/char blending in
`scam/physics/properties.py`. cp constant.

**Reference — Kirchhoff transform.** Define `θ(T) = ∫_{T_back}^{T} k dT′`. At
steady state `d/dy(k dT/dy) = 0` forces `θ(y)` to be **linear** in `y` for any
`k(T)`:

```
θ(T(y)) = θ(T_front) · (1 − y/L)
```

For `k(T) = a + b T` this inverts to a quadratic giving the reference profile;
the steady surface flux is `q = θ(T_front)/L`.

**Criteria:** profile converges first-order to the Kirchhoff reference
(fine-grid err < 1% of the temperature span, ratio ≈ 2 on grid halving); steady
`q_cond` matches `θ(T_front)/L` within 2%.

---

## V3 — Multi-material stack

**Mechanism added:** layer-to-layer interface conductance and optional thermal
contact resistance (`scam/geometry/fvm.py::interface_conductance`,
`scam/numerics/assembly.py`).

**Configuration:** two inert constant-property layers (`K1=2.0`, `K2=0.5`),
fixed end temperatures, run to steady state.

**Reference — series thermal resistance:**

```
q              = (T_front − T_back) / (L1/k1 + R + L2/k2)
dT/dy|_layer   = −q / k_layer
interface drop = q · R
```

**Criteria:**
- **Flux** `q_cond` matches the series value within 1% (`R = 0`).
- **Per-layer slopes** (interior, away from boundaries) match `−q/k` within 2%.
- **Contact resistance** is verified *incrementally* to cancel the half-cell
  boundary artifact: the extra interface temperature jump when `R` is switched
  on, `jump(R) − jump(0)`, equals `q·R` within 5%.

---

## V4 — In-depth decomposition & pyrolysis (prescribed surface temperature)

**Mechanism added:** nodelet Arrhenius decomposition, pyrolysis gas generation,
and the decomposition/gas enthalpy source terms (`scam/physics/decomposition.py`,
`scam/physics/pyrolysis_gas.py`). The surface temperature is *prescribed* (a
ramp to 1800 K), so the SEB and B′-table char ablation are excluded — that is
V5. Real TACOT is the charring material.

There is no closed-form coupled solution, so the references are invariants plus
an independent ODE integration:

| Check | Reference | Criterion |
|-------|-----------|-----------|
| Density bounds | physics | `rho_char ≤ rho ≤ rho_virgin` every node, every step |
| Surface chars | physics | sustained-hot surface `rho → rho_char` (within 5% of span) |
| Cold interior | physics | adiabatic back stays at `rho_virgin` (no spurious decomposition) |
| Pyrolysis ↔ recession | CMA: `s_dot = ṁ_char/ρ_char` (→ 0 here, no B′ table) | `ṁ_pyro > 0` yet `s_dot = 0` every step |
| Inert control | physics | inert layer: no gas, no recession |
| Nodelet kinetics | scipy `solve_ivp` (rtol 1e-9) of `dρ_i/dt = −k(T)ρ_0((ρ_i−ρ_r)/ρ_0)^m` | < 0.1% after 200 steps at fixed `T` |

**Behavior surfaced by this rung.** SCAM's surface mass balance
(`scam/physics/recession.py`) follows the standard CMA model `s_dot =
ṁ_char/ρ_char`: **only char consumption drives recession.** Pyrolysis gas is
generated in-depth and percolates out through the porous char via Darcy flow —
it leaves the char skeleton intact and does **not** recede the surface. V4 is a
prescribed-temperature run with no B′ table, so `ṁ_char = 0` and therefore
`s_dot = 0`: the surface does not recede and no node drops occur, even though
pyrolysis gas is being produced (`ṁ_pyro > 0`). This is the physically correct
result and lets the rigorous mass/kinetics verification run free of any
recession/remap coupling.

---

## V5 — Coupled surface mass & energy balance (full ablation)

**Mechanism added:** the full `ENERGY_BALANCE` path — SEB Newton solve for
`T_wall`, B′-table char ablation, blowing correction, recession, node
drop/merge. This is the TACOT arcjet regime of
`tests/verification/test_tacot_benchmark.py`; the verification rung adds
*conservation/consistency* checks rather than only plausibility bounds.

Each check rebuilds the governing balance from the stored snapshots:

| Check | What is verified | Criterion |
|-------|------------------|-----------|
| **SEB closure** | the surface energy balance, rebuilt from reported `T_wall / q_cond / ṁ_char / ṁ_pyro` (with `h_w` and `ṁ_char` re-looked-up from the B′ table), is ~0 | `\|residual\| < 100·seb_tol` every post-ignition step (observed ~1e-2 W/m²) |
| **Recession** | CMA: `s_dot = ṁ_char/ρ_char` per step; `∫ s_dot dt` matches reported `s_total` | identity to 1e-6; integral within 5% |
| **Blowing** | `alpha_eff ≤ alpha_conv` at run conditions; strictly decreasing in `B'` | monotonic |
| **Stability/plausibility** | finite `T`, `rho`; physical end state | `T_wall ∈ (1500,4000) K`, `s_total ∈ (0.1,25) mm`, `s_dot > 0` |

**Why SEB closure is the energy check.** A separate global volumetric energy
audit on a receding, node-dropping mesh is dominated by frame-change bookkeeping
artifacts and would be flaky. The per-step SEB residual is the surface energy
conservation statement that actually governs the coupling, and verifying it
closes to ≪ 1 W/m² at every output step is the strongest available check.

---

## Acceptance criteria summary

| Rung | Mechanism added | Reference | Tolerance |
|------|-----------------|-----------|-----------|
| V1 | bare conduction | erfc (flux & temp), finite-slab linear | 4–5% transient; first-order steady convergence |
| V2 | T-dependent k, cp | Kirchhoff steady profile | 1% profile (convergent) / 2% flux |
| V3 | multi-layer + contact R | series resistance, ΔT = qR | 1% flux / 2% slope / 5% contact jump |
| V4 | decomposition + pyrolysis | invariants + scipy ODE | 0.1% kinetics; invariants exact |
| V5 | SEB + ablation + recession | SEB closure + conservation audits | ≪ seb_tol; 5% recession integral |

## Interpreting a failure

A rung that fails while lower rungs pass localizes the broken mechanism — e.g.
V2 green but V3 red ⇒ interface conductance / contact-resistance handling. If a
rung reveals a genuine solver bug, report it with the failing assertion and the
reference value rather than relaxing the tolerance.

---

## Current Verification Status (as of 2026-06-21)

### PATO AblationTestCase_1.x and 2.x — Verified (minor open items)

All ablation2 verification cases against modern PATO provide acceptable results.
Minor improvements that remain:

1. **Internal energy discrepancy (small):** A slight difference in energy storage
   between SCAM and PATO appears in `compare_pato_ablation2_beta.py` (mean density
   vs time) and as a very small depth-increasing TC offset in all ablation2 cases.
   Source not yet pinned down.

2. **Cooldown discrepancy — RESOLVED 2026-07-16 (PATO reference artifact):**
   `compare_pato_ablation2_equilibriumElementConservation.py` shows SCAM up to
   +28 K warmer than the PATO reference during cooldown (peak t≈70 s, decaying
   to +14 K at 120 s). Root cause is on the **PATO side**, not SCAM:

   - SCAM's cooldown is identical (±0.4 K) between the EEC and base-2.x setups,
     and matches PATO's *base* 2.x reference within 9.4 K. It is **PATO's own
     EEC reference that cools up to 22.6 K faster than PATO's base case**
     (identical at t=60 s, diverging after flux-off as a surface-launched wave).
   - Instrumented local PATO 3.1 reruns of both cases (field output every 1 s;
     both reproduce their shipped references to ≤0.17 K) show the EEC case's
     `GasPropertiesType Equilibrium` viscosity is unphysical: Mutation++'s
     `tacotair` mixture is missing collision-integral data for most species
     pairs ("Warning: missing collision integral Q11_(C,CH4). Using a constant
     value of 1e-20", etc.), giving `mu_g` ≈ 27–62·10⁻⁵ Pa·s that *rises* as T
     falls — 6–27× the physical Tabulated values (2.3–4.9·10⁻⁵) and inverted in
     trend. h_g and M_g are fine (match the table to 4 digits at nominal Z).
   - The garbage μ throttles the Darcy mobility: same `mDotG`, but 6–16×
     steeper in-depth ∇p, an elevated wall pressure, and a mass-flux dip in the
     wall cell as the cold-wall mobility collapses during cooldown. The
     associated `∇·(GammaHg·∇p)` gas-enthalpy imbalance deposits a ≈1.2 kW/m²
     net cooling sink in the top mm (measured at t=62–65 s), matching the
     ρcp·d(ΔT)/dt of the observed divergence. During heating the B′ SEB
     (∼MW/m²) masks the same artifact, which is why the two references agree
     until flux-off.
   - Conclusion: SCAM's +28 K "overestimate" vs this reference is an artifact
     of the reference's equilibrium transport properties. Do **not** chase it
     by modifying SCAM physics. Comparing against the base 2.x reference (or an
     EEC rerun with Tabulated gas properties) is the meaningful check.
   - Related script hygiene fix (small, ~1 K): the EEC compare script replaces
     `h_g_table` with the Cantera equilibrium h_g and must also clear
     `mat.gas_properties_pT = None`, otherwise the card's (p,T) table silently
     shadows the injected equilibrium h_g in `pyrolysis_gas_enthalpy_abs`
     (same trap class as `h_g_abs_offset`, pato_validation.md §11).

3. **multiPorousMat (ablation1):** `compare_pato_ablation1_multiPorousMat.py` now
   replicates PATO's `virginOrChar char` blending (cp/k/eps_g/h_virgin_sensible all
   forced to char values). Peak gap reduced from 73 K to ~48 K. Residual 48 K from
   decomposition energy mismatch: SCAM Q_vol carries h_bar_absolute ≈ −4429 kJ/kg
   while PATO with hs=0 applies only hp = −4000 kJ/kg. Further closure would
   require hardcoding char absolute enthalpies — physically wrong, not pursued.

### Ablation Test Case Series (Amaryllis reference) — Mostly OK, open symptoms

TC 2.2 and 2.3 work well with tabulated B′. TC 2.2 also works with live Mutation++
or Cantera. TC 2.3 fails with live Mutation++ / Cantera (works with table only).
TC 2.1 matches Amaryllis well overall but SCAM misses the 98% pyrolysis front.


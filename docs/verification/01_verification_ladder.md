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

**Script:** `examples/verification/conduction/v1b_conduction_transient.py`

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

## CHyPS BlaineTest comparison — VERIFIED (re-measured 2026-09-07)

**Script:** `examples/verification/chyps/compare_chyps_blaine.py`  
**Reference data:** `examples/verification/chyps/BlaineTest_chyps_results/` — courtesy of **Dr. Blaine Vollmer (University of Illinois at Urbana-Champaign)**, redistributed with permission; probe files are column-trimmed copies of the original CHyPS output (numerically identical for this comparison).  
**Material:** `tacot_v3.0.yaml` (2-rxn), live Cantera backend. The 3-rxn card
gives an equivalent result (see below); this section previously said 3-rxn while
the script used 2-rxn.  
**Case:** 5 cm slab, two-phase heating (0–60 s: ρu_e C_H ramps 0.2→0.3, h_r = 1.5 MJ/kg; 60–90 s: h_r = 15 MJ/kg; cooldown 90–100 s)

### Current status — agreement is excellent everywhere

| Quantity | t=30 s | t=60 s | t=90 s | t=100 s |
|---|---|---|---|---|
| Surface T (SCAM − CHyPS) | −0.4 K | +0.5 K | −3.2 K | +10.8 K |
| Recession | −0.025 mm | −0.040 mm | −0.062 mm | −0.064 mm |
| T6 (16 mm) | −3.7 K | **+5.9 K** | +0.7 K | +15.0 K |
| T7 (24 mm) | −4.4 K | −5.0 K | +5.0 K | +5.1 K |

Max surface deviation: 12.9 K in phase 1, **3.2 K** in the high-flux phase 2,
11.5 K in cooldown. Recession error is 0.45% of 14.07 mm. The short-lived probes
T1–T5 (consumed by recession) overlay the reference.

Running the 3-rxn card instead (`tacot_v3.0_3rxn.yaml`) gives T6 +0.5 K and
T7 −7.0 K at t = 60 s — equivalent, so the kinetics variant is not a factor.

### Superseded: the "CHyPS omits h_bar" root cause is WRONG

An earlier version of this section recorded deep probes T6/T7 running **~126 K
colder** than CHyPS, and attributed it to CHyPS's solid enthalpy lacking a
composition-dependent (`h_bar`) term — concluding that running SCAM with
`h_bar ≡ 0` "confirmed T6 to match CHyPS better."

**That conclusion is now inverted by direct test.** SCAM still applies `h_bar` at
full strength (`h_bar_chemical = −4.43 MJ/kg` for TACOT v3.0, tables present), and
it is *required* for the agreement above. Re-running with `h_bar_chemical` nulled
(virgin absolute-enthalpy table shifted by +948982 J/kg so `h_bar_chem` = 0 exactly,
leaving cp/k untouched):

| Probe | h_bar active | **h_bar_chem = 0** |
|---|---|---|
| T6 @60 s | +5.9 K | **+227.0 K** |
| T6 @100 s | +15.0 K | +143.4 K |
| T7 @90 s | +5.0 K | **+273.5 K** |
| T7 @100 s | +5.1 K | +254.9 K |

Removing the decomposition sink makes SCAM 220–275 K **too hot** at depth. So
CHyPS evidently *does* carry an equivalent composition-dependent decomposition
energy sink, and the original 126 K deficit was a **SCAM-side bug, since fixed** —
not a CHyPS modelling omission. The old probe-data evidence (`Solid1/enthalpy ==
Solid2/enthalpy`) may still be literally true of that particular output field
while the energy still enters CHyPS's equation by another route; it should not be
read as proof the sink is absent.

**Attribution not pinned.** The `h_bar` sensible/chemical split (`3dd2810`,
2026-06-14) predates the 126 K measurement (written 2026-06-17), so it is not the
fix. The leading candidate is `a43e31d` (2026-06-21) "Fix h_g enthalpy reference:
`h_g_abs_offset` + explicit surface outflow" — a −7.09 MJ/kg reference error in
the in-depth gas enthalpy would act precisely in the pyrolysis zone feeding the
deep probes (see pato_validation.md §11). A bisect over 06-17→07-03 would settle
it if the provenance matters.

---

## Amar (2006) §8.7 — Carbon-Carbon Thermochemical Ablation

**Scripts:** `examples/verification/amar_thesis/compare_amar_cc.py` (§8.7),
`examples/verification/amar_thesis/compare_amar_cp.py` (§8.8, verified)

> **Not in the public distribution.** §8.7 is unresolved (see below), so
> `compare_amar_cc.py` and the §8.7 diagnostic `debug_bprime.py` are not shipped
> on `public-release`, nor are the Fig 8.16/8.19/8.20/8.21 reference curves they
> read. The verified §8.8 carbon-phenolic case **is** published and stands alone
> — the machinery the two share lives in `_amar_common.py`. This section is kept
> because the analysis below is the record of why §8.7 is still open.
**Material:** `carbon_carbon_amar2006.yaml` (Table 8.12, non-decomposing C-C)
**BC:** Digitized ballistic reentry trajectory from Figs 8.14–8.15 (recovery
enthalpy, edge pressure, heat transfer coefficient vs time; turbulent flow from
t ≈ 23.5 s)
**Reference:** CMA code, SODDIT, and Amar research code (Figs 8.16, 8.19–8.21)

### Configuration

- Thickness 12.7 mm (0.5 in), 100 nodes, T₀ = 297.04 K (~534°R, from
  the initial values in Figs. 8.16, 8.19, and 8.20)
- `ENERGY_BALANCE` BC with `rhoUeCH` + `h_r` (enthalpy mode), `C_M = 1.0`
- Amar et al. (2008) Stanton corrections:
  - laminar/stagnation hot-wall correction, Eq. (37), before 23.5 s
  - turbulent Eckert reference-enthalpy correction, Eqs. (38)–(42), after 23.5 s
  - exact Kays blowing correction `Φ/(exp(Φ)-1)`, Eqs. (31)–(33)
  - λ = 0.5 laminar and 0.4 turbulent
- Fig. 8.15 is treated as the heat-transfer coefficient used for aerodynamic
  heating; the hot-wall correction is applied to the mass-transfer/recession
  relation only. Applying it to Fig. 8.15 again double-corrects the heat flux
  and produces substantially worse temperatures.
- Radiation to deep space (T_rad_in = 0), ε = 0.8
- ALE recession (`continuous_remap = True`)
- Two B′ table variants run and overlaid:
  - **Cantera CNO** (`calcarb_bprime_air_mp.yaml`): full species set (C, C2, C3), 6 pressures 0.001–5 atm
  - **ACE-style** (`calcarb_bprime_ace_air.yaml`): GRI-Mech 3.0 (C only, no C2/C3), 6 pressures 0.001–5 atm

### Results (ε = 0.8, T_rad_in = 0) — regenerated 2026-09-07

Current script configuration: faithful `Ω_hw` applied to **both** heat and mass
(`apply_wall_correction_to_heat=True`, per next-step #2 below), both B′ tables run.

| Quantity | SCAM CNO | SCAM ACE | Amar Table 8.14 | Note |
|---|---|---|---|---|
| T_surf peak | 6221°R @30.0 s | 6293°R @30.0 s | 6508°R @29.8 s | −215 to −287°R |
| T_surf @40 s | 4300°R | 4313°R | ~5105°R | ≈ **−800°R** |
| T_surf @49 s | 3834°R | 3845°R | ~4970°R | ≈ **−1125°R** |
| T_back peak | 4832°R @**37.8 s** | 4849°R @**37.6 s** | 5094°R @**41.5 s** | −250°R, **~4 s early** |
| ṡ peak | 7.41×10⁻³ in/s @30.8 s | 6.93×10⁻³ in/s @31.3 s | 9.251×10⁻³ in/s @30.5 s | **20–25% low** |

> **Two reference values are in circulation for §8.7, and both appear below.**
> The *digitized* Figs 8.16/8.19–8.21 curves give T_surf peak ≈ 6541°R and T_back
> peak ≈ 5113°R @42.0 s; Amar's tabulated **Table 8.14** gives 6508°R @29.8 s and
> 5094°R @41.5 s. The table above quotes Table 8.14 (what the script prints as its
> target); the root-cause discussion below quotes the digitized values. They differ
> by ~33°R / ~19°R / 0.5 s — digitization noise, not a discrepancy, and far smaller
> than any gap under discussion.

> **Historical note.** An earlier version of this table reported SCAM (ACE) peak
> 6648°R and ṡ 7.26×10⁻³. Those were produced with the old
> `apply_wall_correction_to_heat=False` band-aid, which the script no longer uses
> (see #4). Under the faithful correction the peak drops to 6221–6293°R, i.e. the
> peak is now **under** the reference rather than slightly over. Note also that
> CNO now gives *more* recession than ACE (7.41 vs 6.93), the inversion #4
> predicted — though both remain well short of 9.25.

The qualitative failure: SCAM is somewhat cold at the 30 s peak, and then its
surface collapses far too fast. Figs 8.19/8.20 show the reference slab goes
**isothermal at ~5000°R (2780 K) and stays flat from 40–50 s** with the adiabatic
back face; SCAM falls to ~3840°R. The back-face peak also arrives ~4 s early, and
peak recession is 20–25% low.

### Root cause #2: the cooldown surface radiation (resolved — §8.7 reference anomaly)

This is an energy-balance impossibility that pins the diagnosis. The reference
slab is isothermal at 2780 K, adiabatic-backed, cooling at only ~3–8 K/s — so its
**net surface loss is ~0.18 MW/m²**. But at 2780 K, SCAM computes

- radiation `εσT⁴` to deep space ≈ **2.6 MW/m²**
- convective cooling `ρ_e u_e C_h (h_r − h_w)` ≈ **0.9 MW/m²** (h_r→0, h_w≈1.65 MJ/kg)

i.e. SCAM sheds **~3.5 MW/m², roughly 20× the reference**, which is exactly why
its surface collapses. An emissivity sweep confirms radiation is the controlling
lever (Tw@49 s: 3809→4343°R as ε 0.8→0.3), **but a uniform ε reduction overshoots
the peak** (6648→7046°R). The error is therefore *time-dependent*: full `εσT⁴` is
about right at the 30 s peak and far too strong during 35–50 s. That is precisely
the signature of Amar's radiative BC (thesis Eq 5.53):

```text
q_rad = ε σ (T_w⁴ − T_res⁴)
```

with a **non-zero, time-varying reservoir temperature `T_res(t)`**, which SCAM
currently hard-codes to 0 (deep space, `T_rad_in = 0`). The obvious physical
candidate — `T_res` = recovery/shock-layer temperature — is only 300–860 K during
cooldown (verified by equilibrating air at `h_r(t)`), too cold to matter, so
**`T_res` for §8.7 is a genuine unknown in Amar's input** that must be recovered.

A **constant-`T_res` sweep** (0 / 1500 / 2000 / 2300 / 2500 K) confirms a single
value cannot work: raising it lifts the cooldown tail (Tw@49 s 3809→4604°R) but
*monotonically overshoots the peak* (6648→6950°R vs ref 6541) and the back-peak
magnitude (5123→5721°R vs ref 5113), trading a cooldown error for a peak error —
even at 2500 K the cooldown is still 366°R short while the peak is 409°R over. The
reservoir must therefore be **time-dependent** (≈0 near the 30 s peak, ≈2500–2800 K
during 40–50 s). Two by-products of the sweep: (a) the **back-peak stays at ~37 s
for every `T_res`** (ref 42 s) — radiation does not fix the timing, so the ~5 s
lead is a separate conduction/diffusivity signature; (b) peak recession barely
moves (7.26→7.85×10⁻³ in/s, ref ~8.9), so the ~15% recession deficit is
independent of the reservoir and tracks the `apply_wall_correction_to_heat=False`
asymmetry (#4).
The boundary inputs themselves are clean: digitized `h_r`, `p_e`, HTC and `u_e`
match Figs 8.14/8.15, and `h_r → 0` at 42 s is consistent with the edge velocity
(`h_r ≥ r·u_e²/2` holds to digitization noise).

**§8.8 cross-check — the radiation reservoir is a dead end; SCAM's physics is
validated, the §8.7 reference cooldown is the anomaly.** Three results overturn the
"recover a hot `T_res(t)`" line:

1. The Amar 2008 paper states the reservoir temperature *is* a time-dependent
   user input (alongside `h_r`, `ρeueCH`). The thesis prints it only for §8.8:
   **`T_res = 414°R` (230 K)**. Running §8.7 with `T_rad_in = 230 K` is
   **byte-identical** to deep space (6292/4314/3845) — 230 K radiates 0.13 W/m²,
   negligible. So the *documented* reservoir is cold.
2. §8.8 carbon-phenolic uses the **same aerodynamic-heating BC** as §8.7 (thesis
   §8.8.1) with that cold `T_res`, and its reference (Fig 8.28) surface peaks
   ~6200°R then **cools normally to ~2200°R** by 50 s, with the back face barely
   warming (537→720°R — phenolic is an insulator). So the cold reservoir produces
   *physical* cooling. A hot reservoir is therefore **not** what Amar used.
3. **SCAM reproduces §8.8**, and as of 2026-09-07 it does so almost exactly:
   surface peak 6206°R @29.0 s (ref ~6137) and back face 535→**713**°R against a
   reference 715°R — i.e. **−2°R**, i.e. the whole back-face history tracks Fig
   8.29 within ±10°R (see §8.8 above; the earlier +240°R back-face error quoted
   here was Amar's erroneous Table D.4 char cp, since superseded by Sutton
   TN D-5930). SCAM's radiation + conduction + cooldown physics is therefore
   validated on the insulator with plain deep-space radiation, to a much tighter
   tolerance than when this argument was first written.

This deepens the §8.7 paradox into an **energy-conservation contradiction in the
reference itself**: a 12.7 mm carbon-carbon slab radiating `εσ(0.8)·(2780 K)⁴ ≈
2.6 MW/m²` cools at `2.6e6/(ρCpL) ≈ 60 K/s` *independent of conductivity* (high `k`
only makes it isothermal, it does not change the lumped heat capacity), yet the
§8.7 reference cools at <8 K/s — i.e. it effectively radiates ~7× less than
`εσ(0.8)T⁴`, with the *same* BC that makes §8.8 radiate fully. No reservoir or
emissivity consistent with §8.8 can produce that. SCAM, validated against §8.8,
predicts the carbon-carbon surface *does* cool — the physically correct behaviour.

**Revised conclusion:** the §8.7 reference (CMA/SODDIT/research-code) cooldown
appears physically anomalous (or its digitized late-time curve is being misread);
SCAM is not obviously wrong here. **§8.8 carbon-phenolic is the more trustworthy
verification target** and SCAM already matches it. The §8.7 peak (T and recession)
is still meaningful and is governed by #4 (faithful `Ω_hw`) + the B′ table; the
cooldown should be treated as an open reference-data question, not a SCAM defect.
**The gap is in the surface closure / reference data, not the in-depth solver.**

### Root cause #1 REFUTED: density is not the driver

The earlier hypothesis blamed the unsupported 1602 kg/m³ bulk density. Direct
test (ρ = 1602 / 1920 / 2200 kg/m³):

- raising ρ delays the back-peak (37.8→40.0 s) but **lowers its magnitude**
  (5123→4523°R, away from the 5113°R target) and **worsens recession**
  (7.26→5.13×10⁻³ in/s);
- the cooldown surface barely moves (3809→3745°R at 49 s).

Density cannot close the ~1200°R cooldown gap and trades against recession and the
back-peak magnitude. **Keep 1602 kg/m³**; it is a defensible C/C value and is not
the error.

### Root cause #3 (secondary): missing surface advective/ablation enthalpy

Amar's SEB (Amar et al. 2008, Eq 28) carries an **explicit** ablation flux
`q_abl = ṁ''·h_w` alongside the aero term, with the incoming solid enthalpy
handled by grid advection — net surface contribution `ṁ_c·(h_c − h_w)`. That is
exactly SCAM's `q_adv` term, which is **zeroed for pre-computed `BPrimeTable`
runs** (only the live `BprimeEvaluator.surface_enthalpies(...)` activates it; see
`physics/surface_energy.py` §8). This case used a table → the term was dropped.
For pure carbon it is ~+0.7 MW/m² heating at peak (`ṁ_c≈0.27`, `h_c−h_w≈+2.6
MJ/kg`), ~0.1 at 49 s.

**Tested and ruled out (negligible).** Re-running the ACE case with the live
`BprimeEvaluator` (carbon-in-air, `gri30`+`graphite`) instead of the table — which
activates `q_adv` — changes the solution by only **+8°R at the 30 s peak and ~1°R
through cooldown** (T_back@49 s 4227→4228°R). The term is genuinely applied
(instrumented: `q_adv≈+0.72 MW/m²` at peak, `surface_enthalpies` returns
`h_c≈6.5`, `h_w≈3.9 MJ/kg`), but the conduction-stiff surface node (`α_F` is large,
so added surface heat conducts straight into the slab) plus the narrow high-`ṁ_c`
window make its net effect a few °R. So #3 is real physics but **not a
contributor to the §8.7 gap** — it does not need the double-counting investigation
to proceed. (Note: the live evaluator requires flooring `p_e` to a small positive
value, e.g. the table's 101.325 Pa floor, because the trajectory pressure starts
near 0 atm and Cantera rejects `p≤0`; `BPrimeTable` clamps to its grid and never
hits this.)

### Root cause #4: the wall-correction asymmetry is a radiation compensator (not a `Ω_hw` error)

Investigating the "T matches CMA but recession is ~18% low" decoupling led, after
ruling out a `Ω_hw` magnitude error, back to #2 (radiation). The chain below shows
recession depends on `Ω_hw · B'_c`; the asymmetric `apply_wall_correction_to_heat`
flag is what masks the radiation over-prediction at the peak.
Instrumenting the mass-transfer chain at the recession peak (~31 s) gives
`ṁ_c = B'_c · ρeueCH · Ω_hw · Ω_blw` with `B'_c=0.207` (ACE table, correct for
3690 K), `ρeueCH=2.05` (Fig 8.15, trusted), `Ω_blw=0.94` (minor) and
**`Ω_hw=0.74`**. The entire recession deficit is that one factor.

**`Ω_hw` is faithfully implemented and correct — do not tune it.** The code
reproduces Cohen–Reshotko (Eq 37, laminar, wall states) and the Eckert
reference-enthalpy method (Eqs 38–42, turbulent) exactly. Decomposing the peak
turbulent factor (`T_w=3690 K`): `Ω_hw = (ρ*_hot/ρ*_cold)^0.8 (μ*_hot/μ*_cold)^0.2
= 0.66^0.8 · 1.26^0.2 = 0.753`. Computing it with four air models — `gri30`,
`gri30_highT`, `air.yaml`, `airNASA9.yaml` — gives **identical `Ω_hw=0.753` to 4
digits** (the hot reference enthalpy maps to only ~3681 K, mildly dissociated, so
GRI-Mech is *not* extrapolating). The cold-wall reference `T_cold=297 K` is the
method-defined flowfield-BC wall, not a free knob. So `Ω_hw=0.75` is genuinely the
Eckert answer. (An earlier draft proposed raising `T_cold` to ~3000 K to soften
`Ω_hw`; that is **disproven** — it is unphysical tuning, not what the methods give.)

**The `apply_wall_correction_to_heat=False` band-aid is a compensator for #2
(radiation), not a wall-correction fix.** Applying the faithful `Ω_hw` to *both*
heat and mass (Amar Eqs 29–30, `apply_wall_correction_to_heat=True`) makes the peak
*too cold*: ACE 6292°R, CNO 6220°R (ref 6541). Amar's own code, using the same
methods, gets 6541 — so SCAM loses ~300°R at peak versus Amar *even with identical
hot-wall physics*. That deficit is the same direction and origin as the cooldown
collapse. Disabling `Ω_hw` on heat (current script) adds heat back and *accidentally
masks* the over-radiation at the peak (ACE 6648 ≈ 6541), but nothing compensates
once aeroheating →0, so cooldown collapses. Decisive confirmation: with the
faithful both-sides correction **plus** reduced radiation as a reservoir proxy
(ACE, ε=0.55), the peak recovers to **6511°R** — proving the peak deficit under the
correct correction is radiative — yet Tw@49 s is still 4113°R (ref 4970), because a
*uniform* ε cut cannot fix the time-dependent radiation error.

**Recession is then set by the B′ table, not `Ω_hw`.** With faithful `Ω_hw=0.75`,
matching ṡ≈8.9×10⁻³ needs `B'_c≈0.25`; ACE gives only 0.207 (→ ṡ 6.9–7.3) while
CNO (sublimation) gives ~0.25. This *inverts* the earlier
"use ACE, CNO over-predicts" conclusion: that preference was compensating for the
mass-only `Ω_hw` asymmetry. Which table is right cannot be settled until the
radiation closure (#2) is fixed and `Ω_hw` is applied consistently.

**Measured 2026-09-07 (correction).** An earlier draft of this paragraph
projected CNO would give ṡ ≈ 8.5–8.6×10⁻³. Actually running it gives **CNO
7.41×10⁻³ and ACE 6.93×10⁻³** against the 9.251×10⁻³ target. So the ordering is
as predicted — CNO recesses more than ACE, inverting the old preference — but
**neither table closes the gap**: CNO is still 20% short. The recession deficit
is therefore not explained by the ACE↔CNO choice alone, and is the most likely
place a genuine SCAM or B′-data issue remains in this case.

**Net:** leave `Ω_hw` faithful (both sides, `T_cold=297 K`). The master variable is
the radiation closure (#2); the back-peak 4–5 s lead is a separate diffusivity
signature unaffected by any of the surface terms.

### B′ database (CNO vs ACE) — still valid

ACE (used by Amar/CMA/SODDIT, no C2/C3) keeps B′_c near the diffusion plateau
(~0.175) to ≈3600 K; Cantera/JANAF with C2/C3 predicts sublimation onset at
~3200 K and gives B′_c ~0.25 at the operating T_wall. The earlier pressure-clamping
fix (tables now span 0.001–5 atm vs the original 1 atm cap) is retained.
`debug_bprime.py` back-calculates the implied B′_c(T_wall) from the reference
curves and demonstrates the sublimation-onset difference. **Which table is correct
is now reopened** (see #4): with the faithful `Ω_hw=0.75`, matching the reference
recession needs `B'_c≈0.25` (CNO), not the ACE plateau — the earlier "use ACE"
preference was compensating for the mass-only `Ω_hw` asymmetry. Defer the table
decision until the radiation closure (#2) is fixed and `Ω_hw` is applied to both.

### Status: §8.7 cooldown is a reference anomaly; §8.8 is the better target

The §8.8 cross-check (above) showed the documented cold reservoir (`T_res=414°R`)
produces physical cooling and that **SCAM already reproduces §8.8** — so SCAM's
radiation/conduction/cooldown is validated and the §8.7 flat cooldown is a
reference-data anomaly (it violates energy conservation by ~7× regardless of any
SCAM choice). Ordered next steps:

1. **Adopt §8.8 carbon-phenolic as the primary cooldown verification** (SCAM
   matches it). **Done** — Figs 8.28–8.32 are digitized in `digitized_data/` and
   SCAM matches surface, back face, recession and the in-depth profiles; see the
   §8.8 section. Prefer polishing that comparison over chasing the §8.7 tail.
2. For §8.7, **restore the faithful hot-wall correction**: `Ω_hw` on both heat and
   mass (`apply_wall_correction_to_heat=True`, `T_cold=297 K`) — do *not* tune
   `T_cold` or the air-property table (proven `Ω_hw=0.75` is correct across four air
   models). Report §8.7 on the *peak* (T and recession), which is physically
   meaningful, and flag the cooldown as an open reference question.
3. With faithful `Ω_hw`, **re-decide the B′ table** (ACE vs CNO): matching the §8.7
   recession needs `B'_c≈0.25` (CNO), so the earlier ACE preference may invert.
4. **Drop** density (#1), the advective term (#3, tested negligible), the radiation
   reservoir `T_res(t)` reconstruction (dead end — §8.8 proves cold reservoir), and
   the `Ω_hw`-tuning idea from the suspect list.
5. The ~4–5 s early §8.7 back-peak is a separate diffusivity signature
   (radiation/chemistry-independent); revisit `k`/`ρCp` only if §8.7 is pursued.

Summary of what was ruled out vs confirmed: density (refuted), advective `q_adv`
(negligible, ~1°R), `Ω_hw` magnitude/air-properties (faithful and correct),
constant or hot `T_res` (dead end — §8.8 cools normally with the cold documented
value), GRI-Mech high-T air (matches `airNASA9` to 4 digits). The §8.7 reference
cooldown is internally inconsistent with `εσ(0.8)T⁴`; §8.8 is the sound target.

---

## Amar (2006) §8.8 — Carbon-Phenolic Thermochemical Ablation

**Script:** `examples/verification/amar_thesis/compare_amar_cp.py`
**Material:** `carbon_phenolic.yaml` (Appendix D — decomposing, Darcy porous flow)
**BC:** Same ballistic reentry trajectory as §8.7 (Figs 8.14–8.15), far-field
radiation source `T_res = 414°R` (§8.8.1, ≈ deep space), back face adiabatic.
**Corrections:** Same faithful Amar et al. (2008) hot-wall (`Ω_hw`, Cohen–Reshotko
/ Eckert) + Kays blowing as §8.7, applied to **both** heat and mass (`Ω_hw` and the
wall-correction machinery are imported from `compare_amar_cc.py`).
**Reference:** research code + CMA, Figs 8.28 (T_surf), 8.29 (T_back), 8.30
(recession, inches), 8.31 (recession rate, in/s) — digitized in `digitized_data/`.

### Results — good agreement across all four histories

| Quantity | SCAM | Reference | Note |
|---|---|---|---|
| T_surf peak | 6206°R @29 s | ~6137°R @~30 s | +69°R |
| Total recession (50 s) | 0.141 in | ~0.137 in | +3% |
| ṡ peak | 12.3×10⁻³ in/s | ~13×10⁻³ in/s | −5% |
| T_back (50 s) | 713°R | ~715°R | −2°R (**closed**, see below) |

Unlike §8.7, **SCAM matches §8.8 across the full transient including the cooldown**
(surface peaks ~6200°R and cools normally to ~2000°R). The faithful hot-wall
correction is essential to the recession: without it the total recession was
0.209 in (+53%); with `Ω_hw` on both heat and mass it drops to ~0.14 in, matching
the reference. This corroborates the §8.7 finding — the faithful `Ω_hw` is correct,
and the cold far-field reservoir produces physical cooling. The carbon-phenolic
case is the **sound code-to-code verification** of SCAM's full decomposing-ablator
surface closure (pyrolysis gas, B′ chemistry, hot-wall + blowing corrections,
recession). Fig 8.32 in-depth profiles are digitized (9 snapshot times, in
`digitized_data/figure8.32_*sec.csv`) and plotted in the bottom panel of
`compare_amar_cp.png`; overall RMS 88.9°R.

### Back-face discrepancy — RESOLVED 2026-09-07 (erroneous cp column in Amar Table D.4)

The back face used to run **+169°R** hot at t=50 s (883.7 vs 715.0°R, measured
2026-09-07; an earlier revision of this doc recorded it as +158°R / 873°R). Root cause is a
**bad number in Amar's own Appendix D**, not SCAM physics:

- **The gap is not inherited from the surface.** Measured gap-vs-time shows the
  back face tracking the reference to ≤3°R through t≈36 s regardless of the
  surface, then opening monotonically to +169°R by t=50 s — *while the surface
  ran 50–105°R **colder** than the reference over that same window*. An excess
  of absorbed surface energy cannot produce an anti-correlated signature like
  that; the mechanism had to be in-depth.
- **It is generated in the unreacted virgin zone.** At t=36–50 s the pyrolysis
  front sits at 4.1→4.6 mm and barely creeps, while the back face is at 9–13 mm
  depth — 5–9 mm of material that is still 100% virgin, where no decomposition
  sink is active. So the gap is straight conduction, not pyrolysis energy.
- **`k_virgin` and `cp_virgin` are exact.** Both reproduce Amar Table D.3 to full
  printed precision under the Btu/(ft·s·°R) conversion; cleared as causes.
- **`cp_char` is the culprit.** The card reproduced Table D.4 exactly (all 16
  values), but D.4's cp column is itself wrong: 0.05 Btu/lbm·°R at 500°R
  = 209 J/kg·K at 278 K, physically impossible for a carbonized char (real
  graphite is ~700 J/kg·K at 300 K; carbon does not reach 209 J/kg·K until
  ~100–130 K). The column runs 4.8× low at 500°R, converging to 0.75× by
  6000°R, making char thermal diffusivity 1.47× too high (8.11e-7 vs 5.53e-7
  m²/s at 3000°R) — the char fails to buffer the pulse and passes it inward.

**Fix:** `cp_char` (and the `h_char` integral) in `carbon_phenolic.yaml` now come
from Kenneth Sutton, *An Experimental Study of a Carbon-Phenolic Ablation
Material*, NASA TN D-5930 (1970), Table VI(b) — an independent measurement of the
same material class, and very likely Appendix D's own ultimate source (Sutton's
virgin cp of 0.238 Btu/lbm·°R at 460°R matches Amar's D.3 virgin cp of 0.24).

This is a **clean single-variable correction**: because `h_f_char = 0`,
`h_bar_chemical = ρ_v·h_f_v/(ρ_v−ρ_c) = −4.264 MJ/kg` is independent of `cp_char`
(verified identical to 3e-13 relative), so the decomposition energy source is
untouched. Only `cp_char` was superseded — `k_char` (agrees with Sutton to ~1% at
2000°R) and char emissivity 0.85 are retained from Amar.

| t [s] | T_back ref | with D.4 cp | with Sutton cp |
|---|---|---|---|
| 36 | 550.7 | 547.9 (−2.7) | 545.2 (−5.4) |
| 40 | 569.7 | 586.0 (+16.3) | 561.7 (−8.1) |
| 44 | 612.9 | 692.8 (+80.0) | 603.4 (−9.5) |
| 48 | 679.9 | 820.0 (+140.1) | 674.3 (−5.6) |
| 50 | 715.0 | 883.7 (**+168.8**) | 713.4 (**−1.5**) |

The back face now tracks Fig 8.29 within ±10°R across the whole transient, with
peak surface T (6207→6206°R) and recession (0.1419→0.1413 in) essentially
unchanged — so nothing else in the closure was perturbed to achieve it.

**Independent corroboration — in-depth profiles (Fig 8.32).** The digitized
Fig 8.32 profiles are a separate dataset from the Fig 8.29 back-face history, and
they confirm the same conclusion. Error of SCAM against all 9 digitized snapshot
profiles, over the depth range the two share:

| cp_char source | RMS | max abs err | mean bias |
|---|---|---|---|
| Amar Table D.4 | 263.5°R | 711.6°R | **+161.6°R** |
| Sutton TN D-5930 | **88.9°R** | 328.8°R | −39.6°R |

The diagnostic detail matters more than the 3× overall improvement: the t=10 s
and t=20 s profiles are **identical** between the two cases (22.1 and ~22°R RMS),
because almost no char exists yet. The D.4 error appears only from t=25 s onward
and saturates at a ~+260°R hot bias — exactly the signature of a char layer whose
volumetric heat capacity is too low. A conduction or surface-closure error would
have biased the early profiles too.

**Fig 8.32 datum.** The digitized curves are depth from the ORIGINAL front face,
not the receded surface — each starts at that snapshot's recession depth (the
t=50 s curve starts at 0.139 in against `s_total` = 0.141 in). `MeshState.y_nodes`
already uses that datum (`y_nodes[0] == s_total`), so the compare script plots
`y_nodes` directly. Subtracting `y_nodes[0]` shifts SCAM left by the recession and
manufactures a mismatch that grows with time.

**Amar Appendix D carries at least three documentation errors.** Besides the cp
column above, the conductivity headers of *both* D.3 and D.4 print
"Btu/ft²-s" — an ft² denominator with no °R at all — while Table 8.12 (C-C) in
the same thesis correctly prints Btu/(ft·s·°R). Fourier's law fixes k as
energy/(length·time·temperature); an ft² denominator is the dimension of a film
coefficient, which cannot be a bulk material property. Sutton's independently
labelled Btu/(ft·sec·°R) virgin k lands within ~6% of Amar's D.3 value at the one
overlapping low-T point (9.0e-5 vs 9.5e-5 at 460°R) — a hidden length-scale
factor would have shown as a large constant offset. **Treat Appendix D values as
suspect until cross-checked against Sutton TN D-5930.**

**Remaining open item:** the cooldown *surface* temperature runs ~50–105°R below
the Fig 8.28 reference (it did before this fix too, and the fix moves it modestly
further negative — physically consistent, since a lower-diffusivity char holds the
pulse near the surface where it reradiates instead of conducting inward). That is
a separate surface-closure question, not a conduction one.

---

## Current Verification Status (as of 2026-09-07)

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

### Amar (2006) §8.7 — Carbon-Carbon — Peak partly matched; cooldown is a reference anomaly

The only case still not matching. Three of the four root causes are closed by
direct test (density refuted; surface advective enthalpy measured at +8 °R;
`Ω_hw` proven faithful at 0.753 across four air models). The cooldown collapse is
attributed to an **energy-conservation contradiction in the reference itself**
(it radiates ~7× less than εσT⁴ demands), an attribution now strongly supported
because SCAM reproduces §8.8 to −2 °R at the back face. Genuinely open: the
back-face peak arrives ~4 s early (`k` is the one diffusivity input never swept),
and peak recession is 20–25% low with neither B′ table closing it. See the §8.7
section above.

### Amar (2006) §8.8 — Carbon-Phenolic — Verified

Good agreement across surface T, back face, recession, and cooldown. The former
~169°R back-face overshoot was **resolved 2026-09-07**: Amar's Table D.4 char
specific-heat column is erroneous (209 J/kg·K at 278 K, physically impossible for
a char) and is now superseded by Sutton NASA TN D-5930 Table VI(b); back face went
from +169°R to −1.5°R at t=50 s with surface T and recession unchanged. See the
§8.8 section above, corroborated independently by the Fig 8.32 in-depth profiles
(RMS 263→89°R). Open: cooldown surface T runs 50–105°R low.

### CHyPS BlaineTest — Verified (2026-09-07)

Agreement is excellent throughout: surface within 3.2 K during the high-flux
phase, recession within 0.45%, and deep probes T6/T7 within ~6 K at t = 60–90 s.
The previously recorded ~126 K deep-probe deficit is **gone**, and the root cause
recorded for it ("CHyPS omits composition-dependent solid enthalpy") is
**disproven** — nulling `h_bar_chemical` now makes SCAM 220–275 K *too hot* at
depth, so the sink is required for the match. See §CHyPS section above.

### Bianchi thesis — Setup pending

Some reference figures digitized (`examples/verification/bianchi_thesis/`).
Simulation cases not yet configured.

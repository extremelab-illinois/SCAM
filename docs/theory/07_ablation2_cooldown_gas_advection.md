# Ablation2 cooldown surface‑temperature gap — pyrolysis‑gas advection at the surface node

**Status:** root cause identified and verified (fresh analysis, June 2026).
**Case:** `examples/verification/ablation2/compare_pato_ablation2.py`, chemistry mode
`live_cantera_zc_default`, vs PATO `AblationTestCase_2.x`.

This document is self‑contained and does not rely on the other `pato_validation.md`
notes; it records a distinct mechanism found by a from‑scratch re‑derivation of the
surface and in‑depth energy balances.

## Symptom

The surface temperature matches PATO well during the heat pulse (0–60 s; PATO ≈ 4–5 K
hotter at t = 60 s, within sampling noise). After flux‑off (t > 60 s) SCAM cools
**faster** than PATO and the gap grows to ≈ −55 K by t ≈ 70 s before slowly closing:

| t (s) | SCAM T_w | PATO T_w | Δ (SCAM−PATO) |
|------:|---------:|---------:|--------------:|
| 60.0  | 1563.9   | 1568.5   | −4.6  |
| 61.1  | 1054.4   | 1070.0   | −15.6 |
| 63.2  | 880.6    | 911.0    | −30   |
| 65.2  | 799.2    | 838.7    | −39   |
| 67.2  | 742.2    | 790.8    | −49   |
| 70.2  | 686.2    | 741.2    | −55   |

## What is *not* the cause

1. **Surface BC.** During cooldown PATO's `Bprime` temperature BC
   (`BprimeBoundaryConditions.C::updateTemperatureBC`, `chemistryOn==0` branch) reduces
   to **pure radiation**: `hconv=0` ⟹ `qConv=0`, `qAdvPyro=qAdvChar=0`, `qRad=0`, so
   `qCond = qRadEmission = −εσ(T⁴ − T_bg⁴)`. SCAM's `seb_residual` cooldown path is
   identical: `alpha_conv=0` ⟹ `q_conv=0`; `q_adv` and `q_mass_removal` are gated to 0
   by `_bprime_ran=False`; only `q_rad_in − q_rad_out` remains, i.e.
   `q_cond = εσ(T_bg⁴ − T_w⁴)`. Both use ε = 0.9 (fully charred surface; PATO virgin
   0.8 / char 0.9, SCAM blends to the same), T_bg = 300 K. Confirmed numerically at
   t ≈ 61 s: SCAM `q_rad_out ≈ 62.9 ≈ −q_cond ≈ 62.5 kW/m²`.

2. **Heating‑phase reservoir.** The near‑surface temperature **profiles at t = 60 s are
   essentially identical** (both ≈ 1327 K at 0.95 mm below the surface, ≈ 1205 K at
   1.5 mm). SCAM did not store less energy during heating.

So at flux‑off the codes have the same surface BC and the same subsurface reservoir,
yet SCAM cools faster from the very first instant. The divergence is therefore in an
**in‑depth energy term that is active during cooldown**, localised at the surface node.

## Root cause

The 1‑D FVM energy assembly advects pyrolysis‑gas enthalpy node‑to‑node
(`scam/numerics/assembly.py`, the `m_dot_g` block). At the surface node it deliberately
**zeros the outflow**:

```python
mg_out = m_dot_g.copy()
mg_out[0] = 0.0                      # "surface exit handled by the SEB via h_wall"
mg_in[:-1] = m_dot_g[1:]
Q_adv = mg_in * h_in - mg_out * h_g_cache    # node 0:  m_dot_g[1]·h_g(T_1) − 0
```

So the surface node **receives** the gas advective enthalpy `m_dot_g[1]·h_g(T_1)` from
below but **emits nothing**, on the assumption the surface energy balance carries the
gas back out at the equilibrium wall enthalpy `h_wall`.

* During **heating** the assumption holds: the CMA term
  `q_conv = ρUeCH_eff·(h_r − h_wall)` removes the gas at `h_wall`, so the net surface
  gas energy is the (physical) transpiration term `m_dot_g·(h_g(T_1) − h_wall)`.
* During **cooldown** the SEB's gas bookkeeping is switched off
  (`_bprime_ran=False` ⟹ `q_adv=0`, `q_mass_removal=0`). The deposit
  `m_dot_g·h_g(T_1)` is then **uncompensated**. Because the pyrolysis‑gas enthalpy is
  strongly **negative** at the cooling near‑surface temperatures
  (`h_g(300 K) ≈ −7 MJ/kg`, crossing zero near ≈ 1340 K), the deposit becomes a
  **spurious cooling sink** at the surface node.

PATO has no such sink: its `Pyrolysis` energy model
(`PyrolysisEnergyModel.C`, term `−∇·(GammaHg·∇p)`) advects the gas enthalpy with the
Darcy flux **conservatively out** through the surface face, so the surface cell's net
advective effect is only the small transpiration term `m_dot_g·(h_g(T_1) − h_g(T_0))`.

### Equivalent statement

The surface‑node gas advection should be conservative (in − out):

```
PATO / conservative :  m_dot_g·h_g(T_1) − m_dot_g·h_g(T_0)
SCAM (current)      :  m_dot_g·h_g(T_1) − 0
SCAM − conservative :  + m_dot_g·h_g(T_w)        (h_g(T_w) < 0 ⟹ a deficit ⟹ colder)
```

SCAM is missing the surface gas **outflow** `m_dot_g·h_g(T_w)`. When the B′ chemistry
SEB is active it supplies this via `h_wall`; when it is not (cooldown / chemistry off)
nothing does.

## Quantitative verification

Instrumenting `Q_adv[0] = m_dot_g[1]·h_g(T_1)` over the run:

| phase    | t (s) | Q_adv[0]        | surface radiative loss | SCAM−PATO T_w |
|----------|------:|----------------:|-----------------------:|--------------:|
| heating  | 55–60 | **+20 kW/m²** (compensated by `h_wall`) | — | ≈ 0 |
| cooldown | 62.8  | −13.4 kW/m²     | ~34 kW/m²              | −25 K |
| cooldown | 65.2  | **−14.1 kW/m²** | ~21 kW/m²              | −39 K |
| cooldown | 70.2  | −9.4 kW/m²      | ~11 kW/m²              | −55 K |

At flux‑off `h_g(T_1)` crosses zero (≈ t = 60.2 s) and `Q_adv[0]` flips from a
+20 kW/m² source to a sink that peaks near −14 kW/m² (≈ t = 65 s) — **comparable to the
entire radiative loss** — then decays as `m_dot_g` falls. This magnitude and timing
account for the full −40…−55 K cooldown deficit, the early‑cooldown peak, and the slow
late recovery; it leaves the matched heating phase and the matched t = 60 s reservoir
untouched, exactly as observed.

## Fix direction

Restore conservation of the surface gas exit **only when the B′ chemistry SEB is not
already carrying it** (`_bprime_ran=False`): remove the pyrolysis gas at the
**pyrolysis‑gas enthalpy** `h_g(T_w)` (not the B′ `h_wall`, whose reference differs —
this is why the `_bprime_ran` gate exists). The net surface gas energy then becomes the
PATO‑consistent transpiration term `m_dot_g·(h_g(T_1) − h_g(T_w))`. The heating path
(`_bprime_ran=True`) is unchanged.

Implemented in `scam/physics/surface_energy.py::seb_residual` (one block gated on
`not _bprime_ran and m_dot_pyro > 0`, adding `m_dot_pyro·h_g(T_w)` to
`q_mass_removal`); the `assembly.py` surface‑node comment cross‑references it. See the
conservation‑closure and secondary‑term diagnostics in
`examples/verification/ablation2/conservation_check_cooldown.py`.

## Verification of the fix

Diagnostic (`conservation_check_cooldown.py`):

* **#1 closure** — the identity `Q_adv0 − Q_adv0_cons − imbalance = 0` holds to machine
  precision; the uncompensated outflow `m_dot_g·h_g(T_w)` ranges −14.9 … −1.7 kW/m²
  over cooldown (peak −14.9 kW/m² ≈ 54–69 % of the radiative loss), and the
  *conservative* net `Q_adv0_cons` is ≤ 0.82 kW/m² (true transpiration is negligible).
* **#2 secondary terms** — surface‑node gas‑storage ≤ 0.054 kW/m² and Darcy expansion
  ≤ 0.036 kW/m², i.e. ~0.4 % and ~0.2 % of the advective imbalance. The advection
  term is the sole driver.

Before → after the fix (max |SCAM − PATO|):

| case | heating | cooling |
|---|---|---|
| base `2.x` (live Cantera) | 34.2 → **34.2 K** (unchanged) | 56.0 → **5.6 K** |
| `2.x_chemistryOff`     | 21.5 → **10.8 K** | 75.4 → **9.9 K** |
| `2.x_multiMat`         | unchanged | unchanged (no `_bprime_ran=False` surface regime) |

The base‑case heating phase is unchanged (the fix is gated to `_bprime_ran=False`).
`chemistryOff` — which runs with `_bprime_ran=False` throughout — improves in *both*
phases, confirming the same mechanism. All 91 unit/verification/integration/validation
tests pass.

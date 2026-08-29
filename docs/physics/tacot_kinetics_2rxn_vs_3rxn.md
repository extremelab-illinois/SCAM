# TACOT 3.0 Kinetics: 3-Reaction vs 2-Component Mapping

## PATO Phase Model

PATO models TACOT as two solid phases inside a porous skeleton:

| Phase | Description | Intrinsic density | Volume fraction | Apparent density |
|---|---|---|---|---|
| 1 | Carbon fiber | 1600 kg/m³ | ε₁ = 0.10 | **160 kg/m³** |
| 2 | Phenolic matrix | 1200 kg/m³ | ε₂ = 0.10 | **120 kg/m³** |

Phase 1 has `nPyroReac[1]=0` — it is inert throughout.  
Phase 2 has `nPyroReac[2]=3` — three Arrhenius reactions consuming fractions F[2][j] of the matrix:

| PATO reaction | F[2][j] | Apparent rho_0 [kg/m³] | A [1/s] | E [J/mol] | m | Gas products |
|---|---|---|---|---|---|---|
| j=1 (resin_A) | 0.25 | 0.25 × 120 = **30.0** | 1.2×10⁴ | 71 130.89 | 3 | H₂O, C₆H₆, C₆H₅OH |
| j=2 (resin_B1) | 0.19 | 0.19 × 120 = **22.8** | 4.97777×10⁸ | 169 975 | 3 | CO₂, CO, CH₄ |
| j=3 (resin_B2) | 0.06 | 0.06 × 120 = **7.2** | 4.97777×10⁸ | 169 975 | 3 | H₂ |

Reactive fraction of phase 2: F_total = 0.25 + 0.19 + 0.06 = **0.50** → 60 kg/m³ pyrolyzes.  
Non-reactive residual: (1 − 0.50) × 120 = **60 kg/m³** stays as char.

### Virgin and char densities

```
rho_virgin = rho_A + rho_B1 + rho_B2 + rho_inert + rho_fiber
           = 30.0  + 22.8   + 7.2    + 60.0       + 160.0
           = 280.0 kg/m³

rho_char   = rho_inert + rho_fiber
           = 60.0      + 160.0
           = 220.0 kg/m³
```

The 60 kg/m³ inert matrix residual is **not a pyrolysis product** — it never reacts. It is already included in `rho_char` from the beginning and has no Arrhenius component in SCAM.

---

## SCAM Rate Law

SCAM's Arrhenius step (`physics/decomposition.py`) uses:

```
dρ_i/dt = −A_i · ρ_{0,i} · ((ρ_i − ρ_{r,i}) / ρ_{0,i})^{m_i}
         = −A_i · (ρ_i − ρ_{r,i})^{m_i} / ρ_{0,i}^{m_i − 1}
```

where `ρ_i` is the current apparent density of component i, `ρ_{0,i}` is its initial apparent density (at t=0, fully virgin), and `ρ_{r,i}` is its residual (char) density after complete reaction.

For components that fully pyrolyze, `ρ_r = 0`, and the equation reduces to:

```
dρ_i/dt = −A_i · ρ_i^{m_i} / ρ_{0,i}^{m_i − 1}
```

---

## Merging Reactions 2 and 3 into One Component

Reactions j=2 and j=3 have **identical** A, E, and m. Their individual rates are:

```
dρ_{B1}/dt = −A_B · ρ_{B1}^3 / 22.8²
dρ_{B2}/dt = −A_B · ρ_{B2}^3 /  7.2²
```

Define `ρ_B = ρ_{B1} + ρ_{B2}`, the total density of both components. Because both start at their respective `ρ_0` values and share the same E (same temperature dependence), the normalized progress variable `ξ = ρ_i / ρ_{0,i}` evolves identically for both at any given temperature. Therefore:

```
ξ_{B1}(t) = ξ_{B2}(t) = ξ_B(t)   at all times
```

and:

```
ρ_B(t) = ρ_{B1}(t) + ρ_{B2}(t)
        = ξ_B(t) · 22.8 + ξ_B(t) · 7.2
        = ξ_B(t) · 30.0
```

So the merged component with `ρ_{0,B} = 30.0`, `ρ_{r,B} = 0` satisfies:

```
dρ_B/dt = −A_B · ρ_B^3 / 30.0²
```

which equals the sum of the two individual rates:

```
dρ_{B1}/dt + dρ_{B2}/dt = −A_B · (ξ_B · 22.8)³/22.8² − A_B · (ξ_B · 7.2)³/7.2²
                        = −A_B · ξ_B³ · (22.8 + 7.2)
                        = −A_B · ξ_B³ · 30.0
                        = −A_B · (ρ_B/30.0)³ · 30.0
                        = −A_B · ρ_B³ / 30.0²  ✓
```

**The correct 2-component merge is `ρ_0 = 30.0`, `ρ_r = 0`.**

---

## Common Error: Including the Inert Residual in ρ_r

A tempting but wrong interpretation is to write resin_B with `ρ_0 = 90`, `ρ_r = 60`, reasoning that the total phase 2 mass is 120 kg/m³, of which 60 kg/m³ remains as char.

This is incorrect because:

1. The 60 kg/m³ inert residual **never participates in any reaction**. PATO does not model it as partially pyrolyzing to a residue — it simply does not assign it to any Arrhenius reaction.

2. With `ρ_0 = 90`, `ρ_r = 60`, SCAM computes the initial rate as:

   ```
   dρ/dt|_{t=0} = −A · (90 − 60)³ / 90² = −A · 30³ / 8100 = −A · 3.33
   ```

   With the correct `ρ_0 = 30`, `ρ_r = 0`:

   ```
   dρ/dt|_{t=0} = −A · 30³ / 30² = −A · 30
   ```

   The error is a **factor of 9×** in the initial rate (and the discrepancy persists throughout pyrolysis because the normalized progress variable `(ρ − ρ_r) / ρ_0` evolves differently).

3. In the ablation1 test case this produces a ~40 K systematic under-prediction of deep-material temperatures (the decomposition front propagates too slowly, releasing less heat).

---

## SCAM Component Tables

### 3-reaction form (`tacot_v3.0_3rxn.yaml`)

| Component | ρ_0 [kg/m³] | ρ_r [kg/m³] | A [1/s] | E [J/mol] | m |
|---|---|---|---|---|---|
| resin_A  | 30.0 | 0 | 1.2×10⁴ | 71 130.89 | 3 |
| resin_B1 | 22.8 | 0 | 4.97777×10⁸ | 169 975 | 3 |
| resin_B2 |  7.2 | 0 | 4.97777×10⁸ | 169 975 | 3 |

### 2-component form (`tacot_v3.0.yaml`)

| Component | ρ_0 [kg/m³] | ρ_r [kg/m³] | A [1/s] | E [J/mol] | m |
|---|---|---|---|---|---|
| resin_A | 30.0 | 0 | 1.2×10⁴ | 71 130.89 | 3 |
| resin_B | 30.0 | 0 | 4.97777×10⁸ | 169 975 | 3 |

Both forms are kinetically equivalent (same density trajectories at every temperature).

---

## On h_decomp and the −4 MJ/kg Pyrolysis Energy

PATO's `constantProperties` lists `h[2][j] = −4×10⁶ J/kg` for each reaction (a comment notes this corrects an error factor of 2 from `TACOT_3.0.xls` which had −2 MJ/kg).

In PATO's `Pyrolysis` energy model, the in-depth source term is:

```
Q_pyro = −π · (h_s + h_p)
```

where:
- **h_s** = blended sensible enthalpy = `(ρ_v·h_v(T) − ρ_c·h_c(T)) / (ρ_v − ρ_c)` — from the solid enthalpy tables
- **h_p** = per-reaction chemical heat = `Σ_j F[2][j] · h[2][j]` = −4 MJ/kg (for all reactions with same h)

In SCAM, `h_s` is captured implicitly by the exact `d(ρh)/dt` storage (the `ρ_old·h_old` term). The equivalent of `h_p` is `h_decomp` per component.

**For the PATO ablation1/ablation2 test cases**, the tutorials run with `detailedSolidEnthalpies no`, which disables the `h[2][j]` term in PATO. PATO therefore uses only `h_s` (from the enthalpy tables), not `h_p`. The correct SCAM setting is `h_decomp = 0`.

**If running a PATO case with `detailedSolidEnthalpies yes`**, set `h_decomp = -4e6` on each component. This is an endothermic contribution (energy consumed by the reaction), distinct from the enthalpy table contribution which is already handled by SCAM's energy storage.

---

## A_rate Discrepancy: PATO vs xls

The two sources give different values for the high-temperature reaction:

| Source | A [1/s] |
|---|---|
| PATO `constantProperties` (AblationTestCase) | 4.97777×10⁸ |
| `TACOT_3.0.xls` (Pyrolysis model sheet) | 4.48×10⁹ |

The factor ~9 difference is significant. `tacot_v3.0.yaml` (and `tacot_v3.0_3rxn.yaml`) use the PATO value (4.97777×10⁸) to match the PATO reference output. If you are validating against a code that uses the xls value, use 4.48×10⁹ instead.

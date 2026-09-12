# Energy Formulation Comparison: Amar (2006) · PATO · SCAM

This document records how each code formulates the in-depth mixture energy equation, the
decomposition energy term, and the surface energy balance, with the key equations quoted
from each source.  It explains why material properties (particularly `h_decomp` and the
formation-enthalpy fields) cannot be transplanted between codes without conversion.

## References

- **Amar 2006** — Amar, A.J. (2006). *Modeling of One-Dimensional Ablation with Porous
  Flow Using Finite Control Volume Procedure*. PhD thesis, North Carolina State University.
  Governing equations: §2.1 (pp. 12–20); energy tables: Appendix A (pp. 141–144).
- **PATO** — Lachaud, J. & Mansour, N.N. (2014). *Porous-material Analysis Toolbox Based
  on OpenFOAM and Applications*. JTHT 28(2).  Energy model: `Pyrolysis` BC in
  `PATO/src/applications/libraries/fluidThermophysicalModels/`.
- **SCAM** — this codebase. Energy assembly: `scam/numerics/assembly.py`.
  Pyrolysis energy: `scam/physics/charring_energy.py`, `scam/physics/properties.py`.

---

## 1  In-Depth Mixture Energy Equation

### 1.1  Amar (2006) — internal-energy formulation

Amar writes the mixture energy equation in integral (CVFEM) form (Eq. 2.1):

```
∫_cs q̇''·dA  +  ∫_cs φ ρ_g h_g v_g·dA  −  ∫_cs ρ h v_cs·dA  +  d/dt ∫_cv ρe dV  =  0
  conduction        gas-enthalpy flux          grid convection           energy content
```

**The stored-energy variable is `ρe`** — specific internal energy `e`, not enthalpy.
Energy tables for each phase are anchored by absolute formation enthalpy (Appendix A,
Eq. A.5):

```
e_s(T) = h_f^o  +  ∫_{T_ref}^{T}  c_v(T') dT'
```

where `h_f^o` is the standard formation enthalpy of the phase (elements at 298 K as
reference).  When the solid transitions from virgin to char, the change in stored energy
`Δ(ρe)` between the two phases automatically captures the decomposition energy because
each phase has a different `h_f^o` anchor:

```
Δ(ρe)_decomp  =  ρ_v e_v(T)  −  ρ_c e_c(T)
              =  ρ_v [h_f^o_v + ∫c_v,v dT]  −  ρ_c [h_f^o_c + ∫c_v,c dT]
```

The decomposition energy is therefore **implicit** — it is embedded in the energy tables
through the `h_f^o` datum shifts, not carried by a separate scalar parameter.

The pyrolysis gas enthalpy table is also shifted (Eq. A.1):

```
h_g(T_k) = [h_g(T_k)]_input  +  (h_f^o)_g
```

so that the advected gas enthalpy is on the same absolute reference as the solid energy.

### 1.2  PATO — enthalpy formulation with explicit decomposition split

PATO solves the enthalpy-based energy equation (differential form, `Bprime` boundary
condition, `Pyrolysis` energy model):

```
d/dt (ρ_s h_s)  =  ∇·(k ∇T)  −  ṁ_g'' · ∇h_g  −  π · (h_s_bar + h_p)
```

where the decomposition energy is **split into two explicit terms**:

| Symbol | Name | Nature | TACOT value |
|--------|------|---------|-------------|
| `h_s_bar` (`hs`) | T-dependent blended solid enthalpy difference | table | varies with T |
| `h_p` (`h[2][i]`) | per-reaction heat of pyrolysis | scalar | −4 MJ/kg (exothermic sign) |

In PATO's `constantProperties` file:

```
h[2][1] = h[2][2] = h[2][3] = -4e6;   // J/kg, exothermic convention (negative = releases heat)
// Note in TACOT file: "in TACOT: -2e6 J/kg (note 11 Ap. 2015: error factor 2 in TACOT_3.0.xls)"
```

PATO's sign convention: **negative = exothermic**.  A decomposing resin that absorbs heat
uses a positive value; most PATO materials use negative values because the net reaction
(pyrolysis of carbon phenolic) is conventionally treated as exothermic at the material
scale in many legacy datasets.

### 1.3  SCAM — exact `d(ρh)/dt` sensible-enthalpy formulation with Picard iteration

SCAM's cell-centred FVM energy equation (semi-implicit Backward Euler, `assembly.py`).
The stored energy variable is the **sensible enthalpy** `h(T) = ∫₀ᵀ cp(T') dT'`:

```
ρ_new · h(T^{n+1}, ρ_new) − ρ_old · h(T^n, ρ_old)
    = G_{n-½} (T_{n-1} − T_n)  −  G_{n+½} (T_n − T_{n+1})
      + Q_decomp_n · A_n · Δ_n
      + Q_pyro_n   · A_n · Δ_n
```

The old-state enthalpy is evaluated at **pre-decomposition** density `ρ_old`.
This exact FVM storage term carries the sensible energy removed by density loss
implicitly through `(ρ_old − ρ_new) · h_old / dt`.

Linearised for Picard iteration at iterate `T^k`:

```
[ρ_new · cp(T^k) / dt] · T^{n+1}   [diagonal M]
    + conduction terms
  =  [ρ_new · cp(T^k) · T^k + ρ_old · h(T^n,ρ_old) − ρ_new · h(T^k,ρ_new)]
       / dt · A · Δ   [Dc_thermal]
     + Q_decomp + Q_pyro
```

`h_old = h(T^n, ρ_old)` is precomputed once per timestep before decomposition and
held fixed throughout the Picard loop. `h_k = h(T^k, ρ_new)` is recomputed each
iteration. At constant `cp` and constant density this reduces to the usual
Backward-Euler heat-capacity form; with decomposition, the density-change
sensible term remains present and conservative.

`SolverOptions.use_rho_old=False` is retained as a diagnostic that reproduces the
PATO-style storage approximation `ρ·dh/dt`: it passes `h_old=None` and
`T_old_rhs=T^n` to the assembly, giving `Dc_thermal = M·T^n`. This option matches
PATO mean density near the heating cutoff but diverges during cooldown, so SCAM's
production default remains the exact `d(ρh)/dt` form above.

The decomposition energy `Q_decomp` has two **mutually exclusive** paths:

**Path A — enthalpy-table split (when absolute enthalpy tables exist):**

```
Q_decomp_n  =  −(dρ/dt)_n · h_bar_chemical(T_n)

h_bar(T)  =  [ ρ_v · h_v(T)  −  ρ_c · h_c(T) ]  /  (ρ_v − ρ_c)
h_bar_chemical(T) = h_bar_absolute(T) − h_bar_sensible(T)
```

Active when `h_virgin_table` and `h_char_table` are present. The sensible piece
is already carried by exact storage, so `Q_decomp` carries only the chemical
remainder. Do not add `h_decomp` on top.

**Path B — `h_decomp` (scalar fallback):**

```
Q_decomp_n  =  Σ_i  (dρ_i/dt)_n · h_decomp_i       [J/m³/s]
```

`h_decomp_i > 0` = endothermic (absorbs energy).  Used by materials without
explicit enthalpy tables.

Using both paths on the same material double-counts decomposition energy.

---

## 2  Decomposition Energy: Sign and Magnitude Conventions

| Code | Parameter | Sign convention | TACOT example |
|------|-----------|----------------|---------------|
| **Amar** | `h_f^o` datum in energy table | absolute (elements at 298 K) | `h_f_virgin = −844,338 J/kg` |
| **PATO** | `h[2][i]` scalar per reaction | negative = exothermic | `−4×10⁶ J/kg` |
| **SCAM** | `h_decomp` scalar per component | positive = endothermic | material-specific; TACOT 3.0 sets `h_decomp=0` because PATO disables `hp` |

For path-B materials, `h_decomp` represents the pure chemical pyrolysis heat
(`hp`) while the sensible `hs` piece is supplied implicitly by exact storage.
For materials that use path A, `h_decomp` is ignored.

### 2.1  Relationship between Amar's `h_f^o` and SCAM's `h_decomp`

Amar's `h_f_virgin`, `h_f_char`, and `h_f_gas` are **absolute formation enthalpies** used
as datum offsets in the internal-energy tables.  They are **not** directly usable as
SCAM's `h_decomp`.  The conversion follows Hess's law:

```
virgin  →  char  +  gas

h_decomp  =  h_f_gas(T_pyrolysis)  +  h_f_char  −  h_f_virgin        [J/kg of resin]
```

For Amar's carbon-phenolic (Appendix D):

```
h_f_virgin = −844,338 J/kg   (from Don Potter / Sandia communication, via Amar Table D.2)
h_f_char   =       0 J/kg    (carbon residue referenced to graphite)
h_f_gas    =  computed from Cantera at T_pyrolysis ≈ 700 K

→  h_decomp  =  h_f_gas(700 K)  +  844,338   [J/kg of decomposed resin]
```

`h_f_gas` must be obtained from equilibrium chemistry (e.g., Cantera) at a
characteristic pyrolysis temperature.  It cannot be read directly from Appendix D because
Amar's `h_f_gas = 0` in the YAML means "referenced to elements at 298 K" within his
internal-energy framework — not that the gas has zero formation enthalpy.

### 2.2  T-dependence of `h_decomp`

`h_decomp` is thermodynamically temperature-dependent: the true decomposition enthalpy
is `ΔH_rxn(T) = h_f_gas(T) + h_f_char(T) − h_f_virgin(T)`.  SCAM stores a single
scalar, which is appropriate for the narrow pyrolysis temperature range (typically
600–900 K for phenolic resins).  Evaluating `h_f_gas` via Cantera at ~700 K gives a
representative value.  For wider temperature ranges, path A (enthalpy tables) is
thermodynamically preferable.

---

## 3  Surface Energy Balance

### 3.1  Amar (2006) — Eq. 2.11

```
q̇''  =  ρ_e u_e C_{h_o} (C_h/C_{h_o}) (h_w − h_r)
         + εσ (T_bnd⁴ − T_res⁴)
         + ṁ_s'' h_w
         + ṁ_g'' h_w
```

Terms: aerodynamic heating (with blowing/hot-wall Stanton correction), radiation, char
ablation enthalpy, pyrolysis gas enthalpy.  The wall enthalpy `h_w` is from equilibrium
B′ chemistry.  Note that both `ṁ_s''` and `ṁ_g''` use the **same** wall enthalpy `h_w`
in Amar's formulation — the advective terms are not split into separate `h_g` and `h_c`
references.

### 3.2  PATO (`Bprime` BC, 2.x cases)

```
q̇''  =  ρ_e u_e C_H (h_r − h_w)              [convection, blowing-corrected]
         − εσ T_w⁴                              [radiation out]
         + ṁ_char'' h_wall                      [char oxidation — from B′ table]
         + ṁ_g'' (h_g − h_w)                   [pyrolysis gas advection]   (qAdvPyro)
         + ṁ_char'' (h_c − h_w)                [char surface enthalpy]     (qAdvChar)
```

The advective terms `qAdvPyro` and `qAdvChar` use separate `h_g` and `h_c` references
(from live equilibrium chemistry), distinct from the wall equilibrium enthalpy `h_wall`.

### 3.3  SCAM (`seb_residual`, `physics/surface_energy.py`)

```
F(T_w)  =  α_conv (T_aw − T_w)                [convective, blowing-corrected]
            − εσ T_w⁴                           [radiation out]
            + ṁ_char'' h_wall                   [char ablation, from B′]
            + q_adv                             [advective; only with BprimeEvaluator]
            − q_cond                            [conduction into solid; from F_cond reduction]
```

where `q_adv = ṁ_g'' (h_g − h_w) + ṁ_char'' (h_c − h_w)` is activated only when the
the chemistry backend exposes `surface_enthalpies(T_w, p, Z_C)`.  This is
provided by `BprimeEvaluator` and by B′ tables generated with pretabulated
`h_g`/`h_c` on the same reference as `h_wall`.

---

## 4  Summary of Key Differences

| Aspect | Amar (2006) | PATO | SCAM |
|--------|-------------|------|------|
| **Primary energy variable** | Internal energy `e = h_f^o + ∫Cv dT` | Enthalpy `h` | Sensible enthalpy `h(T) = ∫₀ᵀ cp dT'` |
| **Spatial discretisation** | CVFEM (control-volume FEM) | FVM (OpenFOAM) | FVM (cell-centred) |
| **Decomposition energy** | Implicit in `h_f^o` datum shift | Explicit: `hs` table + `hp` scalar | exact storage carries `hs`; Q_vol carries `h_bar_chemical` (Path A) OR `h_decomp` chemical `hp` (Path B) |
| **Decomp. energy sign** | n/a (datum difference) | negative = exothermic | positive = endothermic |
| **Decomp. energy per-reaction** | n/a | `h[2][i]` (TACOT: −4 MJ/kg) | `h_decomp` per component (Path B only) |
| **Formation enthalpies in YAML** | Required (`h_f^o` per phase) | In enthalpy tables | Not stored; only difference matters |
| **Iteration** | Single-pass (direct) | Newton–Picard | Picard on `cp(T^k)` (`max_picard=8`) |
| **Gas continuity** | Fully solved (implicit, Darcy) | Fully solved (OpenFOAM) | Simplified (expansion flux / Darcy option) |
| **Solid kinetics** | Direct integration (Arrhenius, §6) | Arrhenius (linearised) | Direct integration (analytic per nodelet) |
| **Grid / recession** | Contracting grid (ALE) | ALE (moving mesh) | Lagrangian node-drop or ALE (`continuous_remap`) |
| **SEB advective terms** | `ṁ'' h_w` (single `h_w` for all) | `h_g − h_w`, `h_c − h_w` separate | `h_g − h_w`, `h_c − h_w` (live or enthalpy-enabled table) |

---

## 5  Implications for Importing Amar Material Data into SCAM

When converting Amar's Appendix D properties to a SCAM YAML:

1. **Kinetics** (`A_rate`, `E_act`, `m_exp`): directly usable after unit conversion
   (`E_over_R [°R] → E_act [J/mol]` via `E_act = E_over_R × (9/5) × R`).

2. **Thermal tables** (`k`, `cp`): directly usable after unit conversion.

3. **Pyrolysis gas enthalpy `h_g`**: Amar's table is on an absolute reference
   (`h_g(T) = [h_g_input] + h_f^o_gas`).  If `h_f^o_gas ≠ 0`, the table is **not**
   on SCAM's implicit reference (h_g = 0 at some low-T anchor).  SCAM's `h_g` table
   should be re-generated from Cantera for the correct pyrolysis gas composition and
   normalised to a consistent reference (typically `h_g(298 K) ≈ 0` for a pure
   carbon-gas).  Amar's table as-is (with large negative values at 500 K) has not
   been verified against SCAM's reference convention.

4. **`h_decomp`**: cannot be read from Amar's `h_f_*` fields.  Must be computed:
   `h_decomp = h_f_gas(T_pyrolysis) + h_f_char − h_f_virgin` via Hess's law,
   with `h_f_gas` evaluated from Cantera at ~700 K.

5. **Formation enthalpies** (`h_f_virgin`, `h_f_char`, `h_f_gas` in YAML): these are
   Amar's internal-energy-table anchors.  They have no corresponding field in SCAM's
   schema and should not be interpreted as SCAM material properties.  The YAML keys
   are non-standard for SCAM and will be silently ignored by the loader.

6. **`inert_components`**, `E_over_R`, `T_min`: non-standard SCAM schema fields in
   Amar's carbon_phenolic.yaml; the loader ignores them.  `inert_components` should
   be folded into the binder contribution to `rho_char`.

7. **B′ table**: Amar's Appendix C format (P, B′_g, B′_c, T_w, h_w) is consistent
   with SCAM's 3-D B′ table format (for example `tacot_v3.0_bprime_air.yaml`).  A provisional table has
   been generated as `carbon_phenolic_bprime_air.yaml` using Cantera with estimated
   pyrolysis gas composition.

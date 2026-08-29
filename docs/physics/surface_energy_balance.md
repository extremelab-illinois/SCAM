<!-- SPDX-License-Identifier: MIT -->
# Surface Energy Balance

This note explains the surface energy balance (SEB) used by SCAM's
`ENERGY_BALANCE` boundary condition.  It is written for developers debugging
PATO-style ablation cases, especially TACOT/TACOT 3.0 cases where the B' lookup,
wall enthalpy, advective enthalpies, blowing correction, and cooldown branch must
all use compatible references.

For the diagnostic chemistry modes in the TACOT 3.0 ablation2 comparison, see
[`surface_chemistry_modes.md`](surface_chemistry_modes.md).

The implementation lives mainly in:

- `scam/physics/surface_energy.py::seb_residual`
- `scam/solvers/surface_solver.py::solve_surface`
- `scam/physics/chemistry.py::lookup_b_prime`
- `scam/io/material_loader.py::BPrimeTable`
- `scam/physics/bprime_evaluator.py::BprimeEvaluator`
- `scam/numerics/assembly.py::compute_F_cond`

## 1. Sign Convention

SCAM uses the following flux convention at the front surface:

```text
positive heat flux = into the material
q_cond > 0         = heat conducted from the surface into the solid
```

The SEB residual is assembled so that the solved wall temperature satisfies

```text
residual(T_w) = heat into surface - heat conducted into solid = 0
```

In code:

```python
q_cond = alpha_F * T_w + beta_F
residual = q_conv + q_adv + q_rad_in - q_rad_out - q_mass_removal - q_cond
```

The Newton solve changes `T_w` until `residual == 0`.

## 2. Coupling to the In-Depth Solver

The in-depth energy equation is tridiagonal.  Before solving the surface
temperature, SCAM partially eliminates the tridiagonal system to express the
unknown conductive flux as a linear function of wall temperature:

```text
q_cond(T_w) = alpha_F * T_w + beta_F
```

That reduction is computed by `numerics/assembly.py::compute_F_cond`.  The SEB
Newton solve only sees a scalar residual in `T_w`; once `T_w` is found, the
already-built tridiagonal system is patched with the resolved `q_cond` and solved
for the full in-depth temperature field.

This is important: the SEB is not a prescribed heat flux.  It is a coupled
boundary condition where the surface temperature, char mass flux, blowing
correction, radiation, and in-depth conduction are solved together.

## 3. Main Enthalpy-Based B' Form

For PATO-style Bprime cases, the active heating branch is:

```text
q_conv = rhoUeCH_eff * (h_r - h_wall)
```

where:

- `rhoUeCH` is the enthalpy-transfer coefficient from the boundary condition
  table, with units kg/m2/s.
- `h_r` is the recovery enthalpy.
- `h_wall` is the equilibrium wall-gas enthalpy returned by the B' backend.
- `rhoUeCH_eff` is `rhoUeCH` reduced by blowing.

The B' lookup also gives the unblown char ablation coefficient:

```text
B'_c = Bprime table or Cantera result
m_dot_char_unblown = B'_c * rho_e_u_e * C_M
```

The pyrolysis-gas blowing parameter is formed from the in-depth gas flux:

```text
B'_g = m_dot_pyro / (rho_e_u_e * C_M)
```

Then SCAM computes the total blowing factor on the unblown basis:

```text
B_total = (max(m_dot_char_unblown, 0) + max(m_dot_pyro, 0)) / denom_blow
F_blow  = 1 / (1 + lambda_blowing * B_total)
```

The same factor reduces both heat and char mass transfer:

```text
rhoUeCH_eff = rhoUeCH * F_blow
m_dot_char  = m_dot_char_unblown * F_blow
```

This shared reduction is intentional.  It follows the Reynolds-analogy logic
used by the CMA/Bprime formulation: injected gases reduce heat and mass transfer
together.  Applying blowing only to `q_conv` but not to `m_dot_char` overpredicts
recession and changes the thermal state through too much surface mass loss.

## 4. Why `q_mass_removal` Is Usually Zero in the B' Branch

In older heat-flux style equations it is natural to write an explicit sink like:

```text
-(m_dot_char + m_dot_pyro) * h_wall
```

In the enthalpy-based Bprime branch, SCAM does not add that sink separately.
The term is already folded into the standard CMA/PATO-style convective expression:

```text
q_conv = rhoUeCH_eff * (h_r - h_wall)
```

Adding another mass-removal term in this branch would double-count the wall-gas
enthalpy loss.  Therefore:

```python
q_mass_removal = 0.0
```

when the B' lookup has run and `rhoUeCH > 0`.

There is a separate temperature-based branch for cases without an enthalpy B'
lookup, where `alpha_conv * (T_aw - T_w)` is used.  In that branch, explicit
mass-removal accounting is only allowed when a B' lookup actually ran and the
enthalpy references are consistent.

## 5. PATO Advective Terms: `qAdvPyro` and `qAdvChar`

PATO's `Bprime` boundary condition includes two surface advective enthalpy terms:

```text
qAdvPyro = mDotGw * (h_g - h_w)
qAdvChar = mDotCw * (h_c - h_w)
```

SCAM implements the same correction as:

```text
q_adv = m_dot_pyro * (h_g - h_wall)
      + m_dot_char * (h_c - h_wall)
```

These terms are physically separate from the B' mass balance:

- `B'_c` controls the char mass flux and recession.
- `h_wall` controls the enthalpy-based convective heating term.
- `h_g` is the incoming pyrolysis-gas enthalpy at the wall.
- `h_c` is the incoming char/surface-carbon enthalpy at the wall.

The signs matter:

- `h_g - h_wall` is often negative, so pyrolysis gas can be a surface cooling
  contribution.
- `h_c - h_wall` is often positive, so char oxidation/reaction can be a surface
  heating contribution.
- The net `q_adv` can be tens of kW/m2, large enough to move `T_wall` by tens of
  kelvin while barely changing recession.

## 6. Enthalpy Reference Consistency

The advective terms are only meaningful when `h_g`, `h_c`, and `h_wall` are on
the same thermochemical reference.  This is the central rule for debugging SEB
differences:

```text
Never subtract enthalpies from different references.
```

In SCAM:

- `h_wall` from a B' lookup is an equilibrium wall-gas enthalpy.
- `h_g` and `h_c` from `BprimeEvaluator.surface_enthalpies()` are Cantera
  enthalpies on the same elemental reference as the Cantera wall gas.
- `enthalpy_char(mat, T)` from material cp integration is a sensible enthalpy on
  SCAM's internal reference, not the same reference as Cantera wall gas.

Therefore `q_adv` is gated on `_bprime_ran`.  If the B' lookup did not run, then
`h_wall` falls back to the material char sensible enthalpy.  In that state,
computing `h_g - h_wall` with Cantera `h_g` would inject a fake source or sink
dominated by reference offset.

The code intentionally does:

```python
q_adv = 0.0
if _bprime_ran and hasattr(b_prime_table, "surface_enthalpies"):
    try:
        h_g_w, h_c_w = b_prime_table.surface_enthalpies(T_w, p_e, Z_C_pyro)
        q_adv = ...
    except Exception:
        q_adv = 0.0
```

Note that `BPrimeTable` always has a `surface_enthalpies()` method in current
SCAM, but old tables without stored `h_g`/`h_c` raise `AttributeError` when the
method is called.  The `try/except` is what makes old tables fall back to
`q_adv = 0`.

## 7. Table B' Backends vs Live Cantera

There are two independent questions:

```text
1. How are B'_c and h_wall obtained?
2. How are h_g and h_c for q_adv obtained?
```

A table and live Cantera can agree perfectly on `B'_c` and `h_wall` while still
using different `h_g`/`h_c` values.  This is exactly the trap that appears in
TACOT 3.0 debugging.

### 7.1 B' lookup

The B' lookup is:

```text
(T_wall, p_e, B'_g[, Z_C_pyro]) -> (B'_c, h_wall)
```

If the table was generated from Cantera with the same gas composition, pressure,
temperature, and `B'_g`, then:

```text
BPrimeTable.lookup(...) ~= BprimeEvaluator.lookup(...)
```

up to interpolation error.

### 7.2 Advective enthalpy lookup

The advective lookup is:

```text
(T_wall, p_e[, Z_C_pyro]) -> (h_g, h_c)
```

For 3-D tables generated by `scam/tools/generate_bprime.py`,
`csv_to_scam_yaml()` stores `h_g` and `h_c` as 2-D arrays over `(T, p)` when it
receives `bprime_config_path`.  In current `main()`, this is done by default for
the 3-D path:

```python
csv_to_scam_yaml(..., bprime_config_path=bprime_config)
```

The stored values are computed with:

```python
Z_C_nominal = ev.pyrolysis_target_fraction("C")
ev.surface_enthalpies(T, p, Z_C_nominal)
```

Using the explicit nominal elemental fraction is intentional: it follows the
same molecular `Z_C_pyro -> pyro_x` reconstruction used by the live validation
backend.  The nominal fraction is recorded as `surface_enthalpy_Z_C_pyro` in the
table metadata.

For 4-D element-transport tables, the generator stores
`h_g(T,p,Z_C_pyro)` and `h_c(T,p)`.  The loader clamps/interpolates the
composition axis in the same way as the B′ lookup.

That difference affects `q_adv`, not recession.  A common symptom is:

```text
recession matches PATO
surface temperature is still off by tens of kelvin
B'_c comparison looks perfect
```

The correct conclusion is not "B' is wrong"; it is "the advective enthalpy path
is not using the same composition/reference convention."

## 8. TACOT 3.0 Composition Detail

For TACOT 3.0, the Cantera config uses the pyrolysis stream:

```text
CH4:0.5551, CO:0.2418, H2O:0.2031
```

This matches the intended TACOT pyrolysis-gas elemental inventory.  The nominal
carbon mass fraction is approximately:

```text
Z_C_pyro = 0.494984
```

Passing `Z_C_pyro` into `BprimeEvaluator` does not strongly change `B'_c` or
`h_wall` for the base ablation2 conditions, but it can strongly change the
pyrolysis-gas enthalpy returned by `surface_enthalpies()` because the evaluator
reconstructs a gas composition from the elemental carbon fraction.

For that reason, both the canonical live comparison and newly generated TACOT
tables use:

```text
explicit nominal Z_C_pyro for the advective enthalpy composition
```

This keeps the tabulated mass balance while making the PATO-style
`qAdvPyro/qAdvChar` terms use the same convention as the live Cantera comparison.

## 9. Cooldown Branch

PATO's ablation2 cooldown is easy to misread.  The boundary table still contains
a small nonzero `rhoUeCH`, but `chemistryOn=0` disables B' chemistry.  In that
branch, PATO's Bprime temperature boundary condition ignores the enthalpy columns
and behaves like a temperature convection branch.

SCAM represents this by setting the B' chemistry driver `rho_e_u_e` to zero
during cooldown.  Consequences:

- no B' lookup,
- no h_wall on the Cantera/B' reference,
- no `q_adv`,
- no mass-removal enthalpy sink from B' chemistry.

This prevents a large artificial cooling sink from subtracting Cantera gas enthalpy
against material sensible char enthalpy.

## 10. Radiation

Radiation is split into incoming and outgoing pieces:

```text
q_rad_in  = epsilon * view_factor * sigma * T_rad_in^4
q_rad_out = epsilon * view_factor * sigma * T_w^4
```

Then:

```text
residual += q_rad_in - q_rad_out
```

When `emissivity_override > 0`, the boundary condition value is used directly.
Otherwise SCAM uses `properties.surface_emissivity(mat, T_w, rho_surface)`,
which blends virgin and char emissivity by the local surface char fraction and
supports temperature-dependent emissivity tables.

For PATO TACOT 3.0 comparisons, the intended behavior is to use the material's
virgin/char emissivity data, not a hard-coded constant override.

## 11. Practical Debug Checklist

When `T_wall` disagrees with PATO but recession or B' comparisons look good,
check these in order:

1. Confirm `B'_c` and `h_wall` agree for the same `(T_wall, p_e, B'_g)`.
2. Confirm the actual blown `m_dot_char` uses the same blowing factor as
   `rhoUeCH_eff`.
3. Print `h_g - h_wall` and `h_c - h_wall` for the backend being used.
4. Confirm table mode and Cantera mode use the same `Z_C_pyro` convention for
   `surface_enthalpies()`.
5. During cooldown, confirm `_bprime_ran == False` and `q_adv == 0`.
6. Confirm `T_rad_in`, emissivity, and `view_factor` match the PATO case.
7. Confirm `q_cond = alpha_F*T_w + beta_F` has the expected sign and magnitude.

Useful symptoms:

```text
B'_c matches, recession matches, T_wall cold:
    likely q_adv or h_wall/h_g/h_c reference issue

recession too high and T_wall cold:
    likely char mass flux / blowing correction issue

cooldown much too cold:
    likely Cantera advective terms or mass-removal sink active after chemistry off

startup or cutoff max error huge but aligned-time values look good:
    likely comparison sampling artifact around steep transients
```

## 12. Current Residual Equation in One Place

For the active B' enthalpy branch:

```text
B'_g       = m_dot_pyro / (rho_e_u_e*C_M)
(B'_c,h_w) = Bprime(T_w, p_e, B'_g, Z_C_pyro)

m_dot_c0  = B'_c * rho_e_u_e * C_M
B_total   = (max(m_dot_c0,0) + max(m_dot_pyro,0)) / (rho_e_u_e*C_M)
F_blow    = 1 / (1 + lambda_blowing*B_total)

m_dot_c   = m_dot_c0 * F_blow
rhoUeCH_eff = rhoUeCH * F_blow

q_conv = rhoUeCH_eff * (h_r - h_w)
q_adv  = m_dot_pyro * (h_g - h_w) + m_dot_c * (h_c - h_w)
q_rad  = epsilon*sigma*(T_rad_in^4 - T_w^4)
q_cond = alpha_F*T_w + beta_F

residual(T_w) = q_conv + q_adv + q_rad - q_cond
```

with the important qualifications:

- `q_adv` is zero unless `h_g/h_c/h_w` are on one reference and the B' lookup ran.
- `m_dot_char` in `q_adv` is the blown char flux.
- `B_total` is formed on the unblown basis.
- During cooldown chemistry-off branches, do not use the enthalpy B' form.

<!-- SPDX-License-Identifier: MIT -->
# Boundary conditions

## Surface

For `surface_bc.type: energy_balance` (the default and primary mode), the
solver iterates Newton on `T_wall` using the residual

```text
q_conv(T_w) + q_rad_in − q_rad_out(T_w) + q_chem(T_w)
  − (m_dot_char + m_dot_g)·h_w(T_w) − F_cond(T_w) = 0
```

`F_cond` is the linear coupling to the in-depth conduction solve
(`q_cond = alpha_F·T_w + beta_F`), produced by backward elimination of the
tridiagonal system — see the theory manual's
[governing equations](../theory/index.md) and
[surface energy balance](../theory/02_surface_energy_balance.md) pages.

Two mutually-exclusive convective modes are supported, selected by whether
`rhoUeCH` is set:

**Temperature-based (default)** — `alpha_conv` + `T_aw`:

```text
q_conv = alpha_conv · (T_aw − T_w)
```

**Enthalpy-based (PATO-style)** — `rhoUeCH` + `h_r`, active whenever
`rhoUeCH > 0` (which then makes `alpha_conv`/`T_aw` irrelevant, whether or
not they are also set):

```text
q_conv = rhoUeCH · (h_r − h_gas(T_w))
```

`h_gas(T_w)` is looked up from the surface material's `h_g` table. In the B′
chemistry branch, `rhoUeCH`/`h_r` instead couple to the B′ wall enthalpy; if
that chemistry branch is inactive, the solver falls back to the
temperature-based form.

All scalar parameters (`alpha_conv`, `T_aw`, `rhoUeCH`, `h_r`, `rho_e_u_e`,
`C_M`, `p_e`, `T_rad_in`) accept either a constant float or a
`[[t, v], ...]` table in the YAML — see
[`01_yaml_case_deck.md`](../reference/01_yaml_case_deck.md).

### `prescribed_flux` and `prescribed_temp`

The other two `surface_bc.type` values bypass the SEB Newton solve
entirely:

- `prescribed_flux`: `q_prescribed(t)` [W/m²], positive into the material.
- `prescribed_temp`: `T_prescribed(t)` [K] is imposed directly as the
  surface node's temperature.

`ablation1_template.yaml` (see
[`04_case_deck_walkthrough.md`](04_case_deck_walkthrough.md)) uses
`prescribed_temp` to isolate the in-depth conduction and pyrolysis physics
from the surface chemistry and recession.

### Blowing correction

The blowing correction reduces the effective heat- and mass-transfer
coefficients when pyrolysis gas and char products are injected into the
boundary layer (Reynolds analogy: `C_H` and `C_M` drop together). Three
models are available via `blowing_model`:

| Model | Formula | Notes |
|---|---|---|
| `"rational"` (default) | `1/(1 + λ·B_total)` | SCAM legacy, Amar-style. |
| `"kays"` | `Φ/(exp(Φ) − 1)`, solved implicitly for the char flux | Amar Eq. 42. |
| `"lees"` | `log(1+Φ)/Φ` | PATO's `constantLambdaBlowingCorrectionModel`; also looks up `B'_g` on the blown basis, matching PATO's `BprimeBoundaryConditions.C`. |

`B_total = B'_c + B'_g` is formed on the **unblown** basis so the algebraic
correction stays bounded under heavy injection. `lambda_blowing`
(0.5 laminar, 0.4 turbulent by convention) is the exponent.

```{warning}
`blowing_model` is documented on the `SurfaceBCConfig` dataclass and can be
set in a YAML deck, but **the loader does not currently read it** — see the
warning in [`01_yaml_case_deck.md`](../reference/01_yaml_case_deck.md). Every
YAML-driven run uses `"rational"` today regardless of what the deck says.
```

## Back face

`back_bc.type` selects one of four modes:

| Type | Behaviour |
|---|---|
| `adiabatic` (default) | Zero heat flux — insulated back face. |
| `prescribed_temp` | Fixed or time-varying temperature `T_back(t)`. |
| `prescribed_flux` | Fixed or time-varying flux `q_back(t)` [W/m²], positive into the material. |
| `radiation` | Re-radiation to an environment at `T_env_back`, optionally plus a convective film. |

The `radiation` mode combines a net radiative loss and an optional
convective film into a single temperature-dependent flux:

```text
q = emissivity_back · σ · view_factor_back · (T_N^4 − T_env_back^4)
    + h_back · (T_N − T_env_back)
```

The `T⁴` term is linearised about the previous iterate each Picard/timestep
pass (Newton form), so it needs no inner iteration. `radiation` requires at
least one of `emissivity_back` or `h_back` to be set — with both at their
zero defaults the loader raises `SCAMInputError` rather than silently
behaving like `adiabatic`.

<!-- SPDX-License-Identifier: MIT -->
# Nomenclature

Symbols used across the theory manual and the source code, gathered in one
place. SI units throughout; heat flux is positive into the material.

## Geometry and mesh

| Symbol | Meaning | Units |
|---|---|---|
| `y` | Depth coordinate, measured from the **original** front face (`y_nodes[0] == s_total` at the surface node) | m |
| `s_total` | Total surface recession since `t=0` | m |
| `s_dot` | Surface recession rate | m/s |
| `A(y)` | Cross-sectional area function (slab: constant; hollow cylinder: `r_inner + y`) | m² (or m²/rad) |

## Temperature, density, and decomposition

| Symbol | Meaning | Units |
|---|---|---|
| `T_wall`, `T_w` | Surface node temperature, the unknown in the Newton SEB solve | K |
| `ρ` | Nodal apparent density (volume-weighted average over nodelets) | kg/m³ |
| `ρ_virgin`, `ρ_v` | Virgin (undecomposed) apparent density | kg/m³ |
| `ρ_char`, `ρ_c` | Fully-charred apparent density | kg/m³ |
| `β` | Char fraction, `(ρ_v − ρ) / (ρ_v − ρ_c)`, 0 = virgin, 1 = fully charred | – |
| `β_c` | Per-Arrhenius-component char fraction, `(ρ_0_c − ρ_c) / (ρ_0_c − ρ_r_c)` | – |
| `m_dot_pyro`, `m_dot_g` | Pyrolysis gas mass flux (from `−dρ/dt`, accumulated through the porous char) | kg/(m²·s) |
| `m_dot_char` | Char consumption mass flux at the surface (drives recession) | kg/(m²·s) |

## Surface energy balance

| Symbol | Meaning | Units |
|---|---|---|
| `q_cond` | Conduction flux into the solid at the surface node | W/m² |
| `q_conv` | Convective heat flux (temperature-based `alpha_conv·(T_aw − T_w)`, or enthalpy-based `rhoUeCH·(h_r − h_wall)`) | W/m² |
| `alpha_conv` | Convective heat transfer coefficient (temperature-based mode) | W/(m²·K) |
| `T_aw` | Adiabatic wall (recovery) temperature (temperature-based mode) | K |
| `rhoUeCH` | Lumped mass/heat-transfer coefficient `ρ_e·u_e·C_H` (enthalpy-based mode); overrides `alpha_conv`/`T_aw` when `> 0` | kg/(m²·s) |
| `h_r` | Freestream recovery enthalpy (enthalpy-based mode), same reference as the material's `h_g` table | J/kg |
| `h_wall`, `h_w` | Wall gas enthalpy, from the B′ lookup or fallback to `enthalpy_char` | J/kg |
| `rho_e_u_e` | Freestream/edge mass flux `ρ_e·u_e` | kg/(m²·s) |
| `C_M` | Mass-transfer Stanton number; `m_dot_char = rho_e_u_e · C_M · B'_c` | – |
| `p_e` | Boundary-layer edge pressure, used for the B′ table lookup | Pa |
| `alpha_F`, `beta_F` | Coefficients of the linear coupling `q_cond = alpha_F·T_w + beta_F` produced by backward elimination of the tridiagonal system; this is what the SEB Newton solve iterates against | W/(m²·K), W/m² |

## Surface chemistry (B′)

| Symbol | Meaning | Units |
|---|---|---|
| `B'_g` | Pyrolysis gas blowing parameter, `m_dot_g / rho_e_u_e` | – |
| `B'_c` | Char blowing / char mass-loss coefficient, from the B′ table or live chemistry backend | – |
| `B_total` | `B'_c + B'_g`, formed on the unblown basis so the blowing correction stays bounded | – |
| `Z_C_pyro` | Carbon elemental mass fraction of the pyrolysis gas at the wall, fed to the 4-D B′ table when element transport is enabled | – |
| `Z_elem` | Elemental mass fraction field (C, H, O, N) transported through the porous medium | – |
| `lambda_blowing`, `λ` | Blowing correction exponent (0.5 laminar, 0.4 turbulent) | – |

## Energy formulation

| Symbol | Meaning | Units |
|---|---|---|
| `h(T)` | Sensible enthalpy `∫₀ᵀ cp dT'`, used instead of `cp·T` in the tridiagonal RHS | J/kg |
| `h_bar` | Per-reaction chemical decomposition enthalpy source (PATO's `hs`, the virgin/char enthalpy difference) | J/kg |
| `h_decomp` | Per-component decomposition enthalpy (`ComponentCard.h_decomp`), used when the material has no explicit virgin/char enthalpy tables | J/kg |
| `h_g_abs_offset` | Sensible-to-absolute reference offset applied to a material's `h_g` table so it matches Cantera's formation-enthalpy reference | J/kg |

See [`02_surface_energy_balance.md`](02_surface_energy_balance.md),
[`04_energy_formulation_comparison.md`](04_energy_formulation_comparison.md),
and [`03_surface_chemistry_modes.md`](03_surface_chemistry_modes.md) for the
equations these symbols appear in.

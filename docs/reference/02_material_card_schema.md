<!-- SPDX-License-Identifier: MIT -->
# Material card schema

A material card is a YAML file describing one material's density, thermal
properties, decomposition kinetics, and surface-chemistry closure, loaded by
`scam.io.material_loader.load_material`. See
[`06_material_library.md`](../user/06_material_library.md) for the material
catalogue and [`07_adding_a_material_card.md`](../user/07_adding_a_material_card.md)
for a step-by-step walkthrough of writing a new one.

Property tables are `[[T_K, value], ...]` pairs, linearly interpolated, with
**flat extrapolation** beyond the table's range (the edge value is held, not
extended by slope).

## Required fields

| YAML key | Type | Notes |
|---|---|---|
| `name` | string | Must match a `stack.layers[].material` entry in the case deck. |
| `rho_virgin` | float | Total virgin apparent density [kg/m³]. |
| `rho_char` | float | Total char apparent density [kg/m³]. |
| `gamma_resin` | float | Resin volume fraction in the virgin composite [-]. |
| `k_virgin`, `k_char` | table | Thermal conductivity [W/(m·K)] vs T, virgin and char phases. |
| `cp_virgin`, `cp_char` | table | Specific heat [J/(kg·K)] vs T, virgin and char phases. |
| `h_g` | table | Pyrolysis gas specific enthalpy [J/kg] vs T, on the **sensible** reference (`h_g(298 K) ≈ 0`) unless `h_g_abs_offset` is also given. |

Density blending between virgin and char follows a linear mixture rule on
the local virgin fraction `eps_virgin = (ρ − ρ_char)/(ρ_virgin − ρ_char)`:
`k(T,ρ) = eps_virgin·k_virgin(T) + (1 − eps_virgin)·k_char(T)`, and similarly
(density-weighted) for `cp`.

## Decomposition

| YAML key | Type | Default | Notes |
|---|---|---|---|
| `decomposing` | bool | `true` | Set `false` for inert layers; then `components` must be empty. |
| `components` | list | `[]` | One entry per Arrhenius decomposition reaction. Required for decomposing materials. |

Each `components[]` entry:

| Key | Required | Notes |
|---|---|---|
| `name` | yes | |
| `rho_0` | yes | Initial (virgin) apparent density of this component [kg/m³]. |
| `rho_r` | yes | Residual (char) apparent density of this component [kg/m³]; must be `< rho_0`. |
| `A_rate` | yes | Arrhenius pre-exponential [1/s]. |
| `m_exp` | yes | Reaction order exponent [-]. |
| `E_act` **or** `E_over_R` | yes (one of) | Activation energy [J/mol], or `E_act/R` directly [K]. |
| `h_decomp` | no (default `0.0`) | Decomposition enthalpy [J/kg]; endothermic (absorbs heat) is positive. |

The nodal density formula is
`rho = fiber + sum_i(vol_avg(rho_i))` where
`fiber = rho_char − sum_i(rho_r_i)` — see the "Nodelet subgrid" note in
CLAUDE.md if you are extending the decomposition solver itself.

## Radiation

| YAML key | Type | Default | Notes |
|---|---|---|---|
| `emissivity` | float | `0.85` | Virgin-phase scalar emissivity. |
| `emissivity_char` | float | `-1.0` | Char-phase scalar emissivity; `-1` means "same as virgin". |
| `emissivity_virgin` | table | — | Optional `[[T_K, ε], ...]`; overrides the scalar `emissivity` when present. |
| `emissivity_char_table` | table | falls back to `emissivity_virgin` | Optional; only meaningful if `emissivity_virgin` is also given. |

When a temperature-dependent table is present, ε is interpolated at the wall
temperature per phase, then blended by the local char fraction — see
`properties.py::surface_emissivity`.

## Pyrolysis gas and porosity

| YAML key | Type | Default | Notes |
|---|---|---|---|
| `porosity` | table | — | Optional `[[·, ε_g], ...]`; if given, its first and last row seed `eps_g_virgin`/`eps_g_char` unless those are set explicitly. |
| `eps_g_virgin`, `eps_g_char` | float | `0.0` each | Gas porosity of the virgin and fully-charred material [-]. Both zero (default) skips the gas energy-storage correction entirely. |
| `gas_molar_mass` | float | `0.022` | Pyrolysis gas molar mass [kg/mol]. |
| `gas_molar_mass_table`, `gas_viscosity_table` | table | — | Optional T-dependent overrides. |
| `gas_pressure` | float | `101325.0` | Ambient pressure for the ideal-gas ρ_g used by the 1-D ambient-pressure gas terms. |
| `gas_properties_pT` | string (companion file) | — | Path to a PATO-format gasProperties `(p, T)` table, resolved relative to this card's directory. When present, μ, M, and ρ_g use bilinear `(T, p)` lookups instead of the scalar/T-only tables above. |
| `h_g_abs_offset` | float | `None` | Sensible→absolute reference offset [J/kg] for `h_g`. Set this when `h_g` is on the sensible reference and you need the SEB advective terms (`q_adv`), which need `h_g` and `h_wall` on the same reference. **Do not** set it if `h_g` (or `gas_properties_pT`) is already absolute — see the "double-application trap" in `docs/verification/02_pato_validation.md` §11. |

## Transport (Darcy / element)

| YAML key | Type | Default | Notes |
|---|---|---|---|
| `permeability` | float or table | `0.0` | Char (or single-value) Darcy permeability [m²]. A table's last row seeds the char value and first row seeds the virgin default. `0.0` skips the pressure-driven Darcy solve. |
| `permeability_virgin` | float | permeability table's first row, or `0.0` | Virgin permeability; blended with `permeability` by char fraction. |
| `klinkenberg_b` | float | `0.0` | Klinkenberg slip-correction coefficient [Pa]; significant only below ~1 kPa. |
| `tortuosity` | float | `1.0` | Pore tortuosity; effective element diffusivity = `element_diffusivity / tortuosity`. |
| `pyro_elem_fracs` | dict of tables, keyed `C`/`H`/`O`/`N` | — | Elemental mass fraction of the pyrolysis gas vs T, per element. Required for `element_transport: true` in the solver. |
| `initial_Z_elem` | `[Z_C, Z_H, Z_O, Z_N]` | ambient air | Initial in-material gas-phase elemental composition. |

## Surface chemistry closure — pick at most one

| YAML key | Notes |
|---|---|
| `b_prime_table` | Path to a companion B′ table YAML (3-D, 4-D, or row-format — see [`08_surface_chemistry_backends.md`](../user/08_surface_chemistry_backends.md)), resolved relative to this card's directory. |
| `kinetic_ablation` | A Kemp (1968) Eq. 12 reaction-rate-limited closure; see below. |

Setting **both** raises `SCAMInputError` at load time ("ambiguous ablation
closure — pick one").

`kinetic_ablation:` sub-fields:

| Key | Required | Notes |
|---|---|---|
| `B` | yes | Pre-exponential [1/s]; must be `> 0`. |
| `E_a` | yes | Activation energy [J/mol]; must be `> 0`. |
| `h_ablation_total_coeffs` | yes | `[a, b, c]` quadratic-in-T fit to the full heat of ablation [J/kg]. |
| `h_ablation_reaction_coeffs` | yes | `[a, b]` linear-in-T fit to the depolymerization-only enthalpy [J/kg]. |
| `rho_sw_override` | no | Override the reacting/wall density if it differs from `rho_char`. |

This closure is for materials whose ablation rate is set by solid-state
Arrhenius kinetics rather than boundary-layer mass transfer (e.g. PTFE) — see
`scam/materials/ablative_organic/teflon.yaml` and
`studies/teflon_ablation/why_ptfe_mdot_depends_on_Tw_not_p.md`.

## Metadata (not used by the solver)

| YAML key | Type | Default | Notes |
|---|---|---|---|
| `material_category` | string | `None` | e.g. `ablative_organic`. |
| `dataset_version` | string | `None` | Upstream dataset release. The legacy key `version` is also accepted. |
| `card_version` | string | `"1.0"` | SCAM-internal revision. |
| `status` | string | `"provisional"` | `verified` \| `provisional` \| `estimate` — see [`06_material_library.md`](../user/06_material_library.md). |

## In-plane conductivity (stored, unused by the 1-D solver)

`k_virgin_ip`, `k_char_ip` — reserved for future anisotropy support.

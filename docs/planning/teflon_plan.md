<!-- SPDX-License-Identifier: MIT -->
# Fluorine Chemistry Support Plan

## Summary

SCAM can load an estimate-only PTFE/Teflon material card today, but PTFE should not be treated like a charring pyrolysis material. It is a low-temperature surface sublimator/volatilizer with nearly no in-depth pyrolysis. The current B' surface-chemistry path is built around carbon ablation: `B'_c`, `m_dot_char`, `h_c`, and optional element transport via `Z_C_pyro`. Real PTFE support requires generalizing that path so surface mass loss can represent fluoropolymer volatilization rather than carbon-char oxidation.

The recommended first implementation is a fixed-composition fluorine chemistry path: add a fluorine-capable Cantera mechanism and PTFE B' table support, while leaving full fluorine element transport and in-depth PTFE decomposition for later passes.

## Key Changes

- Add a Cantera gas mechanism covering at least `C, F, O, N, H` and PTFE-relevant species such as `CF2`, `CF`, `CF4`, `C2F4`, `C2F6`, `F`, `F2`, `HF`, `COF2`, `CO`, `CO2`, `O2`, and `N2`.
- Generalize the B' runtime vocabulary from carbon-only names (`B'_c`, `m_dot_char`, `h_c`) to a surface-ablation concept such as `B'_surface`, `m_dot_surface`, and `h_surface`, while preserving carbon aliases for existing TACOT/PATO validation cases.
- Add material/config metadata for the B' target element or surface reservoir, defaulting to carbon for existing cards and using fluorine-capable fixed PTFE composition for Teflon.
- Create PTFE B' generation inputs and a pretabulated table over `(T_wall, p_e, B'_g)` for runtime use without live Cantera.
- Represent PTFE v1 as `decomposing: false`, `components: []`, and `rho_char == rho_virgin`; the existing surface-recession formula then divides B' surface mass loss by the solid PTFE density instead of a fictitious low-density char.
- Keep PTFE v1 independent of `element_transport`; do not add `F` to the transported element vector until the fixed-composition surface chemistry path is working.

## Test Plan

- Verify all existing TACOT, carbon, and surface-energy tests remain unchanged numerically.
- Add unit coverage for non-carbon B' metadata and backward-compatible carbon defaults.
- Add loader/table tests for a PTFE fluorine B' config and generated table shape.
- Add a smoke test that runs a short PTFE surface-energy case using the pretabulated table and produces finite `T_wall`, surface mass flux, and recession.
- Add reference checks only after suitable PTFE TGA, arc-jet, or equilibrium-composition data are selected.

## Assumptions

- First-pass PTFE chemistry uses a fixed surface-vapor composition, not transported fluorine.
- PTFE surface loss is treated as surface sublimation/volatilization, not in-depth pyrolysis or carbon-char recession.
- Fluorine thermochemistry quality is the main scientific risk; no quantitative PTFE validation claim should be made until the mechanism is checked against literature or independent equilibrium results.

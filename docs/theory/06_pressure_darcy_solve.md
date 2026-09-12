# Gas Pressure (Darcy) Solve

SCAM computes an in-depth pyrolysis-gas pressure field to drive Darcy transport
through the porous char/virgin material.

```{note}
This page covers the 1-D solve only. A 2-D unstructured-mesh implementation
shares the same underlying physics (permeability blend, Klinkenberg slip,
gas viscosity/molar-mass lookups) but differs in source-term treatment and
discretization; it is documented on this section's `main`-only
`90_two_dimensional.md` page, since the 2-D solver does not exist on
`public-release`.
```

## `scam/physics/pressure_darcy.py`

### Governing equation

Quasi-steady 1-D Darcy pressure equation, solved algebraically every
timestep (no explicit time derivative):

```
d/dy( Γ(y) · dp/dy ) = ε_g(y)/T(y) · dT/dt
```

where `Γ = K/μ_g` is gas mobility [m²/(Pa·s)]. The equation is quasi-steady
because the pressure-diffusion timescale (`τ_p ~ L²·μ·ε_g/(K·p₀) ≈ 0.001 s`)
is orders of magnitude shorter than the thermal timescale (~seconds), so the
pressure field is assumed to relax instantaneously to the current
temperature-rate field.

### Boundary conditions

- Surface (`y=0`): Dirichlet, `p = p_surface` (open to freestream).
- Back face (`y=L`): Neumann, `dp/dy = 0` (no-flow/adiabatic).

### Solve procedure (`solve_gas_pressure`)

1. **Permeability** `K` at each node: virgin/char blend by local virgin
   fraction (`eps_virgin`), with optional Klinkenberg slip correction
   `K_app = K·(1 + klinkenberg_b/p)` (significant only sub-atmospheric).
2. **Gas viscosity**: Sutherland's law by default; bilinear `μ(T,p)` lookup
   if the material carries a full `gas_properties_pT` table.
3. **Face mobility** `Γ_face`: harmonic mean of neighboring nodal `Γ`.
4. **Source integration**: nodal source `ε_g/T · dT/dt`, weighted by cell
   thickness, cumulative-summed back-to-front to get `Γ_face · dp/dy` at
   each face — equivalent to integrating the ODE inward from the no-flow
   back wall.
5. **Gradient**: divide by `Γ_face` to get `dp/dy` at each face.
6. **Integrate forward** from the Dirichlet surface value to build `p(y)`.

### Mass flux (`_mass_flux_from_pressure` / `pressure_driven_mass_flux`)

```
J_g(face) = −ρ_g · (K/μ) · dp/dy      [kg/(m²·s)]
```

`ρ_g` from the ideal gas law (`p·M/(R·T)`), using local `(p,T)` table values
when `gas_properties_pT` is present (matters at sub-atmospheric pressure,
where `M` can deviate from its 1-atm value by ~10%).

`solve_pressure_and_flux` bundles the pressure solve and flux reconstruction
in one call to avoid a duplicate solve.

### Energy coupling (`pressure_darcy_energy_source`)

Face mass fluxes are advected with upwind sensible gas enthalpy
(`h_sens(T,p)`, referenced consistently whether or not a `(p,T)` table is
present) to produce a nodal `Q_adv` source added directly into the
tridiagonal RHS.

### Practical magnitude

For TACOT-like permeabilities (`K ≈ 2×10⁻¹¹ m²`, `μ ≈ 3×10⁻⁵ Pa·s`), the
pressure perturbation is typically `Δp < 1 Pa` (~0.001% of atmospheric) and
the resulting gas energy flux is ~0.036% of the conductive flux — physically
correct but negligible for TACOT gap-closure work. The module exists for
completeness and for higher-permeability or faster-heating cases.

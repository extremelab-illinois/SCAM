# Gas Pressure (Darcy) Solve: 1D and 2D

SCAM computes an in-depth pyrolysis-gas pressure field to drive Darcy transport
through the porous char/virgin material. There are two independent
implementations — a 1D column solve and a 2D unstructured-mesh solve — that
share the same underlying physics (permeability blend, Klinkenberg slip,
gas viscosity/molar-mass lookups) but differ in source-term treatment and
discretization, as described below.

## 1D: `scam/physics/pressure_darcy.py`

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

---

## 2D: `scam/physics2d/darcy_pressure_2d.py`

### Two source-term modes

Selected by whether `pi_source` is passed to `solve_darcy_pressure_2d`.

**Mode 1 — legacy thermal-expansion** (`pi_source=None`, Gate-3 style):
matches the 1D equation exactly, discretized on the 2D mesh:

```
∇·(Γ ∇p) = ε_g/T · dT/dt,      Γ = K/μ
```

**Mode 2 — PATO `DarcyLawMassModel`** (`pi_source` given — production path):

```
∂(Eta·p)/∂t − ∇·(Γ ∇p) − π_total = 0

Eta      = ε_g · M / (R·T)          (compressible storage coefficient)
Γ        = ρ_g · K/μ                (mass mobility — note ρ_g included, unlike 1D)
π_total  = pyrolysis gas production + optional mesh-motion source
```

Unlike the 1D quasi-steady algebra, this mode carries a genuine transient
storage term `Eta·∂p/∂t`, because in 2D gas can vent laterally between mesh
columns and a purely algebraic per-column solve would not conserve mass
across the field the way the 1D column-local treatment can.

### Discretization

Built on the shared `DiffusionOp` FVM operator (the same machinery used for
2D heat conduction), with per-cell `Γ` as conductivity and `Eta` as the
storage ("rho_cp") coefficient.

**Boundary conditions**: Dirichlet `p = p_e` on hot (ablating) faces;
zero-flux Neumann elsewhere (`DiffusionOp` default).

**Compressible hot-face conductance**: because `Γ = ρ_g·K/μ` depends on `p`
in production mode, the face `Γ` must be evaluated at a pressure consistent
with the nonlinear `p·∇p` flux. Using only the interior cell pressure
over-estimates conductance by up to ~(p_interior/p_e) in the decomposition
zone; using only the boundary value under-estimates by the same factor.
`hot_face_pressure_gamma` controls this (`"average"` — arithmetic mean,
default and correct linearization; `"boundary"`; `"owner"` — diagnostic
alternatives for isolating PATO-convention effects).

**Non-orthogonal (mesh-skew) correction**: the IsoQ shoulder mesh has
internal faces skewed up to ~60°, where an orthogonal-only Laplacian
under-transmits gas (SCAM's interior pressure ran 25–40% high vs PATO before
this was added). Three schemes, via `pressure_scheme`:

- **`"scam"`** (default, production) — harmonic Γ face interpolation +
  orthogonal decomposition with a deferred non-orthogonal correction
  (Jasak-style, lagged over `n_nonorthog` passes). Do not change this
  default.
- **`"openfoam"`** — reproduces PATO's actual `fvSchemes` convention (linear
  Γ interpolation, over-relaxed non-orthogonal decomposition). Fixes the
  TC9/TC10 shoulder-probe cooling sign in an isolated snapshot replay, but a
  full 120 s ablation3 transient with this scheme regresses agreement with
  PATO almost everywhere (axis RMS, wall temperature, side RMS) — tried at
  several skew-angle thresholds and block-scoped variants, all regress
  similarly. **Do not use for production or recommend it**; kept only as an
  experimental/diagnostic option. See
  `docs/verification/ablation3_debug_status.md` (2026-07-05).
- **`"implicit"`** — folds the tangential correction directly into the
  system matrix (`DiffusionOp.to_scipy_csr_implicit`) instead of lagging it
  on the RHS, removing the lag/relaxation instability mechanism; always
  solved with BiCGSTAB (matrix not guaranteed symmetric). At full relaxation
  (`pressure_implicit_relax=1.0`) it diverges on the full nonlinear
  transient; at reduced relaxation it closes roughly half the recession gap
  to PATO but regresses pointwise TC RMS at every snapshot. Experimental
  only, not production-default.

### Linear solve

CG or BiCGStab (own implementation or SciPy), Jacobi-preconditioned via
`op.precondition`.

### Face fluxes (`darcy_face_fluxes`)

Once `p` is solved, internal- and hot-face mass fluxes are reconstructed as

```
J_int = ρ_g_face · Γ_face · a_f · (p_owner − p_neigh) + J_corr_int
J_hot = ρ_g_face · Γ_face · a_f · (p_owner − p_e)
```

using whichever face conductances (`a_f`, `bnd_a`) and non-orthogonal
correction (`J_corr_int`) the pressure solve actually used, so that
`div(J) == source − storage·dp/dt` holds cell-by-cell up to linear-solver
tolerance.

---

## Key differences between 1D and 2D

| | 1D (`pressure_darcy.py`) | 2D (`darcy_pressure_2d.py`) |
|---|---|---|
| Equation | Quasi-steady, purely algebraic | Mode 1: same quasi-steady form. Mode 2 (production): transient `Eta·∂p/∂t` storage |
| Mobility `Γ` | `K/μ` | Mode 1: `K/μ`. Mode 2: `ρ_g·K/μ` |
| Pyrolysis gas source | Not directly in the pressure equation; handled elsewhere in the 1-D column bookkeeping | Directly in the equation as `π_total` (Mode 2) |
| Mesh | Structured 1-D column, closed-form back-to-front integration | Unstructured `IsoQMesh`, FVM `DiffusionOp` + iterative linear solve |
| Non-orthogonality | N/A (1-D has no skewed faces) | Explicit correction machinery (`scam`/`openfoam`/`implicit` schemes) |
| Practical magnitude | Δp negligible for TACOT (~1 Pa) | Larger effect at skewed shoulder faces; actively used for ablation3 gas venting |

<!-- SPDX-License-Identifier: MIT -->
# Governing equations: FVM assembly

SCAM discretizes the 1-D in-depth energy equation with a cell-centred
finite-volume method (FVM) over an area function `A(y)` (see
[`../reference/01_yaml_case_deck.md`](../reference/01_yaml_case_deck.md) for
the three supported geometries), and solves it with semi-implicit Backward
Euler. This page derives the tridiagonal system assembled by
`scam/numerics/assembly.py`; the solver-loop sequencing that calls it is
described in `CLAUDE.md`'s Architecture section.

## Energy equation

For node $n$:

$$
\begin{aligned}
\frac{d}{dt}\left[\rho_n\,c_{p,n}\,T_n\,\mathcal{A}_n\,\Delta_n\right]
 &= G_{n-1/2}\left(T_{n-1} - T_n\right) - G_{n+1/2}\left(T_n - T_{n+1}\right) \\
 &\quad + Q_{\text{decomp},n}\,\mathcal{A}_n\,\Delta_n \\
 &\quad + Q_{\text{pyro},n}\,\mathcal{A}_n\,\Delta_n
\end{aligned}
$$

$G_{n+1/2}$ is the conductance between nodes $n$ and $n+1$ [W/K], harmonic in
the two nodes' conductivities and areas (`geometry/fvm.py::interface_conductance`,
`face_area`), and includes any inter-layer contact resistance at a stack
boundary.

```{note}
$\mathcal{A}_n$ is the cell **area** and $\Delta_n$ its thickness, so
$\mathcal{A}_n \Delta_n$ is the cell volume. The tridiagonal coefficients
below are separately named $A_n, B_n, C_n, D_n$ — the code writes both the
area and the sub-diagonal coefficient as `A`, so watch the context when
cross-referencing `numerics/assembly.py`.
```

```{note}
This page states the equation with the simpler `rho·cp·T` storage term for
clarity. The actual assembly uses `rho·h(T)` (sensible enthalpy, not `cp·T`)
for the storage RHS — see
[`04_energy_formulation_comparison.md`](04_energy_formulation_comparison.md)
and the `h_old` / `use_rho_old` discussion there. The `M_n`/`D_n` coefficients
below are written in the `cp·T` form; the actual code additionally carries
the `(ρ_old·h_old − ρ_new·h_k)` density-change term folded into `D_n`.
```

## Semi-implicit Backward Euler

Conduction is fully implicit (`theta = 1`); decomposition and pyrolysis-gas
sources are evaluated explicitly at the start of the step (per the solver
sequence in `solvers/indepth_solver.py::step()`). The per-node coefficients
of the tridiagonal system
$A_n T_{n-1} + B_n T_n + C_n T_{n+1} = D_n$ are:

$$
\begin{aligned}
A_n &= -G_{n-1/2} \\
C_n &= -G_{n+1/2} \\
B_n &= M_n + G_{n-1/2} + G_{n+1/2} \\
D_n &= M_n\,T_n^{\,k}
     + \left(Q_{\text{decomp},n} + Q_{\text{pyro},n}\right)\mathcal{A}_n\,\Delta_n
\end{aligned}
$$

where $M_n$ is the cell's thermal mass divided by the timestep:

$$
M_n = \frac{\rho_n\,c_{p,n}\,\mathcal{A}_n\,\Delta_n}{\Delta t}
$$

**Surface node** (`n = 0`, half-node): no left neighbour (`A_0 = 0`); the
left face instead receives the prescribed conduction flux `q_cond`
[W/m²] (positive into the material): `D_0 += q_cond * A_face_0`.

**Back node** (`n = N-1`, half-node): `adiabatic` back BC needs no extra
term (`C_{N-1} = 0` by convention already); `prescribed_temp` pins
`T_{N-1}` externally after the solve; `prescribed_flux` adds
`D_{N-1} += q_back * A_face_back`.

**Layer interfaces**: adjacent half-nodes in neighbouring layers are coupled
by the same `G_interface` conductance formula, with the layer's
`contact_resistance` folded in.

## The `F_cond` reduction — how the SEB couples to conduction

The surface energy balance (SEB) Newton solve needs `q_cond` as a function
of the still-unknown `T_wall`. Rather than solving the full tridiagonal
system for every SEB Newton iterate, `compute_F_cond` performs a backward
elimination of the assembled system down to node 0, producing the linear
relationship

$$
q_\text{cond} = \alpha_F\,T_w + \beta_F
$$

which is what `solvers/surface_solver.py`'s Newton iteration actually
iterates against — one scalar equation, not a full linear solve, per
iteration. Once `T_wall` is found, `compute_F_cond`'s returned
`TridiagSystem` is reused (with `D[0]` patched using the converged `q_cond`)
for the single final tridiagonal solve of that timestep — see the
"Performance-sensitive assembly path" note in `CLAUDE.md`.

```{danger}
**Sign convention, worth getting right if you ever touch this code.** The
backward elimination uses `D_sweep[n-1] -= factor * D_sweep[n]` where
`factor = C[n-1]/B[n]` is **negative** (since `C` is negative). Using `+=`
instead of `-=` silently subtracts rather than adds and produces a wrong
`beta_F` — this was a critical bug that caused Newton non-convergence,
recorded in `CLAUDE.md`'s "F_cond sign" note. There is no compile-time or
type-level guard against this; it only shows up as a subtly wrong surface
temperature.
```

## Picard iteration on the assembled system

Because `k(T)` and `cp(T)` in the coefficients above are themselves
evaluated at the current temperature iterate, `indepth_solver.py` wraps the
whole assemble-and-solve in a fixed-point (Picard) loop: reassemble with
`cp(T^k)`, re-solve, repeat until `max(|T_new - T_k|) < picard_tol`
(`SolverOptions.max_picard`, `picard_tol` — see
[`03_solver_options.md`](../reference/03_solver_options.md)). This converges
in one or two iterations for constant-`cp` materials and needs more for
strongly temperature-dependent `cp(T)`; `max_picard = 1` reproduces the
old single-pass behaviour with no iteration at all.

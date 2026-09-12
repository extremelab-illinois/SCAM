<!-- SPDX-License-Identifier: MIT -->
# Troubleshooting

For installation and environment problems, see
[`01_installation.md`](01_installation.md#troubleshooting). This page covers
problems that show up once a case is actually running.

## `SCAMNumericsError: Newton iteration did not converge after N iterations`

The surface energy balance's Newton solve failed to bring the SEB residual
below `solver.seb_tol` within `solver.max_seb_iter` iterations. Common
causes, roughly in order of likelihood:

- **A boundary condition that changes too fast for the timestep.** Check
  `dt_max_dT` and `dt_init` against how quickly your `alpha_conv`/`T_aw` or
  `rhoUeCH`/`h_r` tables ramp — a table that jumps from 0 to a large value
  over a very short interval (see the 0.001 s ramps in
  `ablation1_template.yaml`) needs a correspondingly small `dt_init` to
  resolve.
- **An out-of-range B′ table lookup silently returning nonsense chemistry**
  at the clamped edge — see the clamping warning in
  [`08_surface_chemistry_backends.md`](08_surface_chemistry_backends.md).
  This doesn't always show up as a Newton failure, but it's a common enough
  root cause to check first when convergence gets worse partway through a
  run.
- **A `radiation` back BC with both `emissivity_back` and `h_back` left at
  zero** would normally be caught as a load error (see
  [`01_yaml_case_deck.md`](../reference/01_yaml_case_deck.md)) rather than a
  convergence failure — but a *very* small nonzero value can produce a
  nearly-adiabatic back face that still leaves the SEB poorly conditioned.
- Try raising `max_seb_iter` and loosening `seb_tol` temporarily to see
  whether the run merely converges slowly rather than diverging — that
  distinguishes a timestep problem from a genuinely inconsistent BC.

## Picard iteration not converging within `max_picard`

The in-depth tridiagonal solve is wrapped in a fixed-point (Picard) loop
that re-assembles with `cp(T^k)` at the current iterate; it should converge
in 1–2 iterations for constant-`cp` materials and more for strong `cp(T)`
variation. If it's pinned at `max_picard` every step:

- Check whether `dt_max_dT`/`dt_max_drho_frac` are too loose for the
  material's actual `cp(T)` curvature — Picard non-convergence at the
  default `max_picard=8` usually means the timestep, not the iteration
  count, needs tightening.
- `max_picard=1` reproduces the old single-pass (no-iteration) behaviour,
  which is useful for isolating whether a discrepancy against a reference
  is a Picard-convergence issue or something else entirely.

## Results look physically wrong (density increasing, runaway recession, …)

- **Density should never increase.** `ρ_new = max(min(ρ_new, ρ_old),
  ρ_residual)` is enforced unconditionally after every Arrhenius step; if
  you see density rising anywhere in a profile plot, that is a bug to
  report, not a parameter to tune around.
- **A `T_wall` sawtooth of a few kelvin** during recession is expected
  under the default Lagrangian scheme (`continuous_remap: false`) — each
  discrete node drop exposes the next, slightly cooler node. It is not a
  numerical error. Set `continuous_remap: true` for a smooth `T_wall` at
  the cost of the ALE recession-CFL limit (`dt ≤ 0.1·h/s_dot`, applied
  automatically).
- **Recession much faster or slower than expected** — check `C_M` and
  `rho_e_u_e` together: `m_dot_char = rho_e_u_e · C_M · B'_c`, so a factor-10
  error in either scales recession by the same factor. Also check the B′
  table's pressure-axis coverage (see
  [`08_surface_chemistry_backends.md`](08_surface_chemistry_backends.md)) —
  a clamped lookup at the wrong pressure gives a plausible-looking but wrong
  `B'_c`.

## A run is much slower than expected

- `bprime_runtime: true` (live Cantera) costs roughly 10–100 ms per SEB
  Newton call, versus a fraction of a millisecond for the pre-computed
  `BPrimeTable`. This is expected, not a bug — use it for reference runs
  only.
- `element_transport: true` adds a per-timestep PDE solve for the four
  elemental fields; combined with `bprime_runtime`, expect the slowest
  runs in the codebase.

## Getting more diagnostic detail

Run with `--verbose` (the CLI default) rather than `--quiet`, and re-raise
exceptions are not swallowed — a simulation exception prints
`ERROR during simulation:` and then re-raises, so the full Python traceback
is visible; read it from the bottom up as usual.

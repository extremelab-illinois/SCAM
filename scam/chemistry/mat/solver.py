"""Robust Newton solver infrastructure for surface equilibrium chemistry (Phase 9).

Provides:

* ``NewtonSolverConfig`` — solver tuning knobs bundled in one dataclass.
* ``SolverDiagnostics`` — per-grid-point convergence metadata.
* ``damped_equilibrate`` — wraps Cantera equilibrate() with damped-initial-
  moles retry: when gibbs→vcs fallback still fails, the condensed-phase
  initial moles are scaled by ``damp_factor`` up to ``max_retries`` times.
  This often rescues ill-conditioned near-depleted-condensed cases.
* ``sort_grid_for_continuation`` — reorders a flat list of (T, p, B'g) dicts
  so that grid points are visited in continuation order (outer loop: pressure,
  middle loop: B'g, inner loop: temperature ascending).  Physically similar
  states are then neighbours, making implicit warm-start effective.
* ``ContinuationSolver`` — thin wrapper around a serial grid sweep that
  enforces continuation ordering and collects per-point diagnostics.

Design intent
-------------
Cantera's ``mix.equilibrate("TP")`` is the inner Newton solver; the chemistry module does not
replace it.  Phase 9 improves the *outer* robustness:

1. Damped retry on initial condensed moles when Cantera's internal solver
   fails (pathological near-zero condensed phase, extreme T or p).
2. Sorted grid traversal so the implicit warm-start (reusing the last
   equilibrated Solution as the starting state) is physically helpful.
3. Diagnostics (converged, solver_used, n_attempts, elapsed time) stored
   alongside every ``BPrimeCase`` so convergence issues are auditable.

All quantities are in SI units (T in K, P in Pa, time in seconds).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NewtonSolverConfig:
    """Solver tuning parameters.

    Attributes
    ----------
    max_steps:
        Maximum inner-solver steps passed to Cantera's ``equilibrate``.
    max_iter:
        Maximum outer iterations passed to Cantera's ``equilibrate``.
    log_level:
        Cantera verbosity (0 = silent).
    damp_factor:
        Factor applied to initial condensed-phase moles on each damped
        retry (0 < damp_factor < 1).  A value of 0.5 halves the moles
        on each attempt.
    max_retries:
        Maximum number of damped retries after all solver variants have
        failed.  Each retry multiplies the initial condensed moles by
        ``damp_factor``.
    species_cutoff:
        Mole fractions below this threshold are treated as numerically
        zero when evaluating species-presence checks.
    use_continuation:
        When ``True`` (default), ``ContinuationSolver`` sorts grid points
        in continuation order.  Set ``False`` to disable.
    """

    max_steps: int = 5000
    max_iter: int = 200
    log_level: int = 0
    damp_factor: float = 0.5
    max_retries: int = 3
    species_cutoff: float = 1e-20
    use_continuation: bool = True

    def __post_init__(self) -> None:
        if not (0.0 < self.damp_factor < 1.0):
            raise ValueError(
                f"damp_factor must be in (0, 1), got {self.damp_factor}."
            )
        if self.max_retries < 0:
            raise ValueError(
                f"max_retries must be non-negative, got {self.max_retries}."
            )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SolverDiagnostics:
    """Convergence metadata for one grid point.

    Attributes
    ----------
    converged:
        ``True`` if Cantera's equilibrate converged on the first or a
        subsequent attempt.
    solver_used:
        Name of the Cantera solver that converged (``"gibbs"``, ``"vcs"``),
        or ``"failed"`` if all attempts failed.
    n_attempts:
        Total number of equilibrate() calls made (1 = converged first try).
    elapsed_s:
        Wall-clock time for this grid point (seconds).
    warm_started:
        ``True`` when the Solution objects carried state from the previous
        grid point (i.e. continuation was active).
    initial_condensed_moles:
        Initial condensed-phase moles used in the successful attempt.
        Equals the configured value when converged on the first try.
    """

    converged: bool
    solver_used: str
    n_attempts: int
    elapsed_s: float
    warm_started: bool = False
    initial_condensed_moles: float = float("nan")


# ---------------------------------------------------------------------------
# Damped-retry equilibration helper
# ---------------------------------------------------------------------------


def damped_equilibrate(
    *,
    build_fn,
    run_fn,
    solvers: list[str],
    initial_condensed_moles: float,
    config: NewtonSolverConfig,
) -> tuple[Any, float, str, int]:
    """Equilibrate with damped-moles retry on failure.

    Tries each solver in *solvers* first.  If all fail, scales
    ``initial_condensed_moles`` by ``config.damp_factor`` and retries from
    the beginning of the solver list, up to ``config.max_retries`` times.

    Parameters
    ----------
    build_fn:
        Callable ``(initial_condensed_moles) -> (wall_gas, mix)`` that
        constructs a fresh Cantera Mixture with the given condensed moles.
    run_fn:
        Callable ``(mix, solver_name) -> None`` that calls
        ``mix.equilibrate("TP", solver=solver_name, ...)``.
    solvers:
        Ordered list of solver names to try (e.g. ``["gibbs", "vcs"]``).
    initial_condensed_moles:
        Starting condensed-phase moles (will be damped on retry).
    config:
        ``NewtonSolverConfig`` instance with retry parameters.

    Returns
    -------
    wall_gas : ct.Solution
        Converged gas-phase Solution.
    converged_moles : float
        The initial condensed moles used in the successful attempt.
    solver_used : str
        Name of the solver that succeeded.
    n_attempts : int
        Total number of ``equilibrate`` calls made.
    """
    n_attempts = 0
    current_moles = float(initial_condensed_moles)
    last_exc: Exception | None = None

    for retry in range(config.max_retries + 1):
        for solver_name in solvers:
            n_attempts += 1
            wall_gas, mix = build_fn(current_moles)
            try:
                run_fn(mix, solver_name)
                return wall_gas, current_moles, solver_name, n_attempts
            except Exception as exc:
                last_exc = exc

        # All solvers failed at this moles level → damp and retry
        current_moles *= config.damp_factor
        if current_moles < 1e-12:
            break

    raise RuntimeError(
        f"Equilibrium did not converge after {n_attempts} attempts "
        f"(all solvers and {config.max_retries} damped retries failed)."
    ) from last_exc


# ---------------------------------------------------------------------------
# Grid continuation ordering
# ---------------------------------------------------------------------------


def sort_grid_for_continuation(
    grid_kwargs: list[dict],
    *,
    outer: str = "pressure_Pa",
    middle: str = "bg",
    inner: str = "temperature_K",
    inner_ascending: bool = True,
) -> list[dict]:
    """Sort grid-point dicts for physically meaningful continuation traversal.

    Default order: outer loop = pressure, middle loop = B'g,
    inner loop = temperature ascending.  Physically similar points are
    adjacent, so the implicit warm-start from reusing Cantera Solution
    objects is maximally effective.

    Parameters
    ----------
    grid_kwargs:
        Flat list of per-point keyword dicts (as produced by
        ``build_table``).
    outer, middle, inner:
        Keys to use for the three sort levels.
    inner_ascending:
        Sort direction for the inner loop.  ``True`` (default) = low T
        first (cold start → hot), which generally converges more reliably
        because B' increases monotonically and each solution is close to the
        next.

    Returns
    -------
    list[dict]
        New list in continuation order.  Original list is not mutated.
    """
    return sorted(
        grid_kwargs,
        key=lambda kw: (
            kw.get(outer, 0.0),
            kw.get(middle, 0.0),
            kw.get(inner, 0.0) if inner_ascending else -kw.get(inner, 0.0),
        ),
    )


# ---------------------------------------------------------------------------
# Continuation solver
# ---------------------------------------------------------------------------


class ContinuationSolver:
    """Serial grid solver with continuation ordering and per-point diagnostics.

    Parameters
    ----------
    config:
        ``NewtonSolverConfig`` instance.
    compute_fn:
        Callable ``(**kwargs) -> BPrimeCase`` — typically
        ``scam.chemistry.thermochemistry.compute_bprime_case``.

    Usage
    -----
    ::

        solver = ContinuationSolver(config, compute_bprime_case)
        cases, diagnostics = solver.run(grid_kwargs, gas=gas, carbon=carbon)
    """

    def __init__(self, config: NewtonSolverConfig, compute_fn) -> None:
        self.config = config
        self._compute_fn = compute_fn

    def run(
        self,
        grid_kwargs: list[dict],
        **shared_kwargs,
    ) -> tuple[list, list[SolverDiagnostics]]:
        """Run the grid sweep with continuation.

        Parameters
        ----------
        grid_kwargs:
            Per-point dicts (temperature_K, pressure_Pa, bg, etc.).
        **shared_kwargs:
            Shared keyword arguments passed to every compute call
            (e.g. ``gas=gas, carbon=carbon``).

        Returns
        -------
        cases : list[BPrimeCase]
            Results in the same order as the (sorted) grid.
        diagnostics : list[SolverDiagnostics]
            One ``SolverDiagnostics`` entry per grid point, aligned with
            ``cases``.
        """
        if self.config.use_continuation:
            ordered = sort_grid_for_continuation(grid_kwargs)
        else:
            ordered = list(grid_kwargs)

        cases = []
        diags: list[SolverDiagnostics] = []
        first_point = True

        for kw in ordered:
            t0 = time.perf_counter()
            try:
                case = self._compute_fn(**kw, **shared_kwargs)
                elapsed = time.perf_counter() - t0
                diag = SolverDiagnostics(
                    converged=case.converged,
                    solver_used=case.solver_used,
                    n_attempts=case.n_solver_attempts,
                    elapsed_s=elapsed,
                    warm_started=not first_point,
                    initial_condensed_moles=float("nan"),
                )
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                diag = SolverDiagnostics(
                    converged=False,
                    solver_used="failed",
                    n_attempts=1,
                    elapsed_s=elapsed,
                    warm_started=not first_point,
                )
                # Re-raise so caller can decide how to handle
                raise exc from exc

            cases.append(case)
            diags.append(diag)
            first_point = False

        return cases, diags

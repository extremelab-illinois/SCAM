# SPDX-License-Identifier: MIT
"""Surface energy balance (SEB) Newton solver.

Finds T_wall such that the SEB residual f(T_wall) = 0 using a finite-
difference Newton iteration (from numerics/nonlinear.py).

For prescribed-temperature or prescribed-flux modes the Newton solve is
bypassed and T_wall / q_cond are set directly.
"""

from __future__ import annotations

from scam.config.boundary import SurfaceBCConfig, SurfaceBCType, eval_bc
from scam.config.solver import SolverOptions
from scam.core.state import SimState
from scam.numerics.nonlinear import newton_scalar, bisect
from scam.physics.surface_energy import seb_residual


def solve_surface(
    state: SimState,
    alpha_F: float,
    beta_F: float,
    bc: SurfaceBCConfig,
    time: float,
    m_dot_pyro: float,
    mat_surface,       # MaterialCard for the surface layer
    b_prime_table,     # BPrimeTable | BprimeEvaluator | None
    options: SolverOptions,
    emissivity_override: float = -1.0,
    Z_C_pyro: float | None = None,
) -> tuple[float, float, float, float]:
    """Solve for the equilibrium surface temperature.

    Parameters
    ----------
    state:
        Current simulation state (provides initial T_wall guess).
    alpha_F, beta_F:
        F_cond coefficients: q_cond = alpha_F * T_w + beta_F.
    bc:
        Surface boundary condition.
    time:
        Current simulation time [s].
    m_dot_pyro:
        Pyrolysis gas mass flux at surface [kg/m^2/s].
    mat_surface:
        MaterialCard for the surface layer.
    b_prime_table:
        BPrimeTable or None.
    options:
        Solver options (tolerances, iteration limits).
    emissivity_override:
        Override emissivity if > 0.

    Returns
    -------
    (T_wall, q_cond, m_dot_char, h_wall)
    """
    rho_surface = float(state.rho[0])

    if bc.bc_type == SurfaceBCType.PRESCRIBED_TEMP:
        T_w = eval_bc(bc.T_prescribed, time)
        _, q_cond, m_dot_char, h_wall = seb_residual(
            T_w, alpha_F, beta_F, bc, time, m_dot_pyro,
            mat_surface, b_prime_table, emissivity_override, Z_C_pyro,
            rho_surface=rho_surface,
        )
        return float(T_w), q_cond, m_dot_char, h_wall

    if bc.bc_type == SurfaceBCType.PRESCRIBED_FLUX:
        q_prescribed = eval_bc(bc.q_prescribed, time) if bc.q_prescribed else 0.0
        T_w_guess = state.T_wall
        if abs(alpha_F) > 0:
            T_w = (q_prescribed - beta_F) / alpha_F
        else:
            T_w = T_w_guess
        return float(T_w), float(q_prescribed), 0.0, 0.0

    # ENERGY_BALANCE mode: Newton on T_wall.  Cache the most recent complete
    # residual breakdown: nonlinear solvers normally finish immediately after
    # evaluating the converged T_w, and the caller needs the same q_cond,
    # m_dot_char, and h_wall values.  Reusing that tuple avoids one duplicate
    # live-chemistry sequence per Picard iteration.
    last_T_w: float | None = None
    last_result: tuple[float, float, float, float] | None = None

    def evaluate(T_w: float) -> tuple[float, float, float, float]:
        nonlocal last_T_w, last_result
        T_w = float(T_w)
        if last_T_w == T_w and last_result is not None:
            return last_result
        last_result = seb_residual(
            T_w, alpha_F, beta_F, bc, time, m_dot_pyro,
            mat_surface, b_prime_table, emissivity_override, Z_C_pyro,
            rho_surface=rho_surface,
        )
        last_T_w = T_w
        return last_result

    def residual(T_w: float) -> float:
        return evaluate(T_w)[0]

    T_w_guess = state.T_wall
    from scam.core.errors import SCAMNumericsError as _SCAMNumericsError
    try:
        T_w = newton_scalar(
            residual,
            x0=T_w_guess,
            tol=options.seb_tol,
            max_iter=options.max_seb_iter,
            dx_fd=5.0,
            x_min=200.0,
            x_max=20000.0,
        )
    except _SCAMNumericsError:
        # Newton failed (likely due to a gradient kink in the B' table's
        # LinearNDInterpolator).  Find a bracket by expanding outward in both
        # directions from the last guess, then bisect.
        T_c = T_w_guess
        f_c = residual(T_c)
        step = max(50.0, abs(T_c) * 0.05)
        T_lo, T_hi = T_c, T_c
        f_lo, f_hi = f_c, f_c
        for _ in range(40):
            if T_lo > 200.0:
                T_lo = max(200.0, T_lo - step)
                f_lo = residual(T_lo)
                if f_lo * f_hi <= 0:
                    break
            if T_hi < 20000.0:
                T_hi = min(20000.0, T_hi + step)
                f_hi = residual(T_hi)
                if f_lo * f_hi <= 0:
                    break
            step *= 1.5
        T_w = bisect(residual, T_lo, T_hi, tol=options.seb_tol)

    _, q_cond, m_dot_char, h_wall = evaluate(T_w)
    return float(T_w), float(q_cond), float(m_dot_char), float(h_wall)

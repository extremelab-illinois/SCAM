# SPDX-License-Identifier: MIT
"""Adaptive time-step control.

The timestep is adapted based on:
1. Maximum absolute temperature change per step across all nodes.
2. Maximum fractional density change per step across all nodelets.
3. Hard limits dt_min and dt_max from SolverOptions.

The adaptation rule is:

    If ``|delta T_max|`` > dt_max_dT: halve dt
    If ``|delta rho / rho_max|`` > dt_max_drho: halve dt
    If both are well within limits: grow dt by 20% (up to dt_max)
"""

from __future__ import annotations

import numpy as np

from scam.config.solver import SolverOptions
from scam.core.state import SimState


def adapt_timestep(
    state_old: SimState,
    state_new: SimState,
    options: SolverOptions,
) -> float:
    """Compute the next timestep based on changes in the current step.

    Parameters
    ----------
    state_old:
        State at the beginning of the completed step.
    state_new:
        State at the end of the completed step.
    options:
        Solver options containing dt_min, dt_max, dt_max_dT, dt_max_drho_frac.

    Returns
    -------
    float
        Recommended dt for the next step [s].
    """
    dt_cur = state_new.dt

    # Temperature criterion — clip to common length (node drop changes N)
    n_common = min(len(state_new.T), len(state_old.T))
    dT_max = float(np.max(np.abs(state_new.T[:n_common] - state_old.T[:n_common])))

    # Density criterion (check rho_components if present)
    drho_frac_max = 0.0
    for l_idx, rho_comp in enumerate(state_new.rho_components):
        if rho_comp is None:
            continue
        if l_idx >= len(state_old.rho_components):
            continue
        rho_old_comp = state_old.rho_components[l_idx]
        if rho_old_comp is None:
            continue
        # Shapes may differ after node drop — compare common prefix
        n_n = rho_comp.shape[1] if rho_comp.ndim > 1 else len(rho_comp)
        n_o = rho_old_comp.shape[1] if rho_old_comp.ndim > 1 else len(rho_old_comp)
        n_c = min(n_n, n_o)
        if rho_comp.ndim > 1:
            rc_new = rho_comp[:, :n_c, :]
            rc_old = rho_old_comp[:, :n_c, :]
        else:
            rc_new = rho_comp[:n_c]
            rc_old = rho_old_comp[:n_c]
        rho_ref = np.where(rc_old > 1.0, rc_old, 1.0)
        frac = np.max(np.abs(rc_new - rc_old) / rho_ref)
        drho_frac_max = max(drho_frac_max, float(frac))

    # Smooth proportional controller.  The old bang-bang rule (halve when over a
    # limit, ×1.2 when well under, hold otherwise) makes dt oscillate between
    # regimes; with backward-Euler that step-to-step dt jitter shows up as a
    # few-K wobble in T_wall.  Instead, scale dt to keep the worst-case
    # normalised change near a target fraction (0.9) of its limit, and cap the
    # per-step change rate (×0.5 .. ×1.2) so dt evolves smoothly.
    err_T   = dT_max / options.dt_max_dT if options.dt_max_dT > 0 else 0.0
    err_rho = (drho_frac_max / options.dt_max_drho_frac
               if options.dt_max_drho_frac > 0 else 0.0)
    err = max(err_T, err_rho)

    if err <= 1e-12:
        factor = 1.2
    else:
        factor = 0.9 / err
    factor = float(np.clip(factor, 0.5, 1.2))
    dt_next = dt_cur * factor

    # Recession CFL limit (continuous moving-mesh only).  With the ALE scheme,
    # the surface layer is re-interpolated each step; if the surface moves more
    # than a fraction of a cell per step, linear interpolation of the steep
    # near-surface front injects a few-K wobble into T_wall.  Cap dt so the
    # surface advances < CFL_RECESSION cells per step, which keeps T_wall smooth.
    if options.continuous_remap and state_new.s_dot > 0.0:
        lo, hi = state_new.mesh.layer_boundaries[0]
        dlt = state_new.mesh.delta_nodes[lo:hi]
        dlt = dlt[dlt > 0]
        if dlt.size:
            h_surf = float(np.min(dlt))
            CFL_RECESSION = 0.1
            dt_cfl = CFL_RECESSION * h_surf / state_new.s_dot
            dt_next = min(dt_next, dt_cfl)

    dt_next = float(np.clip(dt_next, options.dt_min, options.dt_max))
    return dt_next


def is_output_step(time: float, dt: float, output_dt: float) -> bool:
    """Return True if an output snapshot should be saved after this step.

    Uses a tolerance of dt/10 to avoid floating-point misses.
    """
    tol = dt * 0.1
    next_output = np.round(time / output_dt) * output_dt
    return abs(time - next_output) < tol or (time - next_output) > -tol

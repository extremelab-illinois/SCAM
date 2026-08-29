# SPDX-License-Identifier: MIT
"""Post-hoc mass and energy conservation diagnostics.

Usage::

    from scam.diagnostics import conservation_check

    results = run(stack, mat_cards, ...)
    mat_list = [mat_cards[l.material_name] for l in stack.layers]
    diag = conservation_check(results, mat_list)

    # diag['mass_residual']  — should be ≈ 0  (exact by construction for charring)
    # diag['energy_source']  — net internal decomposition + gas-advection energy [J]
    #                          ≈ 0 for inert materials, nonzero for charring

Conservation equations checked
-------------------------------

**Mass** (exact):
    d(∫ρ dV)/dt = -(m_dot_char + m_dot_pyro) * A_surface
    residual     = Δ(∫ρ dV) + Δ(cum_mass_out)  →  should be ≈ 0

**Energy** (surface conduction only):
    d(∫ρh dV)/dt = q_cond * A_surface    (adiabatic back face)
                 + internal energy sources  (decomposition, gas advection)
    energy_source = Δ(∫ρh dV) - Δ(cum_energy_in)
                  = net internal source    ≈ 0 for inert; nonzero for charring

``q_cond`` is the net thermal conduction flux delivered to the solid by the
surface energy balance — it does NOT include advective enthalpy carried by mass
leaving or entering.  For charring materials, ``energy_source`` captures the
net chemical decomposition heat and pyrolysis gas advection contribution.

Back-face note
--------------
``cum_energy_in`` tracks only the surface-side q_cond.  For non-adiabatic back
faces (PRESCRIBED_FLUX), pass ``q_back_cumulative`` (see below) to include it.
For PRESCRIBED_TEMP back faces, the back-face flux is not currently tracked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scam.physics.properties import mixture_enthalpy_array

if TYPE_CHECKING:
    from scam.core.results import Results
    from scam.core.state import SimState


def slab_mass(state: "SimState") -> float:
    """Solid mass in the domain: ∫ρ dV  [kg].

    For SLAB geometry with unit area this equals ∫ρ dy [kg/m²].
    For non-unit areas (cylindrical) the true volume is used.
    """
    return float(np.sum(state.rho * state.mesh.area_nodes * state.mesh.delta_nodes))


def slab_energy(state: "SimState", mat_list: list) -> float:
    """Solid sensible enthalpy: ∫ρ·h(T,ρ) dV  [J].

    Uses the same ``mixture_enthalpy_array`` as the FVM energy storage term,
    so the result is directly comparable to the assembled RHS.
    """
    h = mixture_enthalpy_array(mat_list, state.T, state.rho, state.mesh.layer_id)
    return float(np.sum(state.rho * h * state.mesh.area_nodes * state.mesh.delta_nodes))


def conservation_check(
    results: "Results",
    mat_list: list,
    q_back_cumulative: np.ndarray | None = None,
) -> dict:
    """Compute mass and energy conservation diagnostics from a Results object.

    Parameters
    ----------
    results:
        Output of ``run()``.
    mat_list:
        Ordered list of MaterialCard objects (same order as stack.layers).
    q_back_cumulative:
        Optional array shape (n_snapshots,) with cumulative back-face energy
        input [J] at each snapshot, for non-adiabatic back faces.  If None,
        back-face contribution is assumed zero (adiabatic).

    Returns
    -------
    dict with keys:

    - ``times``              — snapshot times [s], shape (N,)
    - ``mass``               — ∫ρ dV at each snapshot [kg], shape (N,)
    - ``cum_mass_exact``     — exact domain mass loss: ∑Δ(ρ·V) per step [kg], shape (N,)
                               Trivially exact; mass + cum_mass_exact == mass[0] always.
    - ``cum_mass_model``     — CMA model mass-out: ∑(m_dot_char+m_dot_pyro)·A·dt [kg], shape (N,)
    - ``mass_cma_residual``  — cum_mass_exact − cum_mass_model [kg]: discrepancy between the
                               exact domain mass loss and the CMA surface-flux model.
                               Non-zero when rho_wall ≠ rho_char (CMA approximation error).
    - ``energy``             — ∫ρh dV at each snapshot [J], shape (N,)
    - ``cum_energy_in``      — cumulative q_cond·A·dt through surface [J], shape (N,)
    - ``dE_stored``          — Δ(∫ρh dV) since t=0 [J], shape (N,)
    - ``energy_source``      — dE_stored − cum_energy_in [J], shape (N,):
                               net internal source ≈ 0 for inert; decomposition + gas advection
                               contribution for charring materials.
    """
    snaps = results.snapshots

    masses   = np.array([slab_mass(s)             for s in snaps])
    energies = np.array([slab_energy(s, mat_list) for s in snaps])
    times    = np.array([s.time                   for s in snaps])

    # --- Prefer exact per-step cumulative trackers; fall back to trapezoidal ---
    has_exact = hasattr(snaps[0], "cum_mass_exact")
    has_model = hasattr(snaps[0], "cum_mass_out")
    has_energy = hasattr(snaps[0], "cum_energy_in")

    if has_exact:
        cum_mass_exact = np.array([s.cum_mass_exact for s in snaps])
    else:
        # Exact domain mass loss is always computable from the mass field itself
        cum_mass_exact = masses[0] - masses

    if has_model:
        cum_mass_model = np.array([s.cum_mass_out for s in snaps])
    else:
        # Trapezoidal fallback from snapshot values
        A_surf    = np.array([s.mesh.area_nodes[0]        for s in snaps])
        m_dot_tot = np.array([s.m_dot_char + s.m_dot_pyro for s in snaps])
        dt        = np.diff(times)
        avg_m     = 0.5 * (m_dot_tot[:-1] + m_dot_tot[1:])
        avg_A     = 0.5 * (A_surf[:-1]    + A_surf[1:])
        cum_mass_model = np.concatenate([[0.0], np.cumsum(avg_m * avg_A * dt)])

    if has_energy:
        cum_energy_in = np.array([s.cum_energy_in for s in snaps])
    else:
        A_surf     = np.array([s.mesh.area_nodes[0] for s in snaps])
        q_cond_arr = np.array([s.q_cond             for s in snaps])
        dt         = np.diff(times)
        avg_q      = 0.5 * (q_cond_arr[:-1] + q_cond_arr[1:])
        avg_A      = 0.5 * (A_surf[:-1]     + A_surf[1:])
        cum_energy_in = np.concatenate([[0.0], np.cumsum(avg_q * avg_A * dt)])

    if q_back_cumulative is not None:
        cum_energy_in = cum_energy_in + np.asarray(q_back_cumulative)

    # CMA residual: exact domain loss vs surface-flux model
    mass_cma_residual = cum_mass_exact - cum_mass_model

    # Energy
    dE_stored     = energies - energies[0]
    energy_source = dE_stored - cum_energy_in

    return {
        "times":             times,
        "mass":              masses,
        "cum_mass_exact":    cum_mass_exact,
        "cum_mass_model":    cum_mass_model,
        "mass_cma_residual": mass_cma_residual,
        "energy":            energies,
        "cum_energy_in":     cum_energy_in,
        "dE_stored":         dE_stored,
        "energy_source":     energy_source,
    }

# SPDX-License-Identifier: MIT
"""Top-level material response driver.

The main time loop:
    1. Solve one in-depth timestep (mass + energy equations).
    2. Apply surface recession (advance s_total by s_dot * dt).
    3. Check for node-drop trigger; apply drop_and_merge if needed.
    4. Adapt timestep for the next step.
    5. Save results at output intervals.
    6. Repeat until t_end.

Usage (Python API)::

    from scam.solvers.material_response import run
    from scam.config.stack import StackConfig, LayerConfig
    from scam.config.geometry import GeometryConfig
    from scam.config.boundary import SurfaceBCConfig, BackBCConfig
    from scam.config.solver import SolverOptions
    from scam.io.material_loader import load_materials

    cards, bpts = load_materials(["scam/materials/ablative_organic/tacot_v3.0.yaml"])
    stack  = StackConfig([LayerConfig("TACOT", 0.05, 50, 4)])
    geom   = GeometryConfig()
    sbc    = SurfaceBCConfig(alpha_conv=5000., T_aw=8000., ...)
    bbc    = BackBCConfig()
    opts   = SolverOptions(t_end=120., dt_init=0.01, output_dt=1.)

    results = run(stack, cards, bpts, geom, sbc, bbc, opts)
"""

from __future__ import annotations

import copy

import numpy as np

from scam.config.boundary import SurfaceBCConfig, BackBCConfig, SurfaceBCType, eval_bc
from scam.config.geometry import GeometryConfig
from scam.config.stack import StackConfig
from scam.config.solver import SolverOptions
from scam.core.results import Results
from scam.core.state import SimState
from scam.mesh.grid import build_mesh, nominal_h
from scam.mesh.receding import apply_recession, apply_recession_ale, needs_drop
from scam.mesh.remap import drop_and_merge
from scam.numerics.time_integration import adapt_timestep
from scam.solvers.indepth_solver import step


def _interp_tc(y_nodes: np.ndarray, T: np.ndarray, tc_positions: list) -> np.ndarray:
    """Interpolate temperatures at thermocouple probe positions."""
    if not tc_positions:
        return np.array([])
    return np.interp(tc_positions, y_nodes, T)


def run(
    stack: StackConfig,
    mat_cards: dict,        # {name: MaterialCard}
    b_prime_tables: dict,   # {name: BPrimeTable | None}
    geom: GeometryConfig,
    surface_bc: SurfaceBCConfig,
    back_bc: BackBCConfig,
    options: SolverOptions,
    initial_T: float = 300.0,
    verbose: bool = True,
) -> Results:
    """Run the full material response simulation.

    Parameters
    ----------
    stack:
        Multi-layer stack configuration.
    mat_cards:
        Dict of MaterialCard objects keyed by name.
    b_prime_tables:
        Dict of BPrimeTable (or None) keyed by material name.
    geom:
        Geometry configuration.
    surface_bc:
        Surface boundary condition.
    back_bc:
        Back-face boundary condition.
    options:
        Solver options (time range, timestep, convergence, output).
    initial_T:
        Uniform initial temperature [K].
    verbose:
        Print progress information if True.

    Returns
    -------
    Results
        Time-history of temperatures, recession, surface state, and TC probes.
    """
    # ------------------------------------------------------------------
    # Build ordered list of material cards and B' tables by layer index
    # ------------------------------------------------------------------
    mat_list = [mat_cards[lyr.material_name] for lyr in stack.layers]
    bpt_list = [b_prime_tables.get(lyr.material_name) for lyr in stack.layers]

    # ------------------------------------------------------------------
    # Initialise mesh and state
    # ------------------------------------------------------------------
    mesh, state = build_mesh(
        stack, geom,
        s_total=0.0,
        T_init=initial_T,
        mat_registry=mat_cards,
    )
    state.time = options.t_start
    state.dt   = options.dt_init

    # Initialise element transport field
    if options.element_transport:
        from scam.physics.element_transport import initial_Z_elem
        # Use per-material initial composition if provided (e.g. PATO's Zx),
        # otherwise default to ambient air.
        init_z = mat_list[0].initial_Z_elem if mat_list[0].initial_Z_elem is not None else None
        state.Z_elem = initial_Z_elem(mesh.n_nodes_total, z0=init_z)

    results = Results(tc_positions=list(options.tc_positions))

    # ------------------------------------------------------------------
    # Save initial snapshot
    # ------------------------------------------------------------------
    tc_temps = _interp_tc(mesh.y_nodes, state.T, options.tc_positions)
    results.append(state, tc_temps if len(tc_temps) > 0 else None)

    t = options.t_start
    dt = options.dt_init
    next_output_t = options.t_start + options.output_dt
    n_steps = 0
    n_drops  = 0

    if verbose:
        print(f"SCAM: starting run from t={options.t_start:.3f} to t={options.t_end:.3f} s")
        print(f"  Nodes: {mesh.n_nodes_total}, Layers: {len(stack.layers)}")

    # ------------------------------------------------------------------
    # Main time loop
    # ------------------------------------------------------------------
    while t < options.t_end:
        dt = min(dt, options.t_end - t)
        dt = max(dt, options.dt_min)

        state_old = copy.copy(state)

        # 1. In-depth step: mass + energy
        state = step(
            state,
            mat_list, bpt_list,
            stack, surface_bc, back_bc,
            options, t, dt,
        )

        # 2. Apply surface recession
        if options.s_dot_prescribed is not None:
            state.s_dot = options.s_dot_prescribed

        if state.s_dot > 0.0 and options.allow_recession:
            if options.continuous_remap:
                # Continuous moving-mesh (ALE): redistribute + re-interpolate the
                # surface layer every step; no discrete node drops (smooth T_wall).
                state = apply_recession_ale(
                    state, state.s_dot, dt, geom, mat_surface=mat_list[0],
                )
                # ALE re-interpolation shifts T[0] away from the prescribed value;
                # re-pin immediately so the next step's storage term (h_old) is exact.
                if surface_bc.bc_type == SurfaceBCType.PRESCRIBED_TEMP and surface_bc.T_prescribed is not None:
                    t_pin = float(eval_bc(surface_bc.T_prescribed, t + dt))
                    state.T[0] = t_pin
                    state.T_wall = t_pin
            else:
                state = apply_recession(state, state.s_dot, dt, geom)

        # 3. Node-drop check (Lagrangian scheme only; ALE keeps node count fixed)
        if (not options.continuous_remap and state.s_dot > 0.0
                and options.allow_recession and needs_drop(state, stack, options)):
            state = drop_and_merge(state, mat_cards)
            n_drops += 1
            if verbose:
                print(f"  Node drop #{n_drops} at t={t+dt:.4f} s, "
                      f"s_total={state.mesh.s_total*1000:.3f} mm, "
                      f"N={state.mesh.n_nodes_total}")

        # Accumulate conservation trackers (every step, not just output steps).
        # Mass: compare CMA surface-flux model against exact domain mass loss.
        #   cum_mass_out  — CMA model mass leaving: (m_dot_char + m_dot_pyro)*A*dt.
        #                   Agrees with domain mass loss only when rho_wall == rho_char.
        #                   Residual mass_conserved - mass[0] reveals CMA inconsistency.
        #   cum_mass_exact — actual domain mass loss computed directly from ∑ρ·V change;
        #                   trivially exact; mass + cum_mass_exact == mass[0] by construction.
        A_surf = state.mesh.area_nodes[0]
        state.cum_mass_out  = (state_old.cum_mass_out
                               + (state.m_dot_char + state.m_dot_pyro) * A_surf * dt)
        # Exact mass loss: domain integral before minus after (includes node drops & recession)
        _m_old = float(np.sum(state_old.rho * state_old.mesh.area_nodes * state_old.mesh.delta_nodes))
        _m_new = float(np.sum(state.rho  * state.mesh.area_nodes  * state.mesh.delta_nodes))
        state.cum_mass_exact = state_old.cum_mass_exact + (_m_old - _m_new)
        state.cum_energy_in = state_old.cum_energy_in + state.q_cond * A_surf * dt

        t += dt
        state.time = t
        n_steps += 1

        # 4. Adapt timestep
        dt = adapt_timestep(state_old, state, options)

        # 5. Output
        if t >= next_output_t - 1e-10 * options.output_dt:
            tc_temps = _interp_tc(state.mesh.y_nodes, state.T, options.tc_positions)
            results.append(state, tc_temps if len(tc_temps) > 0 else None)
            next_output_t += options.output_dt
            if verbose:
                print(f"  t={t:.3f} s  T_wall={state.T_wall:.1f} K  "
                      f"s={state.mesh.s_total*1000:.3f} mm  "
                      f"s_dot={state.s_dot*1000:.4f} mm/s  dt={dt:.4f} s")

    if verbose:
        print(f"SCAM: done. {n_steps} steps, {n_drops} node drops.")
        print(f"  Final recession: {state.mesh.s_total*1000:.3f} mm")

    return results

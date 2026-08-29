# SPDX-License-Identifier: MIT
"""In-depth solver: one complete timestep of the material response.

Sequence for one timestep (following the CMA algorithm, Figure 6.17):

1. Interpolate thermal-grid temperatures to the decomposition subgrid (nodelets).
2. Update nodelet densities via Arrhenius ODE integration.
3. Project nodalet densities back to thermal-grid nodal densities.
4. Compute pyrolysis gas mass flux m_dot_g(y) from drho_dt_y.
5. Assemble the FVM tridiagonal system for the energy equation.
6. Compute F_cond linear coefficients (alpha_F, beta_F) via back-substitution.
7. Solve the surface energy balance (SEB) Newton iterate → T_wall, q_cond, m_dot_char.
8. Insert q_cond into D[0] of the tridiagonal system and solve for T_new.
9. Return updated SimState.

For prescribed-flux or prescribed-temperature surface BCs, step 7 is
simplified (no Newton iteration required).
"""

from __future__ import annotations

import copy

import numpy as np

from scam.config.boundary import SurfaceBCConfig, BackBCConfig, SurfaceBCType, eval_bc
from scam.config.stack import StackConfig
from scam.config.solver import SolverOptions
from scam.core.state import SimState
from scam.numerics.assembly import assemble_energy_system, compute_F_cond  # assemble_energy_system used for Dirichlet BC path
from scam.numerics.tridiagonal import solve_thomas
from scam.physics.decomposition import (
    update_nodelet_densities,
    nodal_density_from_nodelets,
)
from scam.physics.pyrolysis_gas import pyrolysis_gas_flow_rate
from scam.physics.recession import surface_recession_rate
from scam.mesh.interpolation import interpolate_layer_to_nodelets
from scam.solvers.surface_solver import solve_surface
from scam.physics.element_transport import solve_element_transport
from scam.physics.properties import mixture_enthalpy_array as _mixture_enthalpy


def step(
    state: SimState,
    mat_list: list,          # list[MaterialCard], one per layer
    b_prime_tables: list,    # list[BPrimeTable | None], one per layer
    stack: StackConfig,
    surface_bc: SurfaceBCConfig,
    back_bc: BackBCConfig,
    options: SolverOptions,
    time: float,
    dt: float,
) -> SimState:
    """Advance the simulation by one timestep.

    Parameters
    ----------
    state:
        Current simulation state.
    mat_list:
        Ordered list of MaterialCard objects indexed by layer id.
    b_prime_tables:
        B' tables for each layer (None for inert layers).
    stack:
        Stack configuration.
    surface_bc:
        Surface boundary condition.
    back_bc:
        Back-face boundary condition.
    options:
        Solver options.
    time:
        Current simulation time [s] (before advancing).
    dt:
        Timestep for this step [s].

    Returns
    -------
    SimState
        Updated simulation state at time + dt.
    """
    mesh = state.mesh
    N = mesh.n_nodes_total
    lid = mesh.layer_id

    # ------------------------------------------------------------------
    # 1. Interpolate T to nodelets
    # ------------------------------------------------------------------
    # For each decomposing layer, build a (n_nodes_local, J) temperature array
    # by interpolating between neighbouring thermal nodes.
    T_nodelets_per_layer = []
    for l_idx, layer in enumerate(stack.layers):
        rho_comp_l = state.rho_components[l_idx]
        if rho_comp_l is None:
            T_nodelets_per_layer.append(None)
            continue

        lo, hi = mesh.layer_boundaries[l_idx]
        N_l = hi - lo
        nd_l = mesh.nodelet_delta[l_idx]   # (N_l, J)
        A_face = mesh.area_nodes[lo:hi]    # (N_l,) — approximate face areas

        # Build T_left/T_center/T_right for all nodes in the layer at once
        g = np.arange(lo, hi)                          # global indices (N_l,)
        T_left_arr   = np.where(g > 0,     state.T[np.maximum(g - 1, 0)], state.T[g])
        T_right_arr  = np.where(g < N - 1, state.T[np.minimum(g + 1, N - 1)], state.T[g])
        T_center_arr = state.T[g]

        T_nod = interpolate_layer_to_nodelets(
            T_left_arr, T_center_arr, T_right_arr, nd_l, A_face,
        )

        T_nodelets_per_layer.append(T_nod)

    # ------------------------------------------------------------------
    # 2. Update nodelet densities (Arrhenius decomposition)
    # ------------------------------------------------------------------
    rho_comp_new = []
    drho_dt_y_global = np.zeros(N)   # [kg/m^3/s] per node (negative = decomposing)
    # Per-component nodal rates — allocated lazily when first decomposing layer encountered
    drho_dt_y_comp_global = None

    s_dot_prev = state.s_dot  # recession rate from previous step
    # In ALE mode the ALE step re-interpolates rho_components to new nodelet
    # positions at the end of every step.  Applying the within-step convective
    # correction (v * d_rho/dy) on top of the ALE advection double-counts the
    # same physical effect.  Pass s_dot=0 so update_nodelet_densities uses the
    # pure Lagrangian decomposition rate.  In the discrete Lagrangian scheme
    # (continuous_remap=False) nodelet positions are fixed, so the advective
    # correction is needed to convert to the Eulerian rate at fixed y.
    s_dot_decomp = 0.0 if options.continuous_remap else s_dot_prev
    for l_idx, layer in enumerate(stack.layers):
        rho_comp_l = state.rho_components[l_idx]
        mat = mat_list[l_idx]
        if rho_comp_l is None or not mat.decomposing or len(mat.components) == 0:
            rho_comp_new.append(rho_comp_l)
            continue

        lo, hi = mesh.layer_boundaries[l_idx]
        N_l = hi - lo
        T_nod = T_nodelets_per_layer[l_idx]
        nd_l  = mesh.nodelet_delta[l_idx]

        is_shrinking = np.zeros(N_l, dtype=bool)
        is_shrinking[0] = (l_idx == 0)  # surface node is shrinking for surface layer

        rho_comp_updated, drho_dt_layer, drho_dt_comp_layer = update_nodelet_densities(
            mat, rho_comp_l, T_nod, dt, s_dot_decomp, nd_l, is_shrinking,
            backend=options.array_backend,
        )
        rho_comp_new.append(rho_comp_updated)
        drho_dt_y_global[lo:hi] = drho_dt_layer

        # Accumulate per-component rates if any component has h_decomp != 0
        if any(c.h_decomp != 0.0 for c in mat.components):
            if drho_dt_y_comp_global is None:
                n_comp_max = max(len(m.components) for m in mat_list if m.components)
                drho_dt_y_comp_global = np.zeros((n_comp_max, N))
            n_comp_l = len(mat.components)
            drho_dt_y_comp_global[:n_comp_l, lo:hi] = drho_dt_comp_layer

    # ------------------------------------------------------------------
    # 3. Project nodelet densities back to thermal grid
    # ------------------------------------------------------------------
    rho_new = state.rho.copy()
    for l_idx, layer in enumerate(stack.layers):
        rho_comp_l = rho_comp_new[l_idx]
        mat = mat_list[l_idx]
        if rho_comp_l is None or not mat.decomposing:
            continue
        lo, hi = mesh.layer_boundaries[l_idx]
        nd_l  = mesh.nodelet_delta[l_idx]
        area_l = np.zeros_like(nd_l)
        for n_local in range(hi - lo):
            area_l[n_local, :] = mesh.area_nodes[lo + n_local]  # broadcast

        rho_new[lo:hi] = nodal_density_from_nodelets(
            mat, rho_comp_l, nd_l, area_l,
        )

    # ------------------------------------------------------------------
    # 4. Pyrolysis gas mass flux
    # ------------------------------------------------------------------
    # Only the surface (ablating) layer generates pyrolysis gas.
    surface_layer_bounds = mesh.layer_boundaries[0]
    mat_surface = mat_list[0]
    bpt_surface = b_prime_tables[0] if b_prime_tables else None

    if mat_surface.decomposing and len(mat_surface.components) > 0:
        m_dot_g_nodes = pyrolysis_gas_flow_rate(
            drho_dt_y_global,
            mesh.delta_nodes,
            mesh.area_nodes,
            surface_layer_bounds,
        )
        lo = surface_layer_bounds[0]
        A_surface = mesh.area_nodes[lo]
        m_dot_pyro = m_dot_g_nodes[lo] / A_surface if A_surface > 0.0 else 0.0
    else:
        m_dot_g_nodes = np.zeros(N)
        m_dot_pyro = 0.0

    # ------------------------------------------------------------------
    # 4.5. Element transport (optional — only when Z_elem is initialised)
    # ------------------------------------------------------------------
    Z_C_pyro = None
    Z_elem_new = state.Z_elem
    if state.Z_elem is not None:
        # The element-transport carrier gas IS the pyrolysis gas, so it must be
        # advected with the pyrolysis mass flux (m_dot_g_nodes), NOT the Darcy
        # thermal-expansion flux.  m_dot_g_nodes is the reverse-cumulative sum of
        # (-drho_dt)·A·delta, so the face-flux it defines satisfies cell-wise
        # continuity with the decomposition source exactly:
        #     f_out·A - f_in·A = (-drho_dt)·V.
        # That continuity is what bounds each Z_i to a convex combination of the
        # pyrolysis fractions; using an inconsistent flux (as before) let every
        # Z_i run away and clip to 1.0.  face_flux density [kg/m^2/s] at face j
        # (between nodes j-1 and j) = m_dot_g_nodes[j] / area_nodes[j].
        drho_dt_all = (rho_new - state.rho) / dt
        face_flux_pyro = np.zeros(N + 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            mask = mesh.area_nodes > 0
            face_flux_pyro[:N][mask] = m_dot_g_nodes[mask] / mesh.area_nodes[mask]
        Z_elem_new = solve_element_transport(
            state.Z_elem, mat_list, mesh,
            state.T, rho_new, drho_dt_all, face_flux_pyro, dt,
        )
        # Elemental mass fractions must sum to 1 at each node.  The transport
        # operator solves each element independently and clips to [0,1], which
        # can leave a small (~1-2%) per-node drift; renormalise here (at the
        # physical integration point, keeping the operator itself pure).
        col_sum = Z_elem_new.sum(axis=0)
        good = col_sum > 1e-12
        Z_elem_new[:, good] /= col_sum[np.newaxis, good]
        Z_C_pyro = float(Z_elem_new[0, 0])   # carbon fraction at surface node

    # ------------------------------------------------------------------
    # 5-7. Solve surface BC and assemble/solve the tridiagonal system.
    # ------------------------------------------------------------------
    # Create a temporary state with updated rho for property evaluation.
    # state_updated.T is updated each Picard iteration to evaluate cp(T^k).
    state_updated = copy.copy(state)
    state_updated.rho = rho_new
    state_updated.rho_components = rho_comp_new

    emissivity_override = -1.0
    if surface_bc.emissivity > 0:
        emissivity_override = surface_bc.emissivity

    # Precompute h(T^n, ρ_old) ONCE — fixed throughout Picard iterations.
    # use_rho_old=True (default, exact d(ρh)/dt):
    #   Dc_thermal = ρ_old·h(T^n,ρ_old) − ρ_new·h(T^k,ρ_new)   (per unit volume)
    #   The density-change term (ρ_old − ρ_new)·h̄_sensible provides the sensible
    #   enthalpy carried away by decomposing mass implicitly; Q_vol carries only
    #   h_bar_chemical (for table materials) or h_decomp (pure chemical hp).
    # use_rho_old=False (PATO approximation ρ·dh/dt ≈ ρ·cp·dT/dt):
    #   Dc_thermal = ρ_new·cp·T^n·A·Δ/dt; h̄_sensible not implicit.
    if options.use_rho_old:
        h_old_arr = _mixture_enthalpy(mat_list, state.T, state.rho, lid)
        _rho_old = state.rho
        _T_old_rhs = None
    else:
        h_old_arr = None
        _rho_old = None
        _T_old_rhs = state.T.copy()   # T^n — fixed throughout Picard iterations

    if surface_bc.bc_type == SurfaceBCType.PRESCRIBED_TEMP:
        # --- Dirichlet BC: pin T[0] = T_wall directly ---
        # Use large-conductance penalty in the tridiagonal (no F_cond needed).
        T_wall = float(eval_bc(surface_bc.T_prescribed, time + dt))
        m_dot_char = 0.0

        tri = assemble_energy_system(
            state_updated, mat_list, stack, dt,
            q_cond_surface=0.0,       # not used when Dirichlet is active
            back_bc=back_bc, time=time,
            drho_dt_y=drho_dt_y_global, m_dot_g=m_dot_g_nodes,
            drho_dt_y_comp=drho_dt_y_comp_global,
            rho_old=_rho_old, T_prev=state.T_prev, h_old=h_old_arr,
            T_old_rhs=_T_old_rhs,
            T_surface_dirichlet=T_wall,
        )
        T_new = solve_thomas(tri, backend=options.array_backend)

        # Compute actual surface conduction flux from FVM energy balance on node 0.
        # With the ρ·h(T) formulation: q_cond·A = [ρ_new·h(T_new) − ρ_old·h(T_old)]·V/dt
        #                                           + G01·(T_new[0] − T_new[1])
        from scam.physics.properties import thermal_conductivity, mixture_enthalpy_array as _me
        from scam.geometry.fvm import interface_conductance, face_area
        mesh0 = state_updated.mesh
        mat0 = mat_list[mesh0.layer_id[0]]
        k0 = thermal_conductivity(mat0, T_new[0], rho_new[0])
        k1 = thermal_conductivity(mat_list[mesh0.layer_id[1]], T_new[1], rho_new[1])
        A_f = face_area(mesh0.area_nodes[0], mesh0.area_nodes[1])
        G01 = interface_conductance(k0, mesh0.delta_nodes[0], k1, mesh0.delta_nodes[1], A_f, 0.0)
        lid0 = mesh0.layer_id[:1]
        h_new_0 = float(_me(mat_list, np.array([T_new[0]]),  np.array([rho_new[0]]),  lid0)[0])
        h_old_0 = float(_me(mat_list, np.array([state.T[0]]), np.array([state.rho[0]]), lid0)[0])
        M_vol = mesh0.area_nodes[0] * mesh0.delta_nodes[0]
        A_surface = float(mesh0.area_nodes[0])
        q_cond = ((rho_new[0] * h_new_0 - state.rho[0] * h_old_0) * M_vol / dt
                  + G01 * (T_new[0] - T_new[1])) / A_surface

    else:
        # --- Neumann / SEB BC: F_cond back-substitution with Picard iteration ---
        # Picard loop: re-assemble with cp(T^k) each iteration until T converges.
        T_k = state.T.copy()
        T_wall = state.T_wall      # initial Newton guess (updated each iteration)
        q_cond = state.q_cond
        m_dot_char = 0.0
        h_wall = 0.0

        for _picard in range(options.max_picard):
            state_updated.T = T_k   # update cp(T^k) in the assembly Jacobian
            alpha_F, beta_F, tri_prebuilt = compute_F_cond(
                state_updated, mat_list, stack, dt, back_bc, time,
                drho_dt_y=drho_dt_y_global, m_dot_g=m_dot_g_nodes,
                drho_dt_y_comp=drho_dt_y_comp_global,
                rho_old=_rho_old, T_prev=state.T_prev, h_old=h_old_arr,
                T_old_rhs=_T_old_rhs,
                backend=options.array_backend,
            )

            T_wall, q_cond, m_dot_char, h_wall = solve_surface(
                state, alpha_F, beta_F, surface_bc, time, m_dot_pyro,
                mat_surface, bpt_surface, options, emissivity_override, Z_C_pyro,
            )

            # Patch D[0] with the resolved conductive flux and solve.
            tri_prebuilt.D[0] += q_cond * float(state_updated.mesh.area_nodes[0])
            T_new = solve_thomas(tri_prebuilt, backend=options.array_backend)

            if np.max(np.abs(T_new - T_k)) < options.picard_tol:
                break
            T_k = T_new

    # ------------------------------------------------------------------
    # 8. Compute recession rate and assemble new state
    # ------------------------------------------------------------------
    rho_wall = float(rho_new[0])
    s_dot_new = surface_recession_rate(m_dot_char, max(rho_wall, mat_surface.rho_char))

    new_state = copy.copy(state)
    new_state.time = time + dt
    new_state.dt   = dt
    new_state.T    = T_new
    new_state.T_prev = state.T   # store current T as T_prev for next step's dT/dt estimate
    new_state.rho  = rho_new
    new_state.rho_components = rho_comp_new
    new_state.s_dot      = s_dot_new
    new_state.rho_wall   = rho_wall   # pre-recession surface density; consistent with s_dot
    new_state.T_wall     = T_wall
    new_state.q_cond     = q_cond
    new_state.m_dot_pyro = m_dot_pyro
    new_state.m_dot_char = m_dot_char
    new_state.Z_elem     = Z_elem_new
    return new_state

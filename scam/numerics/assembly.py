# SPDX-License-Identifier: MIT
"""Tridiagonal coefficient assembly for the 1-D FVM energy equation.

Energy equation for node n (cell-centred FVM with area function A(y)):

    d/dt [ rho_n * cp_n * T_n * A_n * delta_n ]
        = G_{n-1/2} * (T_{n-1} - T_n) - G_{n+1/2} * (T_n - T_{n+1})
          + Q_decomp_n * A_n * delta_n
          + Q_pyro_n   * A_n * delta_n

Semi-implicit Backward Euler (theta = 1) for conduction, explicit for sources:

    A_n * T_{n-1}^{k+1} + B_n * T_n^{k+1} + C_n * T_{n+1}^{k+1} = D_n

where:
    A_n = -G_{n-1/2}
    C_n = -G_{n+1/2}
    B_n = M_n + G_{n-1/2} + G_{n+1/2}
    D_n = M_n * T_n^k + (Q_decomp_n + Q_pyro_n) * A_n * delta_n

    M_n  = rho_n * cp_n * A_n * delta_n / dt    [thermal mass / dt]
    G_{n+1/2} = conductance between nodes n and n+1 [W/K]

Surface node (n = 0, half-node):
    The left face receives prescribed heat flux q_cond [W/m^2]:
        (in-material direction positive)
    A_0 = 0 (no left neighbour)
    D_0 += q_cond * A_face_0  (where A_face_0 = A at the surface position)

Back node (n = N-1, half-node):
    Adiabatic: C_{N-1} = 0 (already zero by convention); no flux added to D.
    Prescribed T: handled by pinning T_{N-1} externally after solve.
    Prescribed flux: D_{N-1} += q_back * A_face_back.

F_cond back-substitution
------------------------
The SEB requires q_cond(T_wall).  The tridiagonal can be partially reduced
(forward elimination) to express q_cond as a linear function of T_wall:

    q_cond = alpha_F * T_wall + beta_F

This is computed by assembling the system with T_wall as a free parameter,
running the forward sweep, and reading off the resulting expression at node 0.

Layer interfaces
----------------
Adjacent half-nodes in neighbouring layers are coupled by the inter-layer
conductance G_interface (which includes optional contact resistance).
"""

from __future__ import annotations

from typing import Optional, Callable

import numpy as np
from numpy.typing import NDArray

from scam.core.constants import SIGMA_SB
from scam.config.material import MaterialCard
from scam.config.stack import StackConfig
from scam.config.boundary import BackBCConfig, BackBCType, eval_bc
from scam.core.state import SimState
from scam.geometry.fvm import interface_conductance, face_area
from scam.numerics.tridiagonal import TridiagSystem
from scam.physics.properties import eval_k_array, eval_cp_array


# ---------------------------------------------------------------------------
# Conductance between two adjacent nodes
# ---------------------------------------------------------------------------

def _node_conductance(
    k_n: float, delta_n: float, A_n: float,
    k_m: float, delta_m: float, A_m: float,
    contact_R: float = 0.0,
) -> float:
    """Conductance G_{n → m} between nodes n and m [W/K]."""
    A_f = face_area(A_n, A_m)
    return interface_conductance(k_n, delta_n, k_m, delta_m, A_f, contact_R)


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------

def _build_system(
    state: SimState,
    mat_list: list,
    stack: StackConfig,
    dt: float,
    back_bc: BackBCConfig,
    time: float,
    drho_dt_y: Optional[NDArray] = None,
    m_dot_g: Optional[NDArray] = None,
    drho_dt_y_comp: Optional[NDArray] = None,
    rho_old: Optional[NDArray] = None,
    T_prev: Optional[NDArray] = None,
    h_old: Optional[NDArray] = None,
    T_old_rhs: Optional[NDArray] = None,
    p_surface: float = 101325.0,
) -> TridiagSystem:
    """Build the tridiagonal system without applying the surface flux term.

    Shared core used by both ``assemble_energy_system`` and ``compute_F_cond``
    so that properties, conductances, and source terms are computed only once
    per timestep.  The returned system has D[0] ready for the caller to patch
    with either a Neumann flux (``q_cond * A_surface``) or a Dirichlet penalty.
    """
    mesh  = state.mesh
    T     = state.T
    rho   = state.rho
    N     = mesh.n_nodes_total
    delta = mesh.delta_nodes
    A_n   = mesh.area_nodes
    lid   = mesh.layer_id

    # Per-node material properties.  These helpers group nodes by layer and
    # evaluate the vectorized property formulas once per material.
    k  = eval_k_array(mat_list, T, rho, lid)
    cp = eval_cp_array(mat_list, T, rho, lid)

    # Contact resistances at layer interfaces
    contact_R = np.zeros(N)
    for l_idx in range(1, len(stack.layers)):
        lo, hi = mesh.layer_boundaries[l_idx]
        if lo < N:
            contact_R[lo] = stack.layers[l_idx].contact_resistance

    # Conductances (vectorised)
    k_n  = k[:-1];    k_m  = k[1:]
    d_n  = delta[:-1]; d_m  = delta[1:]
    An_  = A_n[:-1];   Am_  = A_n[1:]
    A_f  = 0.5 * (An_ + Am_)
    G_right          = np.zeros(N)
    G_right[:-1]     = A_f / (d_n / (2.0 * k_n) + d_m / (2.0 * k_m) + contact_R[1:])
    G_left           = np.zeros(N)
    G_left[1:]       = G_right[:-1]

    # Thermal mass — Jacobian coefficient ∂(ρh)/∂T ≈ ρ·cp(T^k) at current iterate
    M = rho * cp * A_n * delta / dt
    # RHS energy storage — exact FVM Picard linearisation of ρ·h(T).
    # h_old = h(T^n, ρ_old) is the stored sensible enthalpy at the OLD density and
    # temperature.  Using ρ_old here means the density-change term
    #   [ρ_old·h(T^n,ρ_old) − ρ_new·h(T^k,ρ_new)]
    # appears in Dc_thermal, naturally providing the implicit π·h̄_sensible source
    # that accounts for the sensible enthalpy carried away by the decomposing mass.
    # At convergence T^k→T^*: ρ_new·h(T^*) = ρ_old·h(T^n) + fluxes·dt + Q_chemical·dt
    # This is the exact FVM conservation law for ρ·h storage.
    # Q_vol must then carry only the CHEMICAL part: h_bar_chemical for table materials,
    # or h_decomp (which is the pure chemical hp) for h_decomp-only materials.
    if h_old is not None:
        from scam.physics.properties import mixture_enthalpy_array as _me_inner
        h_k = _me_inner(mat_list, T, rho, lid)   # h(T^k, ρ_new) at current iterate
        _rho_old_rhs = rho_old if rho_old is not None else rho
        Dc_thermal = M * T + (_rho_old_rhs * h_old - rho * h_k) * A_n * delta / dt
    elif T_old_rhs is not None:
        # PATO ρ·dh/dt ≈ ρ·cp·dT/dt: use fixed T^n so Picard RHS is stable.
        # M uses cp(T^k) (current iterate) but T^n (old timestep) is fixed.
        Dc_thermal = M * T_old_rhs
    elif rho_old is not None:
        cp_old     = eval_cp_array(mat_list, T, rho_old, lid)
        Dc_thermal = rho_old * cp_old * T * A_n * delta / dt
    else:
        Dc_thermal = M * T

    # Source terms
    Q_vol = np.zeros(N)

    if drho_dt_y is not None:
        # Chemical pyrolysis energy: Q -= (dρ/dt)·h_bar_chemical.
        # h_bar_chemical = h_bar_absolute − h̄_sensible is the formation-enthalpy part
        # not captured by the ρ_old·h(T^n,ρ_old) storage change on the LHS.  The
        # sensible piece h̄_sensible is now implicit in (ρ_old·h_old − ρ_new·h_k)·/dt.
        # For materials without explicit enthalpy tables, h_bar_chemical = 0 and only
        # h_decomp (pure chemical hp) contributes below.
        from scam.physics.properties import h_bar_chemical_array as _h_bar_chem
        Q_vol -= drho_dt_y * _h_bar_chem(mat_list, T, lid)

    if drho_dt_y_comp is not None:
        h_decomp_arr = np.zeros_like(drho_dt_y_comp, dtype=float)
        for l_idx in np.unique(lid):
            mat = mat_list[int(l_idx)]
            mask = lid == l_idx
            for ic, comp in enumerate(mat.components):
                if comp.h_decomp != 0.0:
                    h_decomp_arr[ic, mask] = comp.h_decomp
        Q_vol += np.einsum('ij,ij->j', drho_dt_y_comp, h_decomp_arr)

    if m_dot_g is not None:
        from scam.physics.gas_enthalpy import pyrolysis_gas_enthalpy_abs
        # Absolute-reference gas enthalpy per node.  With a (p, T)
        # gasProperties table this is h_g(T, p_ambient) — pressure-aware in
        # the decade that matters (h_g shifts ~1.5 MJ/kg between 0.1 and
        # 1 atm at 800 K); without it, the legacy sensible-table +
        # h_g_abs_offset path (bit-identical).
        h_g_cache = np.empty(N)
        for l_idx in np.unique(lid):
            mat = mat_list[int(l_idx)]
            mask = lid == l_idx
            h_g_cache[mask] = pyrolysis_gas_enthalpy_abs(mat, T[mask])

        # The gas exits the surface node at h_g(T_wall) — included explicitly so the
        # in-depth FVM accounts for the outflow correctly.  The SEB q_adv_pyro term
        # m_dot_g*(h_g(T_w) - h_wall) then adds the BL-interaction correction:
        # net at surface = -m_dot_g*h_g(T[0]) + m_dot_g*(h_g(T_w)-h_wall)
        #                ≈ -m_dot_g*h_wall  (PATO-consistent; T[0]≈T_w at convergence)
        # Gating on _bprime_ran is unnecessary because q_adv and q_mass_removal in
        # surface_energy.py use the same absolute reference as h_g_cache.
        mg_out = m_dot_g.copy()
        mg_in = np.zeros(N)
        mg_in[:-1] = m_dot_g[1:]
        h_in = h_g_cache.copy()
        h_in[:-1] = h_g_cache[1:]
        Q_adv = mg_in * h_in - mg_out * h_g_cache
        denom = A_n * delta
        Q_adv_vol = np.zeros(N)
        np.divide(Q_adv, denom, out=Q_adv_vol, where=denom > 0.0)
        Q_vol += Q_adv_vol

    if T_prev is not None and len(T_prev) == N:
        dTdt     = (T - T_prev) / max(dt, 1e-12)
        has_perm = any(mat_list[int(l)].permeability > 0.0 for l in np.unique(lid))
        if has_perm:
            from scam.physics.pressure_darcy import pressure_darcy_energy_source
            Q_darcy = pressure_darcy_energy_source(mat_list, mesh, T, rho, dTdt, p_surface)
        else:
            from scam.physics.darcy_flow import gas_expansion_energy_source
            Q_darcy = gas_expansion_energy_source(mat_list, mesh, T, rho, dTdt)
        denom = A_n * delta
        _buf = np.zeros(N)
        np.divide(Q_darcy, denom, out=_buf, where=denom > 0.0)
        Q_vol += _buf

    if T_prev is not None and len(T_prev) == N:
        from scam.physics.properties import eps_virgin as _eps_v
        from scam.physics.gas_enthalpy import pyrolysis_gas_enthalpy_abs as _hg_abs
        R_univ = 8.314
        T_s  = np.maximum(T,      1.0)
        T_ps = np.maximum(T_prev, 1.0)
        for l_idx in np.unique(lid):
            mat = mat_list[int(l_idx)]
            if mat.eps_g_char == 0.0 and mat.eps_g_virgin == 0.0:
                continue
            if mat.gas_storage_implicit:
                continue  # implicit mode absorbs this into cp (legacy; deprecated)
            mask = lid == l_idx
            ev       = _eps_v(mat, rho[mask])
            eps_g    = mat.eps_g_char + (mat.eps_g_virgin - mat.eps_g_char) * ev
            rho_g_now  = mat.gas_pressure * mat.gas_molar_mass / (R_univ * T_s[mask])
            rho_g_prev = mat.gas_pressure * mat.gas_molar_mass / (R_univ * T_ps[mask])
            hg_now  = _hg_abs(mat, T[mask])
            hg_prev = _hg_abs(mat, T_prev[mask])
            d_dt = (eps_g * rho_g_now * hg_now - eps_g * rho_g_prev * hg_prev) / max(dt, 1e-12)
            denom_m = A_n[mask] * delta[mask]
            Q_vol[mask] -= np.where(denom_m > 0.0, d_dt, 0.0)

    # Assemble A, B, C, D (vectorised)
    Ac = -G_left
    Cc = -G_right
    Bc = M + G_left + G_right
    Dc = Dc_thermal + Q_vol * A_n * delta

    # Back BC
    if back_bc.bc_type == BackBCType.PRESCRIBED_FLUX:
        q_back = eval_bc(back_bc.q_back, time)
        # Sign: q_back > 0 is heat flowing INTO the material, per BackBCType's
        # documented convention, this function's own docstring above, and
        # diagnostics/conservation.py (which adds q_back_cumulative to energy IN).
        # This was `-=` until 2026-09-07, which silently inverted the BC: a
        # positive q_back cooled the back face. Same class of error as the
        # historical F_cond sign bug.
        Dc[N - 1] += q_back * A_n[N - 1]
    elif back_bc.bc_type == BackBCType.PRESCRIBED_TEMP:
        T_back = eval_bc(back_bc.T_back, time)
        G_bc    = 1.0e12
        Bc[N - 1] += G_bc
        Dc[N - 1] += G_bc * T_back

    elif back_bc.bc_type == BackBCType.RADIATION:
        # Back face radiates (and optionally convects) to an enclosure at T_env.
        #   q_loss(T) = eps*sigma*F*(T^4 - T_env^4) + h*(T - T_env)      [W/m^2]
        # This is nonlinear in T, so linearise about the previous iterate T*
        # (Newton form):
        #   q_loss(T) ≈ q_loss(T*) + q'(T*)·(T - T*),
        #   q'(T*)    = 4*eps*sigma*F*T*^3 + h
        # A loss is a sink, so it enters as  B += q'·A  and  D += (q'·T* - q(T*))·A
        # (consistent with the PRESCRIBED_FLUX sign: D += q_in·A).
        T_env = eval_bc(back_bc.T_env_back, time)
        h_b   = eval_bc(back_bc.h_back, time)
        eps_b = float(back_bc.emissivity_back) * float(back_bc.view_factor_back)

        # Linearisation point: the previous iterate if available, else current T.
        T_star = float(T_prev[N - 1]) if T_prev is not None else float(T[N - 1])
        T_star = max(T_star, 1.0)   # guard against nonphysical/zero temperatures

        q_star  = eps_b * SIGMA_SB * (T_star**4 - T_env**4) + h_b * (T_star - T_env)
        dq_dT   = 4.0 * eps_b * SIGMA_SB * T_star**3 + h_b

        Bc[N - 1] += dq_dT * A_n[N - 1]
        Dc[N - 1] += (dq_dT * T_star - q_star) * A_n[N - 1]

    return TridiagSystem(A=Ac, B=Bc, C=Cc, D=Dc)


def assemble_energy_system(
    state: SimState,
    mat_list: list,
    stack: StackConfig,
    dt: float,
    q_cond_surface: float,
    back_bc: BackBCConfig,
    time: float,
    drho_dt_y: Optional[NDArray] = None,
    m_dot_g: Optional[NDArray] = None,
    T_surface_dirichlet: Optional[float] = None,
    drho_dt_y_comp: Optional[NDArray] = None,
    rho_old: Optional[NDArray] = None,
    T_prev: Optional[NDArray] = None,
    h_old: Optional[NDArray] = None,
    T_old_rhs: Optional[NDArray] = None,
    p_surface: float = 101325.0,
) -> TridiagSystem:
    """Assemble the full tridiagonal system for the energy equation."""
    sys = _build_system(
        state, mat_list, stack, dt, back_bc, time,
        drho_dt_y, m_dot_g, drho_dt_y_comp, rho_old, T_prev, h_old, T_old_rhs,
        p_surface,
    )
    A_surface = float(state.mesh.area_nodes[0])
    if T_surface_dirichlet is not None:
        G_surf = 1.0e12
        sys.B[0] += G_surf
        sys.D[0] += G_surf * T_surface_dirichlet
    else:
        sys.D[0] += q_cond_surface * A_surface
    return sys


# ---------------------------------------------------------------------------
# F_cond back-substitution: compute q_cond as a linear function of T_wall
# ---------------------------------------------------------------------------

def compute_F_cond(
    state: SimState,
    mat_list: list,
    stack: StackConfig,
    dt: float,
    back_bc: BackBCConfig,
    time: float,
    drho_dt_y: Optional[NDArray] = None,
    m_dot_g: Optional[NDArray] = None,
    drho_dt_y_comp: Optional[NDArray] = None,
    rho_old: Optional[NDArray] = None,
    T_prev: Optional[NDArray] = None,
    h_old: Optional[NDArray] = None,
    T_old_rhs: Optional[NDArray] = None,
    backend: str = "numpy",
    p_surface: float = 101325.0,
) -> tuple[float, float, TridiagSystem]:
    """Compute the linear F_cond relationship: q_cond = alpha_F * T_wall + beta_F.

    Also returns the pre-built TridiagSystem so the caller can reuse it for
    the final tridiagonal solve (avoiding a second full assembly pass).

    Returns
    -------
    (alpha_F, beta_F, sys) such that q_cond = alpha_F * T_wall + beta_F,
    and sys is the assembled system without surface flux applied to D[0].
    """
    sys = _build_system(
        state, mat_list, stack, dt, back_bc, time,
        drho_dt_y, m_dot_g, drho_dt_y_comp, rho_old, T_prev, h_old, T_old_rhs,
        p_surface,
    )
    N       = state.mesh.n_nodes_total
    A_n     = state.mesh.area_nodes

    # Backward elimination from node N-1 to node 1:
    #   B_eff[n-1] = B[n-1] - C[n-1]*A[n]/B[n]
    #   D_eff[n-1] = D[n-1] - C[n-1]*D[n]/B[n]
    # After sweep: B_sweep[0]*T_0 = D_sweep[0] + q_cond*A_surface
    # → q_cond = (B_sweep[0]/A_surface)*T_wall - D_sweep[0]/A_surface
    if backend == "jax":
        from scam.numerics.jax_kernels import eliminate_f_cond_jax
        b0, d0 = eliminate_f_cond_jax(sys.A, sys.B, sys.C, sys.D)
    elif backend == "numpy":
        B_sweep = sys.B.copy()
        D_sweep = sys.D.copy()
        for n in range(N - 1, 0, -1):
            if B_sweep[n] == 0:
                continue
            factor        = sys.C[n - 1] / B_sweep[n]
            B_sweep[n - 1] -= factor * sys.A[n]
            D_sweep[n - 1] -= factor * D_sweep[n]
        b0, d0 = B_sweep[0], D_sweep[0]
    else:
        raise ValueError(f"Unknown numerical backend: {backend!r}")

    A_surface = float(A_n[0])
    if A_surface == 0:
        return 0.0, 0.0, sys
    alpha_F = b0 / A_surface
    beta_F  = -d0 / A_surface
    return float(alpha_F), float(beta_F), sys

# SPDX-License-Identifier: MIT
"""Arrhenius decomposition kinetics on the nodelet subgrid.

Each thermal node n contains J nodelets indexed 0..J-1 from surface to back.
The decomposition ODE is integrated analytically at the nodelet temperature
(held fixed at T_nodelet over the timestep dt)::

    drho_i/dt = -k_i * rho_0_i * ((rho_i - rho_r_i) / rho_0_i)^m_i
    k_i = A_rate_i * exp(-E_act_i / (R * T_nodelet))

Two branches depending on reaction order m_i:

    m = 1 (first order):
        rho_i(t+dt) = rho_r + (rho_i(t) - rho_r) * exp(-k * dt)

    m ≠ 1 (power-law):
        Let xi = (rho_i - rho_r) / rho_0,  normalized excess density
        d(xi)/dt = -k * xi^m / rho_0^{m-1}   (actually d(rho)/dt=-k*rho_0*xi^m)
        Analytical solution for constant k::

            xi(t+dt)^{1-m} = xi(t)^{1-m} + (m-1) * k/rho_0^{m-1} * dt   ... (*)

        then rho_i(t+dt) = rho_r + rho_0 * xi_new

        (*) derivation: integrate ``xi^{-m} dxi = -k/rho_0^{m-1} dt``::

            xi^{1-m}/(1-m) = constant + (-k/rho_0^{m-1}) t
            xi^{1-m} = xi_0^{1-m} + (m-1)*k/rho_0^{m-1} * dt  (for m>1, xi decreases)

Irreversibility is enforced by clamping rho_i_new in [rho_r, rho_i_old].

Mass conservation on the moving Lagrangian grid includes a convective correction
for the material velocity at each nodelet (eq. 6.1.42 / 6.1.66 in the CMA reference)::

    drho/dt|_y  (at fixed coordinate) includes an advection term
    from the surface motion s_dot.  This is applied AFTER the Arrhenius update
    and before the volume-averaged density is projected back to the thermal grid.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scam.config.material import MaterialCard, ComponentCard
from scam.core.constants import R_UNIVERSAL


# ---------------------------------------------------------------------------
# Arrhenius rate constant
# ---------------------------------------------------------------------------

def arrhenius_k(comp: ComponentCard, T: float) -> float:
    """Reaction rate constant k [1/s] at temperature T [K]."""
    return comp.A_rate * np.exp(-comp.E_act / (R_UNIVERSAL * T))


# ---------------------------------------------------------------------------
# Exact (analytical) density update for one component, one nodelet
# ---------------------------------------------------------------------------

def _update_rho_i(
    comp: ComponentCard,
    rho_i_old: float,
    T_nodelet: float,
    dt: float,
) -> float:
    """Analytically integrate the Arrhenius ODE for component i over dt.

    Parameters
    ----------
    comp:
        ComponentCard for component i.
    rho_i_old:
        Current density of component i [kg/m^3].
    T_nodelet:
        Nodelet temperature [K] (held constant over dt).
    dt:
        Timestep [s].

    Returns
    -------
    float
        New density rho_i_new [kg/m^3], clamped to [rho_r, rho_i_old].
    """
    rho_r = comp.rho_r
    rho_0 = comp.rho_0
    m     = comp.m_exp

    # Excess density above residual
    excess = rho_i_old - rho_r
    if excess <= 0.0:
        return rho_r  # already fully decomposed

    k = arrhenius_k(comp, T_nodelet)

    if m == 1.0:
        # First-order: exponential decay
        excess_new = excess * np.exp(-k * dt)
    else:
        # Power-law:  d(excess)/dt = -k * rho_0 * (excess/rho_0)^m
        #             d(xi)/dt     = -k * xi^m  where xi = excess / rho_0
        xi = excess / rho_0
        if xi <= 0.0:
            return rho_r
        # xi^{1-m}(t+dt) = xi^{1-m}(t) + (m-1) * k * dt
        # (note: for m>1 and xi>0, xi decreases, so xi^{1-m} increases)
        xi_1m_new = xi ** (1.0 - m) + (m - 1.0) * k * dt
        if xi_1m_new <= 0.0:
            return rho_r   # fully reacted in this step
        xi_new = xi_1m_new ** (1.0 / (1.0 - m))
        excess_new = rho_0 * xi_new

    rho_i_new = rho_r + excess_new

    # Irreversibility: rho must be monotonically non-increasing
    rho_i_new = float(np.clip(rho_i_new, rho_r, rho_i_old))
    return rho_i_new


# ---------------------------------------------------------------------------
# Nodelet-level update for one layer
# ---------------------------------------------------------------------------

def update_nodelet_densities(
    mat: MaterialCard,
    rho_comp_layer: NDArray,    # shape (n_comp, n_nodes_local, J)
    T_nodelets: NDArray,         # shape (n_nodes_local, J) — interpolated from thermal grid
    dt: float,
    s_dot: float,
    delta_nodelets: NDArray,     # shape (n_nodes_local, J) — nodelet thicknesses [m]
    is_shrinking: NDArray,       # bool shape (n_nodes_local,) — True for surface node
    backend: str = "numpy",
) -> tuple[NDArray, NDArray]:
    """Update nodelet densities for all components in one layer.

    Also computes ``drho_dt_y`` (the decomposition rate at constant y per node)
    for use in the energy equation source term.

    Parameters
    ----------
    mat:
        MaterialCard for this layer.
    rho_comp_layer:
        Component density on the nodelet subgrid, shape (n_comp, n_nodes_local, J).
        Modified in-place and also returned.
    T_nodelets:
        Nodelet temperatures [K], shape (n_nodes_local, J).
    dt:
        Timestep [s].
    s_dot:
        Surface recession rate [m/s] (>= 0).
    delta_nodelets:
        Nodelet thicknesses [m], shape (n_nodes_local, J).
    is_shrinking:
        True for the surface node (node 0 of the ablating layer).

    Returns
    -------
    (rho_comp_layer_new, drho_dt_y_nodes, drho_dt_y_comp)
        Updated nodelet densities, total nodal decomposition rates [kg/m^3/s]
        shape (n_nodes_local,), and per-component rates shape (n_comp, n_nodes_local).
    """
    if backend == "jax":
        from scam.numerics.jax_kernels import update_nodelet_densities_jax
        return update_nodelet_densities_jax(
            rho_comp_layer,
            T_nodelets,
            dt,
            s_dot,
            delta_nodelets,
            is_shrinking,
            mat.components,
            R_UNIVERSAL,
        )
    if backend != "numpy":
        raise ValueError(f"Unknown numerical backend: {backend!r}")

    n_comp, N_l, J = rho_comp_layer.shape
    rho_new = rho_comp_layer.copy()

    # Nodelet velocity toward surface (Lagrangian material velocity = 0;
    # in fixed-y frame the density field advects because the surface moves).
    # For normal (non-shrinking) nodes: v = s_dot everywhere.
    # For the shrinking node: velocity varies linearly from s_dot at the surface
    # face to 0 at the back face: v_j = s_dot * (J - j - 0.5) / J  (continuous approx.)
    # This implements eq. 6.1.66 from the reference CMA formulation.

    drho_dt_y      = np.zeros(N_l)
    drho_dt_y_comp = np.zeros((n_comp, N_l))

    # Nodelet velocities for all nodes at once — shape (N_l, J)
    j_arr   = np.arange(J, dtype=float) + 0.5
    v_j_all = np.where(
        is_shrinking[:, None],
        s_dot * (J - j_arr) / J,
        s_dot,
    )

    delta_cells = delta_nodelets.sum(axis=1)  # (N_l,)
    safe_dc     = np.where(delta_cells > 0, delta_cells, 1.0)

    for ic, comp in enumerate(mat.components):
        rho_ic = rho_comp_layer[ic]   # (N_l, J)
        excess = rho_ic - comp.rho_r  # (N_l, J)
        active = excess > 0.0

        # Vectorised Arrhenius ODE over the full (N_l, J) plane
        k_nj = comp.A_rate * np.exp(-comp.E_act / (R_UNIVERSAL * T_nodelets))

        if comp.m_exp == 1.0:
            excess_new = np.where(active, excess * np.exp(-k_nj * dt), 0.0)
        else:
            xi = np.where(active, excess / comp.rho_0, 0.0)
            xi_1m = np.where(
                active & (xi > 0),
                xi ** (1.0 - comp.m_exp) + (comp.m_exp - 1.0) * k_nj * dt,
                0.0,
            )
            xi_new    = np.where(xi_1m > 0, xi_1m ** (1.0 / (1.0 - comp.m_exp)), 0.0)
            excess_new = np.where(active, comp.rho_0 * xi_new, 0.0)

        rho_i_new = np.clip(comp.rho_r + excess_new, comp.rho_r, rho_ic)

        # Convective correction: backward difference in j (surface direction)
        d_rho_dy            = np.zeros((N_l, J))
        d_rho_dy[:, 1:]     = (rho_ic[:, 1:] - rho_ic[:, :-1]) / delta_nodelets[:, 1:]

        drho_i_dt   = (rho_i_new - rho_ic) / dt
        drho_i_dt_y = drho_i_dt - v_j_all * d_rho_dy

        rho_new[ic] = np.clip(rho_ic + drho_i_dt_y * dt, comp.rho_r, rho_ic)

        contrib = (drho_i_dt_y * delta_nodelets).sum(axis=1)  # (N_l,)
        drho_dt_y_comp[ic] = contrib / safe_dc
        drho_dt_y          += contrib

    drho_dt_y /= safe_dc

    return rho_new, drho_dt_y, drho_dt_y_comp


# ---------------------------------------------------------------------------
# Volume-averaged nodal density from nodelet array
# ---------------------------------------------------------------------------

def nodal_density_from_nodelets(
    mat: MaterialCard,
    rho_comp_layer: NDArray,   # (n_comp, N_l, J)
    delta_nodelets: NDArray,   # (N_l, J)
    area_nodelets: NDArray,    # (N_l, J) — A(y) at each nodelet centre
) -> NDArray:
    """Compute volume-averaged total density at each thermal node.

    rho_n = fiber_density + sum_i [volume_avg(rho_i_nodelets)]

    Returns array of shape (N_l,).
    """
    V_nodelets = delta_nodelets * area_nodelets        # (N_l, J)
    V_cells    = V_nodelets.sum(axis=1)               # (N_l,)
    safe_V     = np.where(V_cells > 0, V_cells, 1.0)
    # Non-decomposing fiber = rho_char minus the sum of all component residual
    # densities (rho_r).  mat.rho_char INCLUDES those residuals, so adding
    # sum(rho_comp) on top of rho_char double-counts them when any rho_r > 0.
    # Correct: total = fiber + sum_i(vol_avg(rho_i)).  Verified:
    #   initial (all rho_0): fiber + sum(rho_0) = rho_char-sum(rho_r)+sum(rho_0) = rho_virgin
    #   full char (all rho_r): fiber + sum(rho_r) = rho_char-sum(rho_r)+sum(rho_r) = rho_char
    fiber = mat.rho_char - sum(c.rho_r for c in mat.components)
    rho_nodal = fiber + (
        (rho_comp_layer * V_nodelets[np.newaxis]).sum(axis=(0, 2)) / safe_V
    )
    return rho_nodal

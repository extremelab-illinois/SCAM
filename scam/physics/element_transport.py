# SPDX-License-Identifier: MIT
"""Element mass-fraction transport in the porous gas phase.

Solves the 1-D backward-Euler upwind FVM equation for each element i in
{C, H, O, N} (indices 0–3):

    d(eps_g * rho_g * Z_i) / dt  +  d(m_dot_g * Z_i) / dy  =  pi_i

where:
  eps_g * rho_g = eps_g * p * M_g / (R * T)   (ideal gas, constant pressure)
  m_dot_g[face] = Darcy face mass flux (positive = toward surface, i.e. -y direction)
  pi_i[j]       = (-drho_dt[j]) * Z_i_pyro(T_j)   (elemental source from decomposition)

The tridiagonal system per element re-uses the Thomas algorithm from
numerics/tridiagonal.py.

Coordinate conventions (same as the rest of SCAM):
  y = 0 at the hot surface; y increases into the material.
  Node 0 is the surface node; node N-1 is the back node.
  face_flux[n] is the flux at the face between node n-1 and node n.
  face_flux[0] = surface face;  face_flux[N] = back face (zero).
  Positive face_flux → gas flows from node n toward node n-1 (toward surface).
  Upwind direction for face n: deeper node = node n (when face_flux[n] > 0).

Element index convention: C=0, H=1, O=2, N=3.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scam.core.state import MeshState
from scam.numerics.tridiagonal import TridiagSystem, solve_thomas
from scam.physics.properties import eps_virgin, interp_table

# Number of tracked elements and their names
N_ELEM = 4
ELEM_NAMES = ("C", "H", "O", "N")

# Initial air composition (dry air) as elemental mass fractions [C, H, O, N]
Z_ELEM_AIR = np.array([0.0, 0.0, 0.2314, 0.7553])


def _gas_storage(mat, rho_j: float, T_j: float) -> float:
    """eps_g * rho_g [kg/m³] at node j (ideal gas at constant pressure)."""
    ev = float(eps_virgin(mat, rho_j))
    eps_g = mat.eps_g_char + (mat.eps_g_virgin - mat.eps_g_char) * ev
    if eps_g <= 0.0:
        return 0.0
    R_univ = 8.314
    T_safe = max(T_j, 1.0)
    rho_g = mat.gas_pressure * mat.gas_molar_mass / (R_univ * T_safe)
    return eps_g * rho_g


def _pyro_elem_fracs(mat, T_j: float) -> NDArray:
    """Return the elemental mass fractions [C,H,O,N] of the pyrolysis gas at T_j."""
    if mat.pyro_elem_fracs is None:
        # No elemental data — return air (element transport will be trivial)
        return Z_ELEM_AIR.copy()
    pef = np.asarray(mat.pyro_elem_fracs)
    if pef.ndim == 1:
        # Temperature-independent: shape (4,)
        return pef.copy()
    # Temperature-dependent table: shape (4, n_T, 2)
    result = np.empty(N_ELEM)
    for i in range(N_ELEM):
        result[i] = float(interp_table(pef[i], T_j))
    return result


def solve_element_transport(
    Z_elem: NDArray,           # (4, N) — current element mass fractions
    mat_list: list,            # list[MaterialCard], one per layer
    mesh: MeshState,
    T: NDArray,                # (N,) current temperatures [K]
    rho: NDArray,              # (N,) current mixture densities [kg/m³]
    drho_dt: NDArray,          # (N,) density rate [kg/m³/s], negative during decomp
    face_flux: NDArray,        # (N+1,) Darcy gas face flux [kg/m²/s]
    dt: float,
) -> NDArray:                  # (4, N) updated Z_elem
    """Advance the element mass-fraction field by one timestep.

    Each element i is solved independently via the Thomas algorithm.
    The gas storage term uses the ideal-gas approximation at constant pressure.
    Diffusion is included using a central-difference discretisation (optional:
    skipped when element_diffusivity = 0).

    Returns
    -------
    Z_new : ndarray, shape (4, N)
        Updated element mass fractions clipped to [0, 1].
    """
    N = mesh.n_nodes_total
    lid = mesh.layer_id
    delta = mesh.delta_nodes
    A_n = mesh.area_nodes
    R_univ = 8.314

    # --- Vectorised storage: eps_g * rho_g [kg/m³], layer-wise ---
    storage = np.zeros(N)
    T_s = np.maximum(T, 1.0)
    for l_idx in np.unique(lid):
        mat = mat_list[int(l_idx)]
        if mat.eps_g_char == 0.0 and mat.eps_g_virgin == 0.0:
            continue
        mask = lid == l_idx
        ev    = eps_virgin(mat, rho[mask])
        eps_g = mat.eps_g_char + (mat.eps_g_virgin - mat.eps_g_char) * ev
        rho_g = mat.gas_pressure * mat.gas_molar_mass / (R_univ * T_s[mask])
        storage[mask] = np.where(eps_g > 0.0, eps_g * rho_g, 0.0)

    # --- Vectorised pyrolysis elemental fractions: Z_pyro (N_ELEM, N) ---
    Z_pyro = np.zeros((N_ELEM, N))
    for l_idx in np.unique(lid):
        mat = mat_list[int(l_idx)]
        mask = lid == l_idx
        pef = mat.pyro_elem_fracs
        if pef is None:
            Z_pyro[:, mask] = Z_ELEM_AIR[:, None]
            continue
        pef = np.asarray(pef)
        if pef.ndim == 1:
            Z_pyro[:, mask] = pef[:, None]
        else:  # temperature-dependent table: shape (4, n_T, 2)
            for ie in range(N_ELEM):
                Z_pyro[ie, mask] = np.interp(T[mask], pef[ie, :, 0], pef[ie, :, 1])

    # --- Pre-compute per-cell volumes and face fluxes ---
    V = delta * A_n                      # (N,) cell volumes [m³]
    f_out = face_flux[:N]                # (N,) flux at face j (between j-1 and j)
    f_in  = face_flux[1:N + 1]          # (N,) flux at face j+1; face_flux[N]=0 (back BC)

    # Convective diagonal/off-diagonal contributions (upwind, independent of element i):
    #   face j (f_out): f_out >= 0 → flow j→j-1, upwind=j → B += f_out*A_n
    #                   f_out <  0 → flow j-1→j, upwind=j-1 → A_coef[j] += f_out*A_n (j>0 only)
    #   face j+1 (f_in): f_in >= 0 → flow j+1→j, upwind=j+1 → C_coef[j] -= f_in*A_n
    #                    f_in <  0 → flow j→j+1, upwind=j → B -= f_in*A_n (j<N-1 only,
    #                                but face_flux[N]=0 so f_in[N-1]=0 already)
    conv_B  = np.maximum(f_out, 0.0) * A_n - np.minimum(f_in, 0.0) * A_n
    conv_A  = np.zeros(N)
    conv_A[1:] = np.minimum(f_out[1:], 0.0) * A_n[1:]   # j=0 has no j-1 neighbour
    conv_C  = -np.maximum(f_in, 0.0) * A_n               # C[N-1]=0 (f_in[N-1]=face_flux[N]=0)

    Z_new = np.empty_like(Z_elem)

    for i in range(N_ELEM):
        Zi = Z_elem[i]

        # Storage diagonal and old-time RHS
        B_coef = storage * V / dt + conv_B
        D_rhs  = storage * V / dt * Zi
        # Pyrolysis source: pi_i = (-drho_dt) * Z_i_pyro
        D_rhs += (-drho_dt) * Z_pyro[i] * V
        A_coef = conv_A.copy()
        C_coef = conv_C.copy()

        # --- Diffusion (central difference, implicit) — scalar loop only when D_eff > 0 ---
        for l_idx in np.unique(lid):
            mat = mat_list[int(l_idx)]
            D_eff = mat.element_diffusivity / mat.tortuosity
            if D_eff <= 0.0:
                continue
            idxs = np.where(lid == l_idx)[0]
            for j in idxs:
                A_face = float(A_n[j])
                if j > 0:
                    dy_L   = 0.5 * (float(delta[j - 1]) + float(delta[j]))
                    s_L    = 0.5 * (storage[j - 1] + storage[j])
                    kappa_L = D_eff * s_L * A_face / dy_L
                    B_coef[j] += kappa_L
                    A_coef[j] -= kappa_L
                if j + 1 < N:
                    dy_R   = 0.5 * (float(delta[j]) + float(delta[j + 1]))
                    s_R    = 0.5 * (storage[j] + storage[j + 1])
                    kappa_R = D_eff * s_R * A_face / dy_R
                    B_coef[j] += kappa_R
                    C_coef[j] -= kappa_R

        # Degenerate nodes (zero storage, no porosity): hold Z constant
        degen = (B_coef <= 0.0) | (storage <= 0.0)
        A_coef = np.where(degen, 0.0, A_coef)
        B_coef = np.where(degen, 1.0, B_coef)
        C_coef = np.where(degen, 0.0, C_coef)
        D_rhs  = np.where(degen, Zi,  D_rhs)

        sys = TridiagSystem(A=A_coef, B=B_coef, C=C_coef, D=D_rhs)
        Z_new[i] = np.clip(solve_thomas(sys), 0.0, 1.0)

    return Z_new


def initial_Z_elem(N: int, z0: "NDArray | None" = None) -> NDArray:
    """Return initial element mass fractions, shape (4, N).

    Parameters
    ----------
    N:
        Number of nodes.
    z0:
        Optional (4,) array of initial elemental mass fractions [C, H, O, N].
        Defaults to dry air composition when None.
    """
    if z0 is None:
        z0 = Z_ELEM_AIR
    z0 = np.asarray(z0, dtype=float)
    Z = np.zeros((N_ELEM, N))
    for i in range(N_ELEM):
        Z[i, :] = z0[i]
    return Z

# SPDX-License-Identifier: MIT
"""Gas expansion-driven (Darcy) flow advection energy source.

Implements the constant-pressure approximation for porous ablators: gas pressure
is approximately uniform and equal to the free-stream value p_0.  As the solid
heats up, the ideal-gas density drops (ρ_g ∝ 1/T at constant p), which drives
gas outward toward the surface via Darcy flow.  This gas carries sensible
enthalpy, acting as a heat sink in the energy equation.

Relationship to the gas energy STORAGE correction in properties.py:

The full gas energy term is ``d(eps_g*rho_g*h_g)/dt + d(rho_g*h_g*u_g)/dy``.
Expanding the time-derivative and using gas continuity at constant p shows
that the two parts split into:

- Storage: ``eps_g*rho_g*cp_g*dT/dt`` (handled as a cp correction in properties.py)
- Advection: ``rho_g*u_g*cp_g*dT/dy`` (handled HERE, using sensible h_g)

Using SENSIBLE enthalpy (``h_g - h_g(T_ref)``) in the advection formula avoids
double-counting the formation-energy bookkeeping already included in the cp
correction. The combination of the two terms recovers the full PATO
EnergyType-Pyrolysis + MassType-DarcyLaw behaviour.

Sign conventions (match the rest of SCAM):
    y = 0 at the hot surface, increases INTO the material.
    Gas flows toward the surface → mass_flux > 0 (same convention as m_dot_g).
    Node n=0 is the surface node; node n=N-1 is the back node.
    face_flux[n] is the flux at the face between node n-1 and node n.
    face_flux[0] = surface face; face_flux[N] = back wall (zero, no-flow BC).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scam.config.material import MaterialCard
from scam.core.state import MeshState
from scam.physics.properties import eps_virgin, interp_table


def _any_gas_porosity(mat_list: list, lid: NDArray) -> bool:
    """Return True if any node has non-zero gas porosity."""
    seen: set = set()
    for layer_id in lid:
        if layer_id in seen:
            continue
        seen.add(layer_id)
        mat = mat_list[layer_id]
        if mat.eps_g_char != 0.0 or mat.eps_g_virgin != 0.0:
            return True
    return False


def gas_expansion_mass_flux(
    mat_list: list,
    mesh: MeshState,
    T: NDArray,
    rho: NDArray,
    dTdt: NDArray,
) -> NDArray:
    """Gas mass flux driven by thermal expansion at constant pressure.

    Integrates the 1-D gas continuity equation from the back wall (no-flow
    boundary) toward the surface.  The result matches the m_dot_g sign
    convention: positive values mean gas flows toward the surface.

    Parameters
    ----------
    mat_list : list[MaterialCard]
    mesh     : current MeshState
    T        : nodal temperatures [K], shape (N,)
    rho      : nodal mixture densities [kg/m³], shape (N,)
    dTdt     : temperature time-derivative estimate [K/s], shape (N,)

    Returns
    -------
    face_flux : shape (N+1,) [kg/(m²·s)]
        face_flux[0]  = flux at the surface face (cumulative)
        face_flux[N]  = 0  (back-wall no-flow BC)
        face_flux[n]  = flux at face between node n-1 and node n
    """
    N = mesh.n_nodes_total
    lid = mesh.layer_id
    delta = mesh.delta_nodes

    if not _any_gas_porosity(mat_list, lid):
        return np.zeros(N + 1)

    R_univ = 8.314  # J/(mol·K)
    eps_g = np.zeros(N)
    pM = np.zeros(N)
    for l_idx in np.unique(lid):
        mask = lid == l_idx
        mat = mat_list[int(l_idx)]
        if mat.eps_g_char == 0.0 and mat.eps_g_virgin == 0.0:
            continue
        ev = eps_virgin(mat, rho[mask])
        eps_g[mask] = mat.eps_g_char + (mat.eps_g_virgin - mat.eps_g_char) * ev
        pM[mask] = mat.gas_pressure * mat.gas_molar_mass

    T_safe = np.maximum(T, 1.0)
    # From ideal gas:  ρ_g = p·M/(R·T)  →  ∂ρ_g/∂t = −(p·M)/(R·T²)·∂T/∂t
    # Continuity:  m_flux[n] = m_flux[n+1] + ε_g·(p·M)/(R·T²)·∂T/∂t·Δy_n
    source = eps_g * pM / (R_univ * T_safe ** 2) * dTdt * delta
    face_flux = np.zeros(N + 1)  # face_flux[N] = 0 (back, no-flow)
    face_flux[:-1] = np.cumsum(source[::-1])[::-1]

    return face_flux


def gas_expansion_energy_source(
    mat_list: list,
    mesh: MeshState,
    T: NDArray,
    rho: NDArray,
    dTdt: NDArray,
) -> NDArray:
    """Nodal energy source from gas-expansion advection [W] per cell.

    Uses SENSIBLE enthalpy h_g(T) − h_g(T_ref) to avoid double-counting the
    formation-energy term already included in the cp correction of properties.py.

    Positive value = energy source; negative = sink (gas expansion acts as a
    heat sink where ∂T/∂t > 0 and the temperature gradient is non-zero).

    Parameters
    ----------
    (same as gas_expansion_mass_flux)

    Returns
    -------
    Q_adv : shape (N,) [W]
        Energy source per cell (to be added directly to D[n] in the tridiagonal).
    """
    N = mesh.n_nodes_total
    lid = mesh.layer_id
    delta = mesh.delta_nodes
    A_n = mesh.area_nodes

    if not _any_gas_porosity(mat_list, lid):
        return np.zeros(N)

    face_flux = gas_expansion_mass_flux(mat_list, mesh, T, rho, dTdt)

    # Sensible gas enthalpy: h_g_sensible(T) = h_g(T) - h_g(T_ref)
    # T_ref is the first temperature entry in h_g_table (typically 300 K).
    h_sens = np.empty(N)
    for l_idx in np.unique(lid):
        mask = lid == l_idx
        mat = mat_list[int(l_idx)]
        h_ref = float(mat.h_g_table[0, 1])
        h_sens[mask] = interp_table(mat.h_g_table, T[mask]) - h_ref

    # Upwind: gas flows toward surface (face_flux > 0), upwind node is the
    # deeper one (n for face n, n+1 for face n+1).
    f_in = face_flux[1:]
    h_in = h_sens.copy()
    h_in[:-1] = h_sens[1:]
    Q_adv = f_in * h_in - face_flux[:-1] * h_sens

    return Q_adv

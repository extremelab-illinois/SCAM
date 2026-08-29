# SPDX-License-Identifier: MIT
"""B' table surface chemistry.

The B' table (stored as a 3-D array indexed by T_wall, p_e, B'_g)
maps equilibrium surface chemistry to a dimensionless char ablation rate B'_c
and a wall enthalpy h_wall.

The BPrimeTable class is defined in io/material_loader.py (where it is also
loaded from YAML).  This module re-exports it and provides the helper
functions used by the surface energy balance.
"""

from __future__ import annotations

# Re-export the BPrimeTable class from io/material_loader
from scam.io.material_loader import BPrimeTable


def chemical_heat_flux(
    m_dot_char: float,
    h_wall: float,
    h_char: float,
) -> float:
    """Chemical heat flux into the surface from char ablation [W/m^2].

    q_chem = m_dot_char * (h_wall - h_char)

    where h_wall is the enthalpy of the equilibrium ablation products
    (from the B' table) and h_char is the enthalpy of the incoming char
    (approximated as the char phase enthalpy at T_wall).

    This term is typically positive (energy released by exothermic surface
    reactions) or negative (endothermic sublimation).
    """
    return m_dot_char * (h_wall - h_char)


def lookup_b_prime(
    b_prime_table,
    T_wall: float,
    p_e: float,
    m_dot_pyro: float,
    rho_e_u_e: float,
    C_M: float,
    Z_C_pyro: float | None = None,
) -> tuple[float, float, float]:
    """Perform B' table lookup to get char ablation rate and wall enthalpy.

    Parameters
    ----------
    b_prime_table:
        Loaded BPrimeTable (3-D or 4-D) or BprimeEvaluator for this material.
    T_wall:
        Current surface temperature guess [K].
    p_e:
        Freestream edge pressure [Pa].
    m_dot_pyro:
        Current pyrolysis gas mass flux at surface [kg/m^2/s].
    rho_e_u_e:
        Freestream mass flux [kg/m^2/s].
    C_M:
        Mass transfer Stanton number [-].
    Z_C_pyro:
        Carbon mass fraction in the pyrolysis gas at the surface [-].
        Passed to 4-D B' tables and to BprimeEvaluator; ignored by 3-D tables.
        None → use the table's nominal (middle-axis) value.

    Returns
    -------
    (m_dot_char, B_c_prime, h_wall)
        m_dot_char: char erosion mass flux [kg/m^2/s]
        B_c_prime:  dimensionless char ablation rate [-]
        h_wall:     wall enthalpy of ablation products [J/kg]
    """
    denom = rho_e_u_e * C_M
    B_g = (max(0.0, m_dot_pyro) / denom) if denom > 0 else 0.0

    B_c, h_wall = b_prime_table.lookup(T_wall, p_e, B_g, Z_C_pyro)
    m_dot_char = B_c * denom if denom > 0 else 0.0
    return max(0.0, m_dot_char), max(0.0, B_c), h_wall

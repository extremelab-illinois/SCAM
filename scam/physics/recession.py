# SPDX-License-Identifier: MIT
"""Surface recession rate from the surface mass balance (CMA model).

In the CMA (Charring Material Ablation) model for porous ablators, only char
oxidation drives physical surface recession.  Pyrolysis gas exits through the
porous char via Darcy flow and does NOT ablate the surface.

::

    s_dot = m_dot_char / rho_char    [m/s]

where:

- ``m_dot_char`` = char erosion mass flux [kg/m^2/s] (from B' table)
- ``rho_char`` = fully-charred density of the surface material [kg/m^3]
  (material constant — the surface is assumed fully charred before it
  recedes, consistent with the CMA surface balance)

s_dot >= 0 always (material can only be removed, not added).
"""

from __future__ import annotations


def surface_recession_rate(
    m_dot_char: float,
    rho_char: float,
) -> float:
    """Compute s_dot [m/s] from the CMA surface mass balance.

    Parameters
    ----------
    m_dot_char:
        Char erosion mass flux [kg/m^2/s] (>= 0).
    rho_char:
        Fully-charred density of the surface material [kg/m^3] (must be > 0).

    Returns
    -------
    float
        Surface recession rate [m/s] (>= 0).
    """
    if rho_char <= 0.0 or m_dot_char <= 0.0:
        return 0.0
    return m_dot_char / rho_char

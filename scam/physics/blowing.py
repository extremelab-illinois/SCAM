# SPDX-License-Identifier: MIT
"""Blowing correction to the convective Stanton number.

The presence of pyrolysis gas and char ablation products at the surface
reduces the convective heat transfer due to the "blowing effect".

Standard correction (laminar or turbulent, selectable via lambda_blowing):

    alpha_eff = alpha_conv / (1 + lambda_blowing * B')

where B' is the total dimensionless mass injection rate:

    B' = (m_dot_char + m_dot_pyro) / (rho_e * u_e * C_M)

The individual blowing parameters are:
    B'_g = m_dot_pyro / (rho_e * u_e * C_M)   [pyrolysis gas contribution]
    B'_c = m_dot_char / (rho_e * u_e * C_M)   [char ablation contribution]

Typical values:
    lambda_blowing = 0.5  (laminar boundary layer)
    lambda_blowing = 0.4  (turbulent boundary layer)
"""

from __future__ import annotations


def B_prime_pyro(
    m_dot_pyro: float,
    rho_e_u_e: float,
    C_M: float,
) -> float:
    """Dimensionless pyrolysis gas blowing parameter B'_g [-]."""
    denom = rho_e_u_e * C_M
    if denom <= 0.0:
        return 0.0
    return max(0.0, m_dot_pyro) / denom


def B_prime_char(
    m_dot_char: float,
    rho_e_u_e: float,
    C_M: float,
) -> float:
    """Dimensionless char ablation blowing parameter B'_c [-]."""
    denom = rho_e_u_e * C_M
    if denom <= 0.0:
        return 0.0
    return max(0.0, m_dot_char) / denom


def blowing_corrected_alpha(
    alpha_conv: float,
    m_dot_char: float,
    m_dot_pyro: float,
    rho_e_u_e: float,
    C_M: float,
    lambda_blowing: float = 0.5,
) -> float:
    """Blowing-corrected convective heat transfer coefficient [W/m^2/K].

    Returns alpha_conv unchanged if rho_e_u_e or C_M is zero.
    """
    denom = rho_e_u_e * C_M
    if denom <= 0.0:
        return alpha_conv
    B_total = (max(0.0, m_dot_char) + max(0.0, m_dot_pyro)) / denom
    return alpha_conv / (1.0 + lambda_blowing * B_total)

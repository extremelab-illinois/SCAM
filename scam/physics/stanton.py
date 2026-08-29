# SPDX-License-Identifier: MIT
"""Stanton-number corrections used by legacy CMA-style surface models."""

from __future__ import annotations

import math


def kays_blowing_factor(phi: float) -> float:
    """Return the Kays mass-injection correction ``Phi / (exp(Phi) - 1)``."""
    phi = max(0.0, float(phi))
    if phi < 1.0e-7:
        # Series avoids cancellation in exp(phi)-1.
        return 1.0 - 0.5 * phi + phi * phi / 12.0
    if phi > 700.0:
        return 0.0
    return phi / math.expm1(phi)


def solve_kays_blowing_factor(
    m_dot_char_unblown: float,
    m_dot_pyro: float,
    rho_ue_ch0: float,
    lambda_blowing: float,
) -> float:
    """Solve Amar's implicit Kays correction for the actual char mass flux.

    ``m_dot_pyro`` is an imposed injection flux.  The char flux is reduced by
    the same Stanton correction as heat and mass transfer:

    ``m_char = Omega_blw * m_char_unblown``.
    """
    if rho_ue_ch0 <= 0.0:
        return 1.0

    omega = 1.0
    for _ in range(50):
        m_total = max(0.0, m_dot_pyro) + omega * max(0.0, m_dot_char_unblown)
        phi = 2.0 * max(0.0, lambda_blowing) * m_total / rho_ue_ch0
        updated = kays_blowing_factor(phi)
        if abs(updated - omega) < 1.0e-12:
            return updated
        omega = updated
    return omega


def lees_blowing_factor(phi: float) -> float:
    """Return the Lees mass-injection correction ``log(1 + Phi) / Phi``.

    This is the formula used by PATO's ``constantLambdaBlowingCorrectionModel``.
    For small Phi uses the series expansion to avoid cancellation.
    """
    phi = max(0.0, float(phi))
    if phi < 1.0e-7:
        # ln(1+x)/x = 1 - x/2 + x²/3 - x³/4 + ...
        return 1.0 - 0.5 * phi + phi * phi / 3.0
    return math.log1p(phi) / phi


def solve_lees_blowing_factor(
    m_dot_char_unblown: float,
    m_dot_pyro: float,
    rho_ue_ch: float,
    lambda_blowing: float,
) -> float:
    """Solve PATO's Lees (log) blowing correction self-consistently.

    PATO ``constantLambdaBlowingCorrectionModel`` (lines 124–139)::

        Omega = log(1 + 2*lambda*(B'g_blown + B'c)) / (2*lambda*(B'g_blown + B'c))

    where ``B'g_blown = m_dot_g / (rhoUeCH * Omega)`` (blown basis) and
    ``B'c = m_dot_char_unblown / rhoUeCH`` (unblown, as returned by the B' table).
    Solved by fixed-point iteration; converges in < 10 steps at typical conditions.
    """
    if rho_ue_ch <= 0.0:
        return 1.0
    Bc  = max(0.0, m_dot_char_unblown) / rho_ue_ch
    mg  = max(0.0, m_dot_pyro)
    lam = max(0.0, lambda_blowing)
    omega = 1.0
    for _ in range(50):
        Bg_blown = mg / (rho_ue_ch * omega) if omega > 0 else 0.0
        phi = 2.0 * lam * (Bg_blown + Bc)
        updated = lees_blowing_factor(phi)
        if abs(updated - omega) < 1.0e-12:
            return updated
        omega = updated
    return omega


def eckert_reference_enthalpy(
    h_edge: float,
    h_wall: float,
    edge_velocity: float,
    recovery_factor: float,
) -> float:
    """Reference enthalpy from Amar et al. Eqs. (39)-(40) [J/kg]."""
    return (
        0.5 * (float(h_edge) + float(h_wall))
        + 0.11 * float(recovery_factor) * float(edge_velocity) ** 2
    )


def turbulent_eckert_wall_factor(
    mu_hot_ref: float,
    rho_hot_ref: float,
    mu_cold_ref: float,
    rho_cold_ref: float,
) -> float:
    """Turbulent hot-wall correction from Amar et al. Eq. (38)."""
    if min(mu_hot_ref, rho_hot_ref, mu_cold_ref, rho_cold_ref) <= 0.0:
        raise ValueError("Eckert reference properties must be positive")
    return (
        (float(mu_hot_ref) / float(mu_cold_ref)) ** 0.2
        * (float(rho_hot_ref) / float(rho_cold_ref)) ** 0.8
    )


def laminar_wall_factor(
    mu_hot_wall: float,
    rho_hot_wall: float,
    mu_cold_wall: float,
    rho_cold_wall: float,
) -> float:
    """Laminar/stagnation-point hot-wall correction, Amar et al. Eq. (37)."""
    if min(mu_hot_wall, rho_hot_wall, mu_cold_wall, rho_cold_wall) <= 0.0:
        raise ValueError("wall properties must be positive")
    return (
        float(rho_hot_wall) * float(mu_hot_wall)
        / (float(rho_cold_wall) * float(mu_cold_wall))
    ) ** 0.1

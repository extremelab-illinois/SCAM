# SPDX-License-Identifier: MIT
"""Finite-volume geometry helpers.

Functions for computing cell-face areas, inter-node spacings, and
interface conductances used in the tridiagonal coefficient assembly.
"""

from __future__ import annotations

import numpy as np


def face_area(A_left: float, A_right: float) -> float:
    """Area at the face between two adjacent cells (arithmetic mean)."""
    return 0.5 * (A_left + A_right)


def harmonic_conductivity(k_left: float, k_right: float) -> float:
    """Harmonic-mean conductivity at the interface between two cells.

    The harmonic mean gives the correct effective conductivity for two
    resistances in series (half-cell on each side):
        k_eff = 2 k_L k_R / (k_L + k_R)
    """
    if k_left + k_right == 0.0:
        return 0.0
    return 2.0 * k_left * k_right / (k_left + k_right)


def interface_conductance(
    k_left: float,
    delta_left: float,
    k_right: float,
    delta_right: float,
    A_face: float,
    contact_R: float = 0.0,
) -> float:
    """Effective conductance G [W/K] between two adjacent cells.

    The total resistance between cell centres is the sum of three
    resistances in series (half-cell left, contact, half-cell right):

        1/G = (delta_left/2) / (k_left * A) + contact_R / A + (delta_right/2) / (k_right * A)

    Parameters
    ----------
    k_left, k_right:
        Thermal conductivities of the left and right cells [W/m/K].
    delta_left, delta_right:
        Cell thicknesses [m].
    A_face:
        Face area [m^2].
    contact_R:
        Thermal contact resistance at the interface [m^2 K/W]; 0 = perfect contact.

    Returns
    -------
    float
        Conductance [W/K]; 0 if either conductivity is zero.
    """
    if k_left == 0.0 or k_right == 0.0:
        return 0.0
    R_left    = (delta_left  / 2.0) / (k_left  * A_face) if A_face > 0 else 1e30
    R_contact = contact_R / A_face if A_face > 0 else 0.0
    R_right   = (delta_right / 2.0) / (k_right * A_face) if A_face > 0 else 1e30
    R_total = R_left + R_contact + R_right
    return 1.0 / R_total if R_total > 0 else 0.0


def nodelet_cumulative_volume(
    y_nodelets: np.ndarray,
    delta_nodelets: np.ndarray,
    area_nodelets: np.ndarray,
) -> np.ndarray:
    """Cumulative volume from the first nodelet [m^3].

    Used as the interpolation coordinate for temperature on the subgrid
    (volume-coordinate interpolation, not distance, is correct for
    cylindrical geometry).

    Parameters
    ----------
    y_nodelets:
        Nodelet centre positions [m], shape (J,).
    delta_nodelets:
        Nodelet thicknesses [m], shape (J,).
    area_nodelets:
        Area A(y) at each nodelet centre [m^2], shape (J,).

    Returns
    -------
    np.ndarray
        Cumulative volume V[j] = sum_{i<=j} delta[i]*A[i], shape (J,).
    """
    dV = delta_nodelets * area_nodelets
    return np.cumsum(dV)

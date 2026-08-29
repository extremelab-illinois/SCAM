# SPDX-License-Identifier: MIT
"""Mixture enthalpy and charring energy quantities.

The in-depth energy equation includes a chemical energy source from
the density change:
    Q_decomp = -(drho/dt) * h_bar(T)

where h_bar is the enthalpy difference between virgin and char phases
(see physics/properties.py).

This module assembles these quantities for all nodes and provides
the decomposition enthalpy contribution from component-level h_decomp values.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scam.config.material import MaterialCard
from scam.physics.properties import h_bar as _h_bar


def decomposition_source(
    mat: MaterialCard,
    drho_dt_y: NDArray,  # (N_l,) [kg/m^3/s], negative when decomposing
    T_nodes: NDArray,    # (N_l,) [K]
) -> NDArray:
    """Volumetric decomposition heat source [W/m^3] for all nodes in a layer.

    Q_n = -drho_dt_y[n] * h_bar(T_n)  [W/m^3]

    A positive Q (source) corresponds to exothermic decomposition.
    For TACOT, decomposition is endothermic (h_decomp > 0), so Q < 0 (sink).
    """
    N_l = len(drho_dt_y)
    Q = np.zeros(N_l)
    for n in range(N_l):
        Q[n] = -float(drho_dt_y[n]) * _h_bar(mat, T_nodes[n])
    return Q



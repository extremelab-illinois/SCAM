# SPDX-License-Identifier: MIT
"""Vectorized interpolation between the thermal and nodelet grids."""

from __future__ import annotations

import numpy as np


def interpolate_layer_to_nodelets(
    T_left: np.ndarray,
    T_center: np.ndarray,
    T_right: np.ndarray,
    delta_nodelets: np.ndarray,
    area_nodes: np.ndarray,
) -> np.ndarray:
    """Interpolate temperatures to every nodelet in one material layer."""
    n_nodes, n_subcells = delta_nodelets.shape
    half = n_subcells // 2
    cumulative_volume = np.cumsum(
        delta_nodelets * area_nodes[:, None],
        axis=1,
    )
    cell_volume = cumulative_volume[:, -1]
    result = np.empty((n_nodes, n_subcells))

    if half > 0:
        midpoint_volume = cumulative_volume[:, half - 1]
        xi_left = np.full((n_nodes, half), 0.5)
        np.divide(
            cumulative_volume[:, :half],
            midpoint_volume[:, None],
            out=xi_left,
            where=midpoint_volume[:, None] > 0.0,
        )
        xi_left = np.clip(xi_left, 0.0, 1.0)
        result[:, :half] = (
            (1.0 - xi_left) * T_left[:, None]
            + xi_left * T_center[:, None]
        )
    else:
        midpoint_volume = np.zeros(n_nodes)

    right_volume = cell_volume - midpoint_volume
    xi_right = np.full((n_nodes, n_subcells - half), 0.5)
    np.divide(
        cumulative_volume[:, half:] - midpoint_volume[:, None],
        right_volume[:, None],
        out=xi_right,
        where=right_volume[:, None] > 0.0,
    )
    xi_right = np.clip(xi_right, 0.0, 1.0)
    result[:, half:] = (
        (1.0 - xi_right) * T_center[:, None]
        + xi_right * T_right[:, None]
    )
    return result
